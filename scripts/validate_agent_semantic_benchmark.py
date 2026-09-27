#!/usr/bin/env python3
"""Validate four-variant benchmark outputs and all semantic traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
METHODS = (
    "flat_lexical",
    "graph_lexical",
    "flat_semantic_embedding",
    "graph_semantic_embedding",
)
SEMANTIC_METHODS = METHODS[2:]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    output_dir: Path,
    *,
    expected_queries: int,
    expected_quality_queries: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "run_config.json",
        "metrics_summary.json",
        "metrics_summary.md",
        "per_query_results.json",
        "per_query_metrics.csv",
    )
    for name in required:
        if not (output_dir / name).exists():
            errors.append(f"Missing required file: {name}")
    if errors:
        return {"status": "failed", "errors": errors, "warnings": warnings}

    run_config = _read_json(output_dir / "run_config.json")
    summary = _read_json(output_dir / "metrics_summary.json")
    per_query = _read_json(output_dir / "per_query_results.json")
    rows = per_query.get("queries")
    if not isinstance(rows, list):
        errors.append("per_query_results.json does not contain a queries array")
        rows = []
    if len(rows) != expected_queries:
        errors.append(f"Expected {expected_queries} queries, found {len(rows)}")
    if run_config.get("query_count") != expected_queries:
        errors.append("run_config query_count mismatch")
    if summary.get("query_count") != expected_queries:
        errors.append("metrics_summary query_count mismatch")
    if summary.get("quality_query_count") != expected_quality_queries:
        errors.append("metrics_summary quality_query_count mismatch")
    if not summary.get("generated_at") or not run_config.get("generated_at"):
        errors.append("Generated timestamp is missing")
    source_hashes = run_config.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        errors.append("Source provenance hashes are missing")
    else:
        for value, expected_hash in source_hashes.items():
            path = ROOT / str(value)
            if not path.is_file():
                errors.append(f"Provenance source missing: {value}")
            elif _sha256(path) != expected_hash:
                errors.append(f"Provenance source changed after run: {value}")

    query_ids = [str(row.get("query_id", "")) for row in rows]
    if len(set(query_ids)) != len(query_ids):
        errors.append("Query IDs are not unique")
    eligible_count = sum(
        1
        for row in rows
        if row.get("methods", {})
        .get("flat_lexical", {})
        .get("quality", {})
        .get("eligible")
    )
    if eligible_count != expected_quality_queries:
        errors.append(
            f"Expected {expected_quality_queries} quality rows, found {eligible_count}"
        )

    schema = _read_json(ROOT / "docs" / "schemas" / "agent_query_trace.schema.json")
    validator = Draft202012Validator(schema)
    trace_count = 0
    for row in rows:
        methods = row.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            errors.append(f"{row.get('query_id')}: method set mismatch")
            continue
        for method in METHODS:
            item = methods[method]
            metrics = item.get("metrics", {})
            quality = item.get("quality", {})
            if quality.get("eligible") and quality.get("hit_at_1") is None:
                errors.append(f"{row.get('query_id')} {method}: missing eligible hit@1")
            if metrics.get("views_checked") is None or metrics.get("elapsed_ms") is None:
                errors.append(f"{row.get('query_id')} {method}: missing efficiency metric")
            if method not in SEMANTIC_METHODS:
                if metrics.get("model_call_count") != 0:
                    errors.append(f"{row.get('query_id')} {method}: lexical model calls nonzero")
                if item.get("trace_path") is not None:
                    errors.append(f"{row.get('query_id')} {method}: unexpected trace path")
                continue

            if int(metrics.get("model_call_count", 0)) <= 0:
                errors.append(f"{row.get('query_id')} {method}: no semantic model calls")
            if metrics.get("model_name") != "BAAI/bge-small-en-v1.5":
                errors.append(f"{row.get('query_id')} {method}: model provenance mismatch")
            if metrics.get("model_input_tokens") != metrics.get("input_tokens"):
                errors.append(f"{row.get('query_id')} {method}: model token mismatch")
            trace_value = item.get("trace_path")
            if not trace_value:
                errors.append(f"{row.get('query_id')} {method}: missing trace path")
                continue
            trace_path = ROOT / Path(str(trace_value))
            if not trace_path.exists():
                errors.append(f"{row.get('query_id')} {method}: trace file missing")
                continue
            trace = _read_json(trace_path)
            trace_count += 1
            for error in validator.iter_errors(trace):
                errors.append(
                    f"{row.get('query_id')} {method}: trace schema error at "
                    f"{'/'.join(str(item) for item in error.path)}: {error.message}"
                )
            accounting = trace.get("accounting", {})
            if accounting.get("input_tokens") != metrics.get("model_input_tokens"):
                errors.append(f"{row.get('query_id')} {method}: trace token mismatch")
            if accounting.get("output_tokens") != 0:
                errors.append(f"{row.get('query_id')} {method}: embedding output tokens not zero")
            if trace.get("status") != "completed":
                errors.append(f"{row.get('query_id')} {method}: trace not completed")
            result = trace.get("result", {})
            if result.get("selected_view_id") != metrics.get("selected_view_id"):
                errors.append(f"{row.get('query_id')} {method}: selected view mismatch")
            if result.get("matched_object") != metrics.get("matched_object"):
                errors.append(f"{row.get('query_id')} {method}: matched object mismatch")

    expected_traces = expected_queries * len(SEMANTIC_METHODS)
    if trace_count != expected_traces:
        errors.append(f"Expected {expected_traces} validated traces, found {trace_count}")
    actual_trace_files = list((output_dir / "traces").rglob("*.json"))
    if len(actual_trace_files) != expected_traces:
        errors.append(
            f"Expected {expected_traces} trace files on disk, found {len(actual_trace_files)}"
        )

    with (output_dir / "per_query_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv_rows = expected_queries * len(METHODS)
    if len(csv_rows) != expected_csv_rows:
        errors.append(
            f"Expected {expected_csv_rows} CSV method rows, found {len(csv_rows)}"
        )

    for method in METHODS:
        item = summary.get("methods", {}).get(method)
        if not isinstance(item, dict):
            errors.append(f"Summary missing method {method}")
            continue
        if item.get("query_count") != expected_queries:
            errors.append(f"Summary query count mismatch for {method}")
        if item.get("quality", {}).get("denominator") != expected_quality_queries:
            errors.append(f"Summary quality denominator mismatch for {method}")
    if summary.get("negative_quality_status") != (
        "not_reported_no_independent_negative_gt"
    ):
        errors.append("Negative-query truth boundary is missing")
    if len(summary.get("by_query_type", {})) != 6:
        errors.append("Expected six query-type breakdowns")
    if len(summary.get("by_scene", {})) != 5:
        errors.append("Expected five scene breakdowns")

    return {
        "schema_version": "semanticsplat.agent_semantic_validation.v1",
        "status": "passed" if not errors else "failed",
        "output_dir": str(output_dir),
        "query_count": len(rows),
        "quality_query_count": eligible_count,
        "method_count": len(METHODS),
        "trace_count": trace_count,
        "csv_row_count": len(csv_rows),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=150)
    parser.add_argument("--expected-quality-queries", type=int, default=125)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    report = validate(
        output_dir,
        expected_queries=args.expected_queries,
        expected_quality_queries=args.expected_quality_queries,
    )
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
