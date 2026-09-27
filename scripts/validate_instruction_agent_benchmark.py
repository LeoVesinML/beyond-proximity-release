#!/usr/bin/env python3
"""Validate completeness and accounting of an instruction-agent benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "instruction_agent"
        / "five_scene_qwen25_05b_v1",
    )
    parser.add_argument("--expected-queries", type=int, default=150)
    args = parser.parse_args()
    output = args.output.resolve()
    errors: list[str] = []

    required = (
        "metrics_summary.json",
        "per_query_results.json",
        "per_query_metrics.csv",
        "run_config.json",
        "metrics_summary.md",
    )
    for name in required:
        if not (output / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1

    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "per_query_results.json").read_text(encoding="utf-8"))
    methods = tuple(summary.get("methods", {}))
    with (output / "per_query_metrics.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    if summary.get("status") != "complete":
        errors.append("summary status is not complete")
    if len(rows) != args.expected_queries:
        errors.append(f"expected {args.expected_queries} queries, found {len(rows)}")
    if not methods:
        errors.append("summary contains no methods")
    if len(csv_rows) != args.expected_queries * len(methods):
        errors.append("CSV row count does not equal queries x methods")
    if summary.get("quality_query_count") != 125 and args.expected_queries == 150:
        errors.append("quality denominator is not 125")
    if not summary.get("model", {}).get("instruction_tuned"):
        errors.append("model is not marked instruction-tuned")

    trace_count = 0
    for row in rows:
        for method in methods:
            value = row.get("methods", {}).get(method)
            if not isinstance(value, dict):
                errors.append(f"{row.get('query_id')}: missing {method}")
                continue
            accounting = value.get("accounting", {})
            for key in ("model_call_count", "input_tokens", "output_tokens", "total_tokens", "latency_ms"):
                number = accounting.get(key)
                if not isinstance(number, (int, float)) or number < 0:
                    errors.append(f"{row.get('query_id')}/{method}: invalid {key}")
            if accounting.get("total_tokens") != accounting.get("input_tokens", 0) + accounting.get("output_tokens", 0):
                errors.append(f"{row.get('query_id')}/{method}: token total mismatch")
            trace_path = ROOT / str(value.get("trace_path", ""))
            if not trace_path.is_file():
                errors.append(f"{row.get('query_id')}/{method}: missing trace")
                continue
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_count += 1
            calls = trace.get("calls", [])
            if len(calls) != accounting.get("model_call_count"):
                errors.append(f"{row.get('query_id')}/{method}: call count mismatch")
            for call in calls:
                if not call.get("system_prompt") or not call.get("user_prompts"):
                    errors.append(f"{row.get('query_id')}/{method}: prompt missing")
                if "raw_response" not in call:
                    errors.append(f"{row.get('query_id')}/{method}: response missing")

    report = {
        "schema_version": "semanticsplat.instruction_agent_validation.v1",
        "status": "passed" if not errors else "failed",
        "query_count": len(rows),
        "method_count": len(methods),
        "trace_count": trace_count,
        "csv_row_count": len(csv_rows),
        "errors": errors,
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
