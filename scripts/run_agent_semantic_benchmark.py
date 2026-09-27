#!/usr/bin/env python3
"""Run the four controlled lexical/semantic graph-vs-flat variants."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import (
    DEFAULT_LOCAL_MODEL,
    METHOD_PROTOCOL_ID,
    FastEmbedSemanticEncoder,
    semantic_flat_search,
    semantic_graph_search,
    write_agent_trace,
)
from backend.query.graph_vs_flat import (
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    query_has_view_gt,
    resolve_scene_dir,
    simulate_flat_search,
    simulate_graph_search,
)


METHODS = (
    "flat_lexical",
    "graph_lexical",
    "flat_semantic_embedding",
    "graph_semantic_embedding",
)
BOOTSTRAP_SEED = 20260723
PROVENANCE_FILES = (
    "backend/query/agent_protocol.py",
    "backend/query/graph_vs_flat.py",
    "backend/mcp/tools/agent_trace_tools.py",
    "backend/mcp/server.py",
    "scripts/run_agent_semantic_benchmark.py",
    "scripts/validate_agent_semantic_benchmark.py",
    "docs/schemas/agent_query_trace.schema.json",
    "docs/project/agent_mcp_method_contract.md",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(_sha256(item).encode("ascii"))
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_dirty() -> bool | None:
    try:
        value = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(value.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes() -> dict[str, str]:
    return {
        value: _sha256(ROOT / value)
        for value in PROVENANCE_FILES
        if (ROOT / value).is_file()
    }


def _safe_stem(scene_id: str, query_id: str) -> str:
    raw = f"{scene_id}__{query_id}"
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def _record_rss(metrics: Any, process: psutil.Process) -> None:
    metrics.process_rss_mb_observed = round(
        process.memory_info().rss / (1024 * 1024),
        6,
    )


def _hit_at_k(ranked: list[str], expected: list[str], k: int) -> bool:
    wanted = set(expected)
    return any(view_id in wanted for view_id in ranked[:k])


def _reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    wanted = set(expected)
    for index, view_id in enumerate(ranked, start=1):
        if view_id in wanted:
            return 1.0 / index
    return 0.0


def _quality(metrics: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    eligible = query_has_view_gt(query)
    expected = [str(item) for item in query.get("expected_view_ids", []) if item]
    ranked = [
        str(item)
        for item in (
            metrics.get("all_ranked_view_ids")
            or metrics.get("ranked_view_ids")
            or []
        )
        if item
    ]
    expected_object_labels = {
        str(item).strip().lower()
        for item in query.get("expected_object_labels", [])
        if str(item).strip()
    }
    matched_object = str(metrics.get("matched_object") or "").strip().lower()
    zone_evidence = query.get("expected_zone_view_evidence")
    expected_zone_views = {
        str(view_id)
        for values in zone_evidence.values()
        for view_id in values
        if view_id
    } if isinstance(zone_evidence, dict) else set()
    selected_view_id = str(metrics.get("selected_view_id") or "")
    object_eligible = eligible and bool(expected_object_labels)
    zone_eligible = eligible and bool(expected_zone_views)
    return {
        "eligible": eligible,
        "expected_view_ids": expected,
        "expected_object_labels": sorted(expected_object_labels),
        "ranked_view_ids": ranked,
        "hit_at_1": _hit_at_k(ranked, expected, 1) if eligible else None,
        "hit_at_3": _hit_at_k(ranked, expected, 3) if eligible else None,
        "reciprocal_rank": _reciprocal_rank(ranked, expected) if eligible else None,
        "expected_object_hit": (
            matched_object in expected_object_labels if object_eligible else None
        ),
        "expected_zone_view_hit": (
            selected_view_id in expected_zone_views if zone_eligible else None
        ),
        "wrong_zone_view": (
            selected_view_id not in expected_zone_views if zone_eligible else None
        ),
        "wrong_room": None,
        "wrong_room_status": "unavailable_no_independent_room_gt",
    }


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    estimates = array[indices].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def _average(rows: list[dict[str, Any]], method: str, field: str) -> float:
    values = [
        float(value)
        for row in rows
        if (value := row["methods"][method]["metrics"].get(field)) is not None
    ]
    return round(float(np.mean(values)), 6) if values else 0.0


def _mean_optional(values: Iterable[bool | float | None]) -> float | None:
    measured = [float(value) for value in values if value is not None]
    return round(float(np.mean(measured)), 6) if measured else None


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    eligible = [
        row["methods"][method]["quality"]
        for row in rows
        if row["methods"][method]["quality"]["eligible"]
    ]
    first_metrics = rows[0]["methods"][method]["metrics"] if rows else {}
    quality = {
        "denominator": len(eligible),
        "hit_at_1": round(
            float(np.mean([float(item["hit_at_1"]) for item in eligible])), 6
        )
        if eligible
        else None,
        "hit_at_3": round(
            float(np.mean([float(item["hit_at_3"]) for item in eligible])), 6
        )
        if eligible
        else None,
        "mrr": round(
            float(np.mean([float(item["reciprocal_rank"]) for item in eligible])), 6
        )
        if eligible
        else None,
        "expected_object_hit": _mean_optional(
            item["expected_object_hit"] for item in eligible
        ),
        "expected_object_denominator": len(
            [item for item in eligible if item["expected_object_hit"] is not None]
        ),
        "expected_zone_view_hit": _mean_optional(
            item["expected_zone_view_hit"] for item in eligible
        ),
        "expected_zone_denominator": len(
            [item for item in eligible if item["expected_zone_view_hit"] is not None]
        ),
        "wrong_zone_view_rate": _mean_optional(
            item["wrong_zone_view"] for item in eligible
        ),
        "wrong_room_rate": None,
        "wrong_room_status": "unavailable_no_independent_room_gt",
    }
    return {
        "query_count": len(rows),
        "quality": quality,
        "efficiency": {
            field: _average(rows, method, field)
            for field in (
                "views_checked",
                "nodes_visited",
                "semantic_entries_scanned",
                "context_chars",
                "input_tokens",
                "elapsed_ms",
                "model_call_count",
                "model_inference_ms",
                "process_rss_mb_observed",
            )
        },
        "accounting": {
            "model_backend": first_metrics.get("model_backend"),
            "model_name": first_metrics.get("model_name"),
            "token_method": (
                first_metrics.get("model_token_method")
                if first_metrics.get("model_backend") != "none"
                else "tiktoken:cl100k_base_serialized_context"
            ),
            "total_input_tokens": sum(
                int(row["methods"][method]["metrics"]["input_tokens"]) for row in rows
            ),
            "total_model_calls": sum(
                int(row["methods"][method]["metrics"]["model_call_count"])
                for row in rows
            ),
        },
    }


def _group_summaries(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(field, "unknown")) for row in rows})
    return {
        value: {
            method: _method_summary(
                [row for row in rows if str(row.get(field, "unknown")) == value],
                method,
            )
            for method in METHODS
        }
        for value in values
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    *,
    graph_method: str,
    flat_method: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    efficiency: dict[str, Any] = {}
    for offset, field in enumerate(
        ("views_checked", "context_chars", "input_tokens", "elapsed_ms")
    ):
        graph = np.asarray(
            [float(row["methods"][graph_method]["metrics"][field]) for row in rows],
            dtype=np.float64,
        )
        flat = np.asarray(
            [float(row["methods"][flat_method]["metrics"][field]) for row in rows],
            dtype=np.float64,
        )
        valid = flat > 0
        reductions = 100.0 * (1.0 - graph[valid] / flat[valid])
        ratio_reduction = (
            100.0 * (1.0 - float(graph.mean()) / float(flat.mean()))
            if float(flat.mean()) > 0
            else None
        )
        efficiency[field] = {
            "graph_mean": round(float(graph.mean()), 6),
            "flat_mean": round(float(flat.mean()), 6),
            "ratio_of_means_reduction_pct": (
                round(ratio_reduction, 6) if ratio_reduction is not None else None
            ),
            "mean_per_query_reduction_pct": (
                round(float(reductions.mean()), 6) if reductions.size else None
            ),
            "mean_per_query_reduction_ci95": _bootstrap_mean_ci(
                reductions,
                samples=samples,
                seed=seed + offset,
            ),
        }

    quality_rows = [
        row
        for row in rows
        if row["methods"][graph_method]["quality"]["eligible"]
        and row["methods"][flat_method]["quality"]["eligible"]
    ]
    quality: dict[str, Any] = {"denominator": len(quality_rows)}
    for offset, field in enumerate(
        (
            "hit_at_1",
            "hit_at_3",
            "reciprocal_rank",
            "expected_object_hit",
            "expected_zone_view_hit",
        ),
        start=20,
    ):
        differences = [
            float(row["methods"][graph_method]["quality"][field])
            - float(row["methods"][flat_method]["quality"][field])
            for row in quality_rows
            if row["methods"][graph_method]["quality"][field] is not None
            and row["methods"][flat_method]["quality"][field] is not None
        ]
        quality[field] = {
            "denominator": len(differences),
            "graph_minus_flat_mean": round(float(np.mean(differences)), 6)
            if differences
            else None,
            "graph_minus_flat_ci95": _bootstrap_mean_ci(
                differences,
                samples=samples,
                seed=seed + offset,
            ),
        }
    return {
        "graph_method": graph_method,
        "flat_method": flat_method,
        "efficiency": efficiency,
        "quality": quality,
    }


def _write_csv(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "scene_id",
        "query_type",
        "method",
        "eligible",
        "hit_at_1",
        "hit_at_3",
        "reciprocal_rank",
        "expected_object_hit",
        "expected_zone_view_hit",
        "wrong_zone_view",
        "views_checked",
        "nodes_visited",
        "semantic_entries_scanned",
        "context_chars",
        "input_tokens",
        "elapsed_ms",
        "model_call_count",
        "model_inference_ms",
        "process_rss_mb_observed",
        "selected_view_id",
        "matched_object",
    ]
    with (out_dir / "per_query_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                item = row["methods"][method]
                metrics = item["metrics"]
                quality = item["quality"]
                writer.writerow(
                    {
                        "query_id": row["query_id"],
                        "scene_id": row["scene_id"],
                        "query_type": row["query_type"],
                        "method": method,
                        "eligible": quality["eligible"],
                        "hit_at_1": quality["hit_at_1"],
                        "hit_at_3": quality["hit_at_3"],
                        "reciprocal_rank": quality["reciprocal_rank"],
                        "expected_object_hit": quality["expected_object_hit"],
                        "expected_zone_view_hit": quality["expected_zone_view_hit"],
                        "wrong_zone_view": quality["wrong_zone_view"],
                        "views_checked": metrics["views_checked"],
                        "nodes_visited": metrics["nodes_visited"],
                        "semantic_entries_scanned": metrics[
                            "semantic_entries_scanned"
                        ],
                        "context_chars": metrics["context_chars"],
                        "input_tokens": metrics["input_tokens"],
                        "elapsed_ms": metrics["elapsed_ms"],
                        "model_call_count": metrics["model_call_count"],
                        "model_inference_ms": metrics["model_inference_ms"],
                        "process_rss_mb_observed": metrics[
                            "process_rss_mb_observed"
                        ],
                        "selected_view_id": metrics["selected_view_id"],
                        "matched_object": metrics["matched_object"],
                    }
                )


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Agent-semantic four-variant benchmark",
        "",
        f"- Protocol: `{summary['method_protocol_id']}`",
        f"- Queries: {summary['query_count']}",
        f"- Quality-eligible queries: {summary['quality_query_count']}",
        f"- Local model: `{summary['local_model']['model_name']}`",
        f"- Bootstrap samples: {summary['bootstrap']['samples']}",
        "",
        "## Method results",
        "",
        "| Method | hit@1 | hit@3 | MRR | Object hit | Zone-view hit | Views | Input tokens | Runtime ms | Model calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["methods"][method]
        quality = item["quality"]
        efficiency = item["efficiency"]
        lines.append(
            f"| `{method}` | {quality['hit_at_1']} | {quality['hit_at_3']} | "
            f"{quality['mrr']} | {quality['expected_object_hit']} | "
            f"{quality['expected_zone_view_hit']} | {efficiency['views_checked']} | "
            f"{efficiency['input_tokens']} | {efficiency['elapsed_ms']} | "
            f"{efficiency['model_call_count']} |"
        )
    lines.extend(["", "## Paired graph-versus-flat comparisons", ""])
    for key in ("lexical", "semantic"):
        item = summary["comparisons"][key]
        views = item["efficiency"]["views_checked"]
        tokens = item["efficiency"]["input_tokens"]
        hit1 = item["quality"]["hit_at_1"]
        hit3 = item["quality"]["hit_at_3"]
        lines.extend(
            [
                f"### {key.title()} scorer",
                "",
                f"- Views reduction, ratio of means: {views['ratio_of_means_reduction_pct']}%",
                f"- Views reduction, mean per query: {views['mean_per_query_reduction_pct']}% "
                f"(95% CI {views['mean_per_query_reduction_ci95']})",
                f"- Input-token reduction, ratio of means: {tokens['ratio_of_means_reduction_pct']}%",
                f"- Input-token reduction, mean per query: {tokens['mean_per_query_reduction_pct']}% "
                f"(95% CI {tokens['mean_per_query_reduction_ci95']})",
                f"- hit@1 graph-minus-flat: {hit1['graph_minus_flat_mean']} "
                f"(95% CI {hit1['graph_minus_flat_ci95']})",
                f"- hit@3 graph-minus-flat: {hit3['graph_minus_flat_mean']} "
                f"(95% CI {hit3['graph_minus_flat_ci95']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The semantic variants use a pinned neural embedding encoder. They are not "
            "Cursor Agent or LLM executions. Cursor uses the same MCP traversal and trace "
            "contract, but its quality requires separately saved Cursor traces.",
            "",
            "Negative-query correctness is not reported because independent negative GT "
            "is unavailable.",
            "",
            "Wrong-room rate is not reported because this benchmark provides manual zone "
            "references, not independent room-level GT. Zone-view error is reported.",
        ]
    )
    (out_dir / "metrics_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = _utc_now()
    benchmark_path = args.benchmark.resolve()
    scene_root = args.scene_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark_path))
    if args.limit is not None:
        queries = queries[: args.limit]

    model_started = time.perf_counter()
    encoder = FastEmbedSemanticEncoder(
        args.model,
        args.model_cache,
        threads=args.threads,
    )
    model_load_seconds = time.perf_counter() - model_started
    warmup_started = time.perf_counter()
    encoder.encode(["semantic traversal warmup"])
    warmup_seconds = time.perf_counter() - warmup_started
    process = psutil.Process()
    rss_after_model_load_mb = round(process.memory_info().rss / (1024 * 1024), 6)

    bundles: dict[str, dict[str, Any]] = {}
    bundle_load_seconds = 0.0
    rows: list[dict[str, Any]] = []
    trace_root = out_dir / "traces"

    for index, query in enumerate(queries):
        benchmark_scene_id = str(query["scene_id"])
        if benchmark_scene_id not in bundles:
            scene_dir = resolve_scene_dir(benchmark_scene_id, scene_root)
            started = time.perf_counter()
            bundles[benchmark_scene_id] = load_scene_bundle(scene_dir)
            bundle_load_seconds += time.perf_counter() - started
        bundle = bundles[benchmark_scene_id]
        query_text = str(query["query"])
        query_id = str(query["query_id"])

        flat_lexical = simulate_flat_search(
            bundle,
            query_text,
            query_id=query_id,
            benchmark_scene_id=benchmark_scene_id,
        )
        _record_rss(flat_lexical, process)
        graph_lexical = simulate_graph_search(
            bundle,
            query_text,
            query_id=query_id,
            benchmark_scene_id=benchmark_scene_id,
        )
        _record_rss(graph_lexical, process)
        flat_lexical.model_token_method = "tiktoken:cl100k_base_serialized_context"
        graph_lexical.model_token_method = "tiktoken:cl100k_base_serialized_context"

        semantic_kwargs = {
            "query_id": query_id,
            "benchmark_scene_id": benchmark_scene_id,
        }
        if index % 2 == 0:
            flat_semantic, flat_trace = semantic_flat_search(
                bundle, query_text, encoder, **semantic_kwargs
            )
            _record_rss(flat_semantic, process)
            graph_semantic, graph_trace = semantic_graph_search(
                bundle,
                query_text,
                encoder,
                keep_margin=args.keep_margin,
                minimum_branch_score=args.minimum_branch_score,
                max_children=args.max_children,
                fallback_child_limit=args.fallback_child_limit,
                **semantic_kwargs,
            )
            _record_rss(graph_semantic, process)
        else:
            graph_semantic, graph_trace = semantic_graph_search(
                bundle,
                query_text,
                encoder,
                keep_margin=args.keep_margin,
                minimum_branch_score=args.minimum_branch_score,
                max_children=args.max_children,
                fallback_child_limit=args.fallback_child_limit,
                **semantic_kwargs,
            )
            _record_rss(graph_semantic, process)
            flat_semantic, flat_trace = semantic_flat_search(
                bundle, query_text, encoder, **semantic_kwargs
            )
            _record_rss(flat_semantic, process)

        stem = _safe_stem(benchmark_scene_id, query_id)
        flat_trace_path = trace_root / "flat_semantic_embedding" / f"{stem}.json"
        graph_trace_path = trace_root / "graph_semantic_embedding" / f"{stem}.json"
        write_agent_trace(flat_trace, flat_trace_path)
        write_agent_trace(graph_trace, graph_trace_path)

        method_metrics = {
            "flat_lexical": flat_lexical.to_dict(),
            "graph_lexical": graph_lexical.to_dict(),
            "flat_semantic_embedding": flat_semantic.to_dict(),
            "graph_semantic_embedding": graph_semantic.to_dict(),
        }
        rows.append(
            {
                "query_id": query_id,
                "scene_id": benchmark_scene_id,
                "captured_scene_id": str(bundle["scene_id"]),
                "query": query_text,
                "query_type": query.get("query_type"),
                "verification_status": query.get("verification_status"),
                "methods": {
                    method: {
                        "metrics": metrics,
                        "quality": _quality(metrics, query),
                        "trace_path": (
                            str(flat_trace_path.relative_to(ROOT))
                            if method == "flat_semantic_embedding"
                            else str(graph_trace_path.relative_to(ROOT))
                            if method == "graph_semantic_embedding"
                            else None
                        ),
                    }
                    for method, metrics in method_metrics.items()
                },
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(queries):
            print(f"Completed {index + 1}/{len(queries)} queries", flush=True)

    methods = {method: _method_summary(rows, method) for method in METHODS}
    quality_query_count = sum(
        1 for row in rows if row["methods"]["flat_lexical"]["quality"]["eligible"]
    )
    summary = {
        "schema_version": "semanticsplat.agent_semantic_benchmark_summary.v1",
        "generated_at": generated_at,
        "method_protocol_id": METHOD_PROTOCOL_ID,
        "query_count": len(rows),
        "quality_query_count": quality_query_count,
        "negative_quality_status": "not_reported_no_independent_negative_gt",
        "methods": methods,
        "by_query_type": _group_summaries(rows, "query_type"),
        "by_scene": _group_summaries(rows, "scene_id"),
        "comparisons": {
            "lexical": _paired_comparison(
                rows,
                graph_method="graph_lexical",
                flat_method="flat_lexical",
                samples=args.bootstrap_samples,
                seed=BOOTSTRAP_SEED,
            ),
            "semantic": _paired_comparison(
                rows,
                graph_method="graph_semantic_embedding",
                flat_method="flat_semantic_embedding",
                samples=args.bootstrap_samples,
                seed=BOOTSTRAP_SEED + 100,
            ),
        },
        "local_model": {
            "model_name": encoder.model_name,
            "backend": encoder.backend_name,
            "token_method": encoder.token_method,
            "cache_dir": str(encoder.cache_dir),
            "cache_fingerprint_sha256": _directory_fingerprint(encoder.cache_dir),
            "load_seconds": round(model_load_seconds, 6),
            "warmup_seconds": round(warmup_seconds, 6),
            "threads": args.threads,
            "device": "cpu",
            "process_rss_after_model_load_mb": rss_after_model_load_mb,
            "peak_observed_process_rss_mb": max(
                float(row["methods"][method]["metrics"]["process_rss_mb_observed"])
                for row in rows
                for method in METHODS
            )
            if rows
            else rss_after_model_load_mb,
            "memory_measurement": (
                "psutil process RSS observed after each method; not per-method attribution"
            ),
        },
        "construction_and_load_cost": {
            "scene_bundle_load_seconds": round(bundle_load_seconds, 6),
            "model_load_seconds": round(model_load_seconds, 6),
            "model_warmup_seconds": round(warmup_seconds, 6),
            "excluded_from_per_query_metrics": True,
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
            "method": "paired_query_resampling_percentile_95",
        },
    }
    run_config = {
        "schema_version": "semanticsplat.agent_semantic_benchmark_run.v1",
        "generated_at": generated_at,
        "method_protocol_id": METHOD_PROTOCOL_ID,
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "source_sha256": _source_hashes(),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "scene_root": str(scene_root),
        "scene_map": DEFAULT_SCENE_MAP,
        "query_count": len(rows),
        "method_order": list(METHODS),
        "semantic_execution_order": "alternating_flat_first_and_graph_first",
        "semantic_parameters": {
            "keep_margin": args.keep_margin,
            "minimum_branch_score": args.minimum_branch_score,
            "max_children": args.max_children,
            "fallback_child_limit": args.fallback_child_limit,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "fastembed": _package_version("fastembed"),
            "onnxruntime": _package_version("onnxruntime"),
            "psutil": _package_version("psutil"),
        },
        "method_contract": "docs/project/agent_mcp_method_contract.md",
        "trace_schema": "docs/schemas/agent_query_trace.schema.json",
    }
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(out_dir / "metrics_summary.json", summary)
    _write_json(
        out_dir / "per_query_results.json",
        {
            "schema_version": "semanticsplat.agent_semantic_per_query.v1",
            "method_protocol_id": METHOD_PROTOCOL_ID,
            "queries": rows,
        },
    )
    _write_csv(out_dir, rows)
    _write_report(out_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "backend" / "data" / "scenes",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=ROOT / ".cache" / "fastembed",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-margin", type=float, default=0.08)
    parser.add_argument("--minimum-branch-score", type=float, default=0.20)
    parser.add_argument("--max-children", type=int, default=3)
    parser.add_argument("--fallback-child-limit", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    summary = run(args)
    print(
        f"Wrote {summary['query_count']} four-variant query results to {args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
