#!/usr/bin/env python3
"""Run graph-pruning and affordance ablations on the captured benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (  # noqa: E402
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    hit_at_k,
    load_benchmark_queries,
    load_scene_bundle,
    query_has_view_gt,
    resolve_scene_dir,
    simulate_flat_search,
    simulate_graph_search,
)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    run: Callable[[dict[str, Any], dict[str, Any]], Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ranked_views(result: Any) -> list[str]:
    ranked = list(getattr(result, "ranked_view_ids", []) or [])
    selected = getattr(result, "selected_view_id", None)
    if selected and selected not in ranked:
        ranked.insert(0, selected)
    return ranked


def quality(result: Any, query: dict[str, Any]) -> dict[str, Any]:
    expected = [str(v) for v in query.get("expected_view_ids", []) if v]
    gt_available = query_has_view_gt(query)
    views = ranked_views(result)
    return {
        "gt_available": gt_available,
        "hit_at_1": hit_at_k(views, expected, 1) if gt_available else None,
        "hit_at_3": hit_at_k(views, expected, 3) if gt_available else None,
        "ranked_view_ids": views,
    }


def variants() -> list[Variant]:
    return [
        Variant(
            name="flat_lexical",
            description="Exhaustive lexical scan over every view.",
            run=lambda bundle, query: simulate_flat_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                use_affordance=False,
            ),
        ),
        Variant(
            name="flat_affordance",
            description="Exhaustive scan with deterministic affordance expansion.",
            run=lambda bundle, query: simulate_flat_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                use_affordance=True,
            ),
        ),
        Variant(
            name="graph_tight",
            description="Graph traversal with strict pruning: keep only max-score branches.",
            run=lambda bundle, query: simulate_graph_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                branch_keep_ratio=1.0,
                fallback_child_limit=1,
                mode_suffix="_tight",
            ),
        ),
        Variant(
            name="graph_default",
            description="Graph traversal with the paper default pruning policy.",
            run=lambda bundle, query: simulate_graph_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                branch_keep_ratio=0.5,
                fallback_child_limit=3,
            ),
        ),
        Variant(
            name="graph_broad",
            description="Graph traversal with broad recall-oriented branch keeping.",
            run=lambda bundle, query: simulate_graph_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                branch_keep_ratio=0.0,
                fallback_child_limit=9999,
                mode_suffix="_broad",
            ),
        ),
        Variant(
            name="graph_affordance",
            description="Default graph traversal with deterministic affordance expansion.",
            run=lambda bundle, query: simulate_graph_search(
                bundle,
                str(query["query"]),
                query_id=str(query["query_id"]),
                benchmark_scene_id=str(query["scene_id"]),
                use_affordance=True,
                branch_keep_ratio=0.5,
                fallback_child_limit=3,
            ),
        ),
    ]


def average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: list[dict[str, Any]], variant_defs: list[Variant]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "query_count": len(rows),
        "gt_query_count": sum(1 for row in rows if row["gt_available"]),
        "variants": {},
        "by_query_type": {},
    }
    flat_views = [row["variants"]["flat_lexical"]["views_checked"] for row in rows]
    flat_tokens = [row["variants"]["flat_lexical"]["input_tokens"] for row in rows]
    flat_view_avg = average([float(v) for v in flat_views]) or 0.0
    flat_token_avg = average([float(v) for v in flat_tokens]) or 0.0

    for variant in variant_defs:
        name = variant.name
        entries = [row["variants"][name] for row in rows]
        hit1 = [entry["quality"]["hit_at_1"] for entry in entries if entry["quality"]["hit_at_1"] is not None]
        hit3 = [entry["quality"]["hit_at_3"] for entry in entries if entry["quality"]["hit_at_3"] is not None]
        views = average([float(entry["views_checked"]) for entry in entries]) or 0.0
        tokens = average([float(entry["input_tokens"]) for entry in entries]) or 0.0
        summary["variants"][name] = {
            "description": variant.description,
            "avg_views_checked": views,
            "avg_input_tokens": tokens,
            "avg_context_chars": average([float(entry["context_chars"]) for entry in entries]),
            "avg_elapsed_ms": average([float(entry["elapsed_ms"]) for entry in entries]),
            "avg_nodes_visited": average([float(entry["nodes_visited"]) for entry in entries]),
            "view_savings_vs_flat_pct": round(100 * (1 - views / flat_view_avg), 2) if flat_view_avg else None,
            "token_savings_vs_flat_pct": round(100 * (1 - tokens / flat_token_avg), 2) if flat_token_avg else None,
            "hit_at_1": average([float(v) for v in hit1]),
            "hit_at_3": average([float(v) for v in hit3]),
        }

    query_types = sorted({str(row["query_type"]) for row in rows})
    for query_type in query_types:
        subset = [row for row in rows if str(row["query_type"]) == query_type]
        summary["by_query_type"][query_type] = {}
        for variant in variant_defs:
            entries = [row["variants"][variant.name] for row in subset]
            summary["by_query_type"][query_type][variant.name] = {
                "avg_views_checked": average([float(entry["views_checked"]) for entry in entries]),
                "avg_input_tokens": average([float(entry["input_tokens"]) for entry in entries]),
            }
    return summary


def write_outputs(run: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_results.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = run["summary"]
    (out_dir / "ablation_summary.json").write_text(
        json.dumps({"generated_at": run["generated_at"], "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "variant",
            "avg_views_checked",
            "view_savings_vs_flat_pct",
            "avg_input_tokens",
            "token_savings_vs_flat_pct",
            "hit_at_1",
            "hit_at_3",
        ])
        for name, data in summary["variants"].items():
            writer.writerow([
                name,
                data["avg_views_checked"],
                data["view_savings_vs_flat_pct"],
                data["avg_input_tokens"],
                data["token_savings_vs_flat_pct"],
                data["hit_at_1"],
                data["hit_at_3"],
            ])

    lines = [
        "# Graph Ablation Study",
        "",
        f"- Generated: {run['generated_at']}",
        f"- Benchmark: `{run['benchmark_path']}`",
        f"- Queries: {summary['query_count']}",
        f"- Queries with verified view labels: {summary['gt_query_count']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Views | View savings vs flat | Tokens | Token savings vs flat | hit@1 | hit@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary["variants"].items():
        lines.append(
            f"| `{name}` | {data['avg_views_checked']} | {data['view_savings_vs_flat_pct']}% | "
            f"{data['avg_input_tokens']} | {data['token_savings_vs_flat_pct']}% | "
            f"{data['hit_at_1']} | {data['hit_at_3']} |"
        )
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- These are deterministic lexical/stub ablations, not live VLM results.",
        "- Manual view labels are reference labels, not independent dataset ground truth.",
        "- `graph_broad` is a recall-oriented stress setting; `graph_tight` is a cost-oriented stress setting.",
    ])
    (out_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", default="backend/data/scenes")
    parser.add_argument("--out", default="outputs/graph_vs_flat/ablations_v1")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of queries")
    args = parser.parse_args()

    benchmark_path = (ROOT / args.benchmark).resolve()
    scene_root = (ROOT / args.scene_root).resolve()
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark_path))
    if args.limit:
        queries = queries[: args.limit]
    variant_defs = variants()
    bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for query in queries:
        scene_id = str(query["scene_id"])
        if scene_id not in bundles:
            bundles[scene_id] = load_scene_bundle(resolve_scene_dir(scene_id, scene_root, DEFAULT_SCENE_MAP))
        bundle = bundles[scene_id]
        variant_rows: dict[str, Any] = {}
        for variant in variant_defs:
            result = variant.run(bundle, query)
            data = result.to_dict()
            data["quality"] = quality(result, query)
            variant_rows[variant.name] = data
        rows.append({
            "query_id": query.get("query_id"),
            "scene_id": scene_id,
            "query": query.get("query"),
            "query_type": query.get("query_type"),
            "gt_available": query_has_view_gt(query),
            "variants": variant_rows,
        })

    run = {
        "generated_at": utc_now(),
        "git_revision": git_revision(),
        "benchmark_path": str(benchmark_path),
        "scene_root": str(scene_root),
        "scene_map": DEFAULT_SCENE_MAP,
        "queries": rows,
        "summary": summarize(rows, variant_defs),
    }
    out_dir = (ROOT / args.out).resolve()
    write_outputs(run, out_dir)
    print(f"Wrote graph ablation study to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
