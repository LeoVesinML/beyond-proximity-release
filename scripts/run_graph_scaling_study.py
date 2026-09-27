#!/usr/bin/env python3
"""Run an in-memory graph-vs-flat scaling stress test.

The study duplicates captured semantic indexes in memory to estimate query-time
cost growth as the number of indexed views increases. It is a stress test of the
current tree/search algorithm, not an accuracy benchmark and not a replacement
for Replica/ScanNet experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (  # noqa: E402
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    resolve_scene_dir,
    simulate_flat_search,
    simulate_graph_search,
)


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


def prefixed(prefix: str, value: str) -> str:
    return f"{prefix}__{value}"


def scaled_bundle(bundle: dict[str, Any], multiplier: int) -> dict[str, Any]:
    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")

    root_id = str(bundle["manifest"].get("root_node_id") or bundle["manifest"].get("root_id"))
    new_views: dict[str, dict[str, Any]] = {}
    new_tree: dict[str, dict[str, Any]] = {}
    new_root = copy.deepcopy(bundle["tree"][root_id])
    new_root["children_ids"] = []
    new_root["view_ids"] = []
    new_tree[root_id] = new_root

    for index in range(multiplier):
        prefix = f"scale{index + 1:03d}"
        view_map = {view_id: prefixed(prefix, view_id) for view_id in bundle["views"]}
        node_map = {
            node_id: (root_id if node_id == root_id else prefixed(prefix, node_id))
            for node_id in bundle["tree"]
        }
        for view_id, view in bundle["views"].items():
            copied = copy.deepcopy(view)
            copied["view_id"] = view_map[view_id]
            copied["scaling_source_view_id"] = view_id
            copied["scaling_duplicate_index"] = index + 1
            new_views[view_map[view_id]] = copied
        for node_id, node in bundle["tree"].items():
            if node_id == root_id:
                for child in node.get("children_ids", []):
                    child_id = str(child)
                    if child_id in node_map and node_map[child_id] not in new_root["children_ids"]:
                        new_root["children_ids"].append(node_map[child_id])
                continue
            copied = copy.deepcopy(node)
            copied["node_id"] = node_map[node_id]
            copied["id"] = node_map[node_id]
            copied["children_ids"] = [
                node_map[str(child)]
                for child in copied.get("children_ids", [])
                if str(child) in node_map and str(child) != root_id
            ]
            copied["view_ids"] = [
                view_map[str(view_id)]
                for view_id in copied.get("view_ids", [])
                if str(view_id) in view_map
            ]
            new_tree[node_map[node_id]] = copied

    manifest = copy.deepcopy(bundle["manifest"])
    manifest["root_node_id"] = root_id
    manifest["root_id"] = root_id
    return {
        **bundle,
        "views": new_views,
        "tree": new_tree,
        "manifest": manifest,
        "counts": {
            **bundle.get("counts", {}),
            "views": len(new_views),
            "tree_nodes": len(new_tree),
            "scaling_multiplier": multiplier,
        },
    }


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run_for_multiplier(
    bundles: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    multiplier: int,
) -> dict[str, Any]:
    scaled = {scene_id: scaled_bundle(bundle, multiplier) for scene_id, bundle in bundles.items()}
    rows = []
    for query in queries:
        bundle = scaled[str(query["scene_id"])]
        flat = simulate_flat_search(
            bundle,
            str(query["query"]),
            query_id=str(query["query_id"]),
            benchmark_scene_id=str(query["scene_id"]),
        )
        graph = simulate_graph_search(
            bundle,
            str(query["query"]),
            query_id=str(query["query_id"]),
            benchmark_scene_id=str(query["scene_id"]),
        )
        rows.append({
            "query_id": query.get("query_id"),
            "scene_id": query.get("scene_id"),
            "query_type": query.get("query_type"),
            "flat": flat.to_dict(),
            "graph": graph.to_dict(),
        })

    flat_views = average([float(row["flat"]["views_checked"]) for row in rows])
    graph_views = average([float(row["graph"]["views_checked"]) for row in rows])
    flat_tokens = average([float(row["flat"]["input_tokens"]) for row in rows])
    graph_tokens = average([float(row["graph"]["input_tokens"]) for row in rows])
    return {
        "multiplier": multiplier,
        "avg_indexed_views": average([float(len(bundle["views"])) for bundle in scaled.values()]),
        "avg_tree_nodes": average([float(len(bundle["tree"])) for bundle in scaled.values()]),
        "query_count": len(rows),
        "flat_avg_views_checked": flat_views,
        "graph_avg_views_checked": graph_views,
        "view_savings_pct": round(100 * (1 - graph_views / flat_views), 2) if flat_views else None,
        "flat_avg_input_tokens": flat_tokens,
        "graph_avg_input_tokens": graph_tokens,
        "token_savings_pct": round(100 * (1 - graph_tokens / flat_tokens), 2) if flat_tokens else None,
        "flat_avg_elapsed_ms": average([float(row["flat"]["elapsed_ms"]) for row in rows]),
        "graph_avg_elapsed_ms": average([float(row["graph"]["elapsed_ms"]) for row in rows]),
        "rows": rows,
    }


def write_outputs(run: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling_results.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "scaling_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "multiplier",
            "avg_indexed_views",
            "avg_tree_nodes",
            "flat_avg_views_checked",
            "graph_avg_views_checked",
            "view_savings_pct",
            "flat_avg_input_tokens",
            "graph_avg_input_tokens",
            "token_savings_pct",
            "flat_avg_elapsed_ms",
            "graph_avg_elapsed_ms",
        ])
        for item in run["summary"]:
            writer.writerow([
                item["multiplier"],
                item["avg_indexed_views"],
                item["avg_tree_nodes"],
                item["flat_avg_views_checked"],
                item["graph_avg_views_checked"],
                item["view_savings_pct"],
                item["flat_avg_input_tokens"],
                item["graph_avg_input_tokens"],
                item["token_savings_pct"],
                item["flat_avg_elapsed_ms"],
                item["graph_avg_elapsed_ms"],
            ])

    lines = [
        "# Graph Scaling Stress Study",
        "",
        f"- Generated: {run['generated_at']}",
        f"- Benchmark: `{run['benchmark_path']}`",
        f"- Query count per multiplier: {run['query_count_per_multiplier']}",
        "",
        "| Multiplier | Avg indexed views | Flat views | Graph views | View savings | Flat tokens | Graph tokens | Token savings |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in run["summary"]:
        lines.append(
            f"| {item['multiplier']} | {item['avg_indexed_views']} | "
            f"{item['flat_avg_views_checked']} | {item['graph_avg_views_checked']} | {item['view_savings_pct']}% | "
            f"{item['flat_avg_input_tokens']} | {item['graph_avg_input_tokens']} | {item['token_savings_pct']}% |"
        )
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- This duplicates existing semantic views in memory; it measures query-time scaling behavior, not new scene accuracy.",
        "- The current captured tree is shallow and object-heavy, so this is a conservative stress test rather than a deep hierarchy showcase.",
        "- Public Replica/ScanNet scaling remains required for a main-conference claim.",
    ])
    (out_dir / "scaling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", default="backend/data/scenes")
    parser.add_argument("--out", default="outputs/graph_vs_flat/scaling_stress_v1")
    parser.add_argument("--multipliers", default="1,2,5,10")
    parser.add_argument("--limit", type=int, default=60, help="Limit queries per multiplier; 0 means all")
    args = parser.parse_args()

    benchmark_path = (ROOT / args.benchmark).resolve()
    scene_root = (ROOT / args.scene_root).resolve()
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark_path))
    if args.limit:
        queries = queries[: args.limit]
    multipliers = [int(part.strip()) for part in args.multipliers.split(",") if part.strip()]
    bundles = {
        scene_id: load_scene_bundle(resolve_scene_dir(scene_id, scene_root, DEFAULT_SCENE_MAP))
        for scene_id in sorted({str(query["scene_id"]) for query in queries})
    }
    summary = [run_for_multiplier(bundles, queries, multiplier) for multiplier in multipliers]
    run = {
        "generated_at": utc_now(),
        "git_revision": git_revision(),
        "benchmark_path": str(benchmark_path),
        "scene_root": str(scene_root),
        "multipliers": multipliers,
        "query_count_per_multiplier": len(queries),
        "summary": [
            {key: value for key, value in item.items() if key != "rows"}
            for item in summary
        ],
        "details": summary,
    }
    out_dir = (ROOT / args.out).resolve()
    write_outputs(run, out_dir)
    print(f"Wrote graph scaling stress study to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
