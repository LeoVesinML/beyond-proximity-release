#!/usr/bin/env python3
"""Measure graph and flat retrieval as real captured-scene pools grow."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    simulate_flat_search,
    simulate_graph_search,
)


def _prefixed_bundle(scene_ids: list[str], scene_root: Path) -> dict[str, Any]:
    views: dict[str, Any] = {}
    tree: dict[str, Any] = {}
    scene_roots: list[str] = []
    for benchmark_id in scene_ids:
        capture_id = DEFAULT_SCENE_MAP[benchmark_id]
        source = load_scene_bundle(scene_root / capture_id)
        prefix = f"{benchmark_id}::"
        root_id = str(source["manifest"].get("root_node_id") or source["manifest"].get("root_id"))
        for view_id, view in source["views"].items():
            views[prefix + view_id] = view
        for node_id, node in source["tree"].items():
            copied = dict(node)
            copied["node_id"] = prefix + node_id
            copied["parent_id"] = (
                "collection_root" if node_id == root_id else prefix + str(node["parent_id"])
            )
            copied["children_ids"] = [prefix + str(value) for value in node.get("children_ids", [])]
            copied["view_ids"] = [prefix + str(value) for value in node.get("view_ids", [])]
            if node_id == root_id:
                copied["name"] = benchmark_id
                copied["summary"] = f"Scene scope {benchmark_id}. {node.get('summary', '')}"
                scene_roots.append(prefix + node_id)
            tree[prefix + node_id] = copied
    tree["collection_root"] = {
        "node_id": "collection_root",
        "node_type": "root",
        "name": "captured scene collection",
        "summary": "Scene-scoped semantic record collection.",
        "parent_id": None,
        "children_ids": scene_roots,
        "view_ids": sorted(views),
        "depth": 0,
    }
    return {
        "scene_id": "pooled_" + str(len(scene_ids)),
        "views": views,
        "tree": tree,
        "manifest": {"root_node_id": "collection_root"},
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(fmean(float(row[key]) for row in rows), 6)


def run(benchmark: Path, scene_root: Path, out: Path) -> dict[str, Any]:
    all_queries = filter_five_scene_queries(load_benchmark_queries(benchmark))
    order = ["ConferenceHall", "Museume", "Theater", "outdoor-street", "outdoor-drone"]
    per_query: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for count in range(1, len(order) + 1):
        scene_ids = order[:count]
        bundle = _prefixed_bundle(scene_ids, scene_root)
        queries = [row for row in all_queries if str(row["scene_id"]) in scene_ids]
        pool_rows: list[dict[str, Any]] = []
        for query in queries:
            scoped_query = f"scene {query['scene_id']} {query['query']}"
            flat = simulate_flat_search(bundle, scoped_query, query_id=str(query["query_id"]))
            graph = simulate_graph_search(
                bundle,
                scoped_query,
                query_id=str(query["query_id"]),
                branch_keep_ratio=0.5,
                fallback_child_limit=1,
            )
            expected = {f"{query['scene_id']}::{value}" for value in query.get("expected_view_ids", [])}
            row = {
                "pool_scene_count": count,
                "pool_view_count": len(bundle["views"]),
                "query_id": str(query["query_id"]),
                "query_scene_id": str(query["scene_id"]),
                "scene_scope_supplied": True,
                "flat_views": flat.views_checked,
                "graph_views": graph.views_checked,
                "flat_entries": flat.semantic_entries_scanned,
                "graph_entries": graph.semantic_entries_scanned,
                "flat_tokens": flat.input_tokens,
                "graph_tokens": graph.input_tokens,
                "flat_hit_at_3": bool(expected & set(flat.all_ranked_view_ids[:3])) if expected else None,
                "graph_hit_at_3": bool(expected & set(graph.all_ranked_view_ids[:3])) if expected else None,
            }
            pool_rows.append(row)
            per_query.append(row)
        measured.append(
            {
                "kind": "measured",
                "scene_count": count,
                "view_count": len(bundle["views"]),
                "query_count": len(pool_rows),
                "flat_views": _mean(pool_rows, "flat_views"),
                "graph_views": _mean(pool_rows, "graph_views"),
                "flat_entries": _mean(pool_rows, "flat_entries"),
                "graph_entries": _mean(pool_rows, "graph_entries"),
                "flat_tokens": _mean(pool_rows, "flat_tokens"),
                "graph_tokens": _mean(pool_rows, "graph_tokens"),
            }
        )

    extrapolated: list[dict[str, Any]] = []
    x = np.asarray([row["view_count"] for row in measured], dtype=float)
    for view_count in (250, 500, 1000):
        row: dict[str, Any] = {"kind": "linear_extrapolation", "view_count": view_count}
        for key in ("flat_views", "graph_views", "flat_entries", "graph_entries", "flat_tokens", "graph_tokens"):
            slope, intercept = np.polyfit(x, np.asarray([item[key] for item in measured]), 1)
            row[key] = round(max(0.0, float(slope * view_count + intercept)), 6)
        extrapolated.append(row)

    summary = {
        "schema_version": "beyond_proximity.reviewer_scaling.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": {
            "measured": "Cumulative pools of the five captured scenes; scene scope is supplied to both methods as query metadata.",
            "extrapolation": "Ordinary least-squares trend fitted only to the five measured pool sizes; no quality or latency is extrapolated.",
            "provider_latency_or_cost_inferred": False,
        },
        "measured": measured,
        "extrapolated": extrapolated,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "scaling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(per_query, indent=2) + "\n", encoding="utf-8")
    with (out / "scaling_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in measured + extrapolated for key in row}))
        writer.writeheader()
        writer.writerows(measured + extrapolated)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reviewer_revision/scaling_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.benchmark.resolve(), args.scene_root.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
