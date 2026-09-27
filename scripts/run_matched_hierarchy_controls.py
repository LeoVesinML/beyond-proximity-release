#!/usr/bin/env python3
"""Evaluate construction variants at matched budgets and structural complexity."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.geometry.spatial_graph import build_spatial_clusters
from backend.io.view_store import list_view_summaries
from backend.query.affordance import score_text_with_tokens, tokenize
from backend.query.graph_vs_flat import load_benchmark_queries, load_scene_bundle, simulate_graph_search
from backend.query.model_client import _view_text
from scripts.evaluate_hierarchy_construction import (
    _agent_zones,
    _cluster_records,
    _manual_zones,
    _positions,
    _tree_with_zones,
    _zone_assignment_metrics,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_zones(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "zone_id": str(row["zone_id"]),
        "zone_name": str(row["zone_label"]),
        "view_ids": sorted(str(value) for value in row["view_ids"]),
        "rationale": str(row.get("description") or ""),
    } for row in rows]


def _disjoint_projection(rows: list[dict[str, Any]], view_ids: set[str]) -> list[dict[str, Any]]:
    membership: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for view_id in row.get("view_ids", []):
            membership[str(view_id)].append(row)
    assigned: dict[str, list[str]] = defaultdict(list)
    for view_id in sorted(view_ids):
        options = membership.get(view_id, [])
        if not options:
            assigned["projected_orphan"].append(view_id)
            continue
        chosen = max(options, key=lambda row: (int(row.get("query_reference_count") or 0), len(row.get("view_ids", [])), str(row["zone_id"])))
        assigned[str(chosen["zone_id"])].append(view_id)
    return [{"zone_id": zone_id, "zone_name": zone_id, "view_ids": members, "rationale": "disjoint projection"} for zone_id, members in sorted(assigned.items()) if members]


def _random_partition(view_ids: set[str], sizes: list[int], rng: random.Random) -> list[dict[str, Any]]:
    ordered = sorted(view_ids)
    rng.shuffle(ordered)
    zones = []
    offset = 0
    for index, size in enumerate(sizes):
        zones.append({"zone_id": f"random_{index:02d}", "view_ids": ordered[offset:offset + size]})
        offset += size
    if offset < len(ordered):
        zones[-1]["view_ids"].extend(ordered[offset:])
    return zones


def _rank_views(views: dict[str, Any], ids: set[str], query: str) -> list[str]:
    terms = tokenize(query)
    scored = []
    for view_id in ids:
        overlap, ratio = score_text_with_tokens(_view_text(views[view_id]), terms)
        scored.append((overlap, ratio, view_id))
    return [view_id for _overlap, _ratio, view_id in sorted(scored, reverse=True)]


def _variants(config: dict[str, Any], scene_id: str, bundle: dict[str, Any], reference: list[dict[str, Any]], decisions: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    summaries = list_view_summaries(str(ROOT / config["scene_root"] / scene_id))
    positions = _positions(ROOT / config["scene_root"] / scene_id)
    pose = _cluster_records(build_spatial_clusters(positions, scale_factor=float(config["spatial_scale_factor"])), summaries, variant="pose_only")
    semantic = _cluster_records(build_spatial_clusters(positions, scale_factor=float(config["spatial_scale_factor"]), view_summaries=summaries), summaries, variant="pose_semantic_merge")
    return {
        "manual_reference": _reference_zones(reference),
        "pose_only": pose,
        "pose_semantic_merge": semantic,
        "cursor_agent_mcp": _agent_zones(decisions, scene_id, set(bundle["views"])),
    }


def run(config_path: Path, out: Path) -> dict[str, Any]:
    config = _read(config_path)
    benchmark = load_benchmark_queries(ROOT / config["benchmark_path"])
    reference_by_scene = _manual_zones(ROOT / config["manual_zone_reference_path"])
    decisions = _read(ROOT / config["agent_decisions_path"])
    scene_map = {str(key): str(value) for key, value in config["benchmark_scene_map"].items()}
    budgets = [4, 6, 8, 10]
    seed = int(config["seed"])
    query_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for scene_id in config["scene_ids"]:
        scene_id = str(scene_id)
        bundle = load_scene_bundle(ROOT / config["scene_root"] / scene_id)
        reference = reference_by_scene[scene_id]
        variants = _variants(config, scene_id, bundle, reference, decisions)
        projected = _disjoint_projection(reference, set(bundle["views"]))
        projection_metric = _zone_assignment_metrics(projected, reference, set(bundle["views"]))
        structure_rows.append({
            "scene_id": scene_id,
            "variant": "disjoint_manual_projection",
            "zone_count": len(projected),
            "pairwise_f1": projection_metric["pairwise_f1"],
            "best_zone_jaccard": projection_metric["weighted_best_zone_jaccard"],
        })
        scene_queries = [row for row in benchmark if str(row.get("scene_id")) == scene_map[scene_id]]
        for variant, zones in variants.items():
            metric = _zone_assignment_metrics(zones, reference, set(bundle["views"]))
            structure_rows.append({
                "scene_id": scene_id,
                "variant": variant,
                "zone_count": len(zones),
                "pairwise_f1": metric["pairwise_f1"],
                "best_zone_jaccard": metric["weighted_best_zone_jaccard"],
            })
            sizes = [len(zone["view_ids"]) for zone in zones]
            rng = random.Random(seed + sum(ord(char) for char in scene_id + variant))
            null_f1 = []
            for _ in range(1000):
                random_metric = _zone_assignment_metrics(_random_partition(set(bundle["views"]), sizes, rng), reference, set(bundle["views"]))
                null_f1.append(float(random_metric["pairwise_f1"]))
            ordered = sorted(null_f1)
            random_rows.append({
                "scene_id": scene_id,
                "variant": variant,
                "zone_count": len(zones),
                "random_trials": 1000,
                "random_pairwise_f1_mean": round(fmean(null_f1), 6),
                "random_pairwise_f1_low": round(ordered[24], 6),
                "random_pairwise_f1_high": round(ordered[974], 6),
            })
            tree_bundle = _tree_with_zones(bundle, zones, variant=variant)
            for query in scene_queries:
                graph = simulate_graph_search(
                    tree_bundle, str(query["query"]), query_id=str(query["query_id"]),
                    benchmark_scene_id=scene_map[scene_id], branch_keep_ratio=0.5,
                    fallback_child_limit=3,
                )
                branch_rank = _rank_views(bundle["views"], set(graph.views_checked_ids), str(query["query"]))
                flat_rank = _rank_views(bundle["views"], set(bundle["views"]), str(query["query"]))
                priority = list(dict.fromkeys([*branch_rank, *flat_rank]))
                expected = {str(value) for value in query.get("expected_view_ids", [])}
                for budget in budgets:
                    retained = priority[: min(budget, len(priority))]
                    query_rows.append({
                        "scene_id": scene_id,
                        "query_id": str(query["query_id"]),
                        "variant": variant,
                        "budget": budget,
                        "retained_count": len(retained),
                        "hit_at_1": bool(retained[:1] and retained[0] in expected) if expected else None,
                        "hit_at_3": bool(expected & set(retained[:3])) if expected else None,
                    })
    matched: list[dict[str, Any]] = []
    for variant in config["variants"]:
        for budget in budgets:
            rows = [row for row in query_rows if row["variant"] == variant and row["budget"] == budget]
            eligible = [row for row in rows if row["hit_at_1"] is not None]
            matched.append({
                "variant": variant,
                "budget": budget,
                "query_count": len(rows),
                "quality_denominator": len(eligible),
                "mean_retained_count": round(fmean(row["retained_count"] for row in rows), 6),
                "hit_at_1": round(fmean(row["hit_at_1"] for row in eligible), 6),
                "hit_at_3": round(fmean(row["hit_at_3"] for row in eligible), 6),
            })
    summary = {
        "schema_version": "beyond_proximity.matched_hierarchy_controls.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "matched_budgets": matched,
        "structure": {
            "per_scene": structure_rows,
            "random_matched_complexity": random_rows,
            "disjoint_projection_note": "Each view is assigned to one of its manual zones using only frozen zone prevalence; this is an attainable reference under a one-cluster-per-view representation, not independent GT or a mathematical upper bound.",
        },
        "protocol": {
            "budgets": budgets,
            "shared_leaf_scorer": "deterministic lexical overlap",
            "retention": "Hierarchy-prioritized views are truncated or filled by the shared flat leaf order to exactly min(budget, scene views).",
            "random_trials_per_scene_variant": 1000,
            "seed": seed,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_hierarchy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(query_rows, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("matched_budgets.csv", matched), ("structure.csv", structure_rows), ("random_baseline.csv", random_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hierarchy_construction_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reviewer_revision/matched_hierarchy_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
