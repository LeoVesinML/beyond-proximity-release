#!/usr/bin/env python3
"""Rerank an identical frozen ScanNet candidate set with and without relations."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.grounding import (
    _fresh,
    _rank_lexical,
    build_candidates,
    graph_pruned_candidate_ids,
    parse_relation_query,
    rerank_relations,
)
from scripts.evaluate_grounding import bbox_iou_3d, expected_boxes
from scripts.run_experiment import load_scene


def _metric(candidates: list[Any], query: dict[str, Any]) -> tuple[bool, bool, bool]:
    expected = {str(value) for value in query.get("expected_object_ids", [])}
    candidate = candidates[0] if candidates else None
    hit = bool(candidate and candidate.object_id in expected)
    hit3 = bool(expected & {candidate.object_id for candidate in candidates[:3]})
    ious = [bbox_iou_3d(candidate.bbox_3d, box) for box in expected_boxes(query)] if candidate else []
    valid = [float(value) for value in ious if value is not None]
    return hit, hit3, bool(valid and max(valid) >= 0.25)


def _bootstrap(rows: list[dict[str, Any]], key_with: str, key_without: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(10_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        deltas.append(fmean(float(row[key_with]) - float(row[key_without]) for row in sample))
    ordered = sorted(deltas)
    return {
        "estimate": round(fmean(float(row[key_with]) - float(row[key_without]) for row in rows), 6),
        "ci_95": {"low": round(ordered[249], 6), "high": round(ordered[9749], 6)},
        "samples": 10_000,
        "seed": seed,
    }


def run(benchmark_path: Path, scene_root: Path, out: Path) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for query in benchmark["queries"]:
        scene_id = str(query["scene_id"])
        if scene_id not in bundles:
            loaded = load_scene(scene_root / scene_id)
            bundles[scene_id] = {"views": loaded["existing_views"], "tree": loaded["existing_tree"]}
        bundle = bundles[scene_id]
        canonical = build_candidates(bundle["views"], bundle["tree"])
        parsed = parse_relation_query(str(query["query"]))
        frozen_ids = graph_pruned_candidate_ids(canonical, bundle["tree"], parsed.target or str(query["query"]), branch_keep_ratio=0.5)
        frozen = [candidate for candidate in canonical if candidate.object_id in frozen_ids]
        lexical = _rank_lexical(_fresh(frozen), parsed.target or str(query["query"]))
        with_relation, unresolved, _ = rerank_relations(
            _fresh(lexical), _fresh(canonical), parsed, use_relation=True,
            relation_boost=0.20, relation_confidence_threshold=0.65,
            relation_score_margin=0.15, anchor_confidence_threshold=0.55,
            anchor_margin_threshold=0.10,
        )
        without_relation = _rank_lexical(_fresh(frozen), parsed.target or str(query["query"]))
        with_top = with_relation[0] if with_relation else None
        without_top = without_relation[0] if without_relation else None
        with_hit, with_hit3, with_acc = _metric(with_relation, query)
        without_hit, without_hit3, without_acc = _metric(without_relation, query)
        rows.append({
            "query_id": str(query["query_id"]),
            "scene_id": scene_id,
            "query_type": str(query.get("query_type") or "unspecified"),
            "relation": parsed.relation,
            "frozen_candidate_count": len(frozen),
            "frozen_candidate_ids": sorted(frozen_ids),
            "with_relation_top1": with_top.object_id if with_top else None,
            "without_relation_top1": without_top.object_id if without_top else None,
            "with_relation_hit_at_1": with_hit,
            "with_relation_hit_at_3": with_hit3,
            "without_relation_hit_at_1": without_hit,
            "without_relation_hit_at_3": without_hit3,
            "with_relation_acc_at_0_25": with_acc,
            "without_relation_acc_at_0_25": without_acc,
            "relation_unresolved": unresolved,
        })
    summary = {
        "schema_version": "beyond_proximity.matched_relation_ablation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "query_count": len(rows),
        "identical_candidate_sets_verified": all(row["frozen_candidate_count"] == len(row["frozen_candidate_ids"]) for row in rows),
        "mean_frozen_candidates": round(fmean(row["frozen_candidate_count"] for row in rows), 6),
        "with_relation": {
            "hit_at_1": round(fmean(row["with_relation_hit_at_1"] for row in rows), 6),
            "hit_at_3": round(fmean(row["with_relation_hit_at_3"] for row in rows), 6),
            "acc_at_0_25": round(fmean(row["with_relation_acc_at_0_25"] for row in rows), 6),
        },
        "without_relation": {
            "hit_at_1": round(fmean(row["without_relation_hit_at_1"] for row in rows), 6),
            "hit_at_3": round(fmean(row["without_relation_hit_at_3"] for row in rows), 6),
            "acc_at_0_25": round(fmean(row["without_relation_acc_at_0_25"] for row in rows), 6),
        },
        "paired_bootstrap": {
            "hit_at_1": _bootstrap(rows, "with_relation_hit_at_1", "without_relation_hit_at_1", 20260813),
            "acc_at_0_25": _bootstrap(rows, "with_relation_acc_at_0_25", "without_relation_acc_at_0_25", 20260814),
        },
        "limitation": "The paired sample contains 48 oracle-map queries from eight ScanNet scenes.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_relation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        serializable = [{**row, "frozen_candidate_ids": "|".join(row["frozen_candidate_ids"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader(); writer.writerows(serializable)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/scannet_bbq_grounding_pilot_v1.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reviewer_revision/matched_relation_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.benchmark.resolve(), args.scene_root.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
