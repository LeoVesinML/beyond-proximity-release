#!/usr/bin/env python3
"""Sweep recall-control constants on the frozen development split only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import filter_five_scene_queries, load_benchmark_queries
from backend.query.recall_controlled import RecallControlledConfig, search_prepared_scene
from scripts.run_ieee_recall_frontier import _frozen_split, _load_scenes, _summary


SWEEPS: dict[str, tuple[str, list[float | int]]] = {
    "branch_mass": ("branch_mass_threshold", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "temperature": ("score_temperature", [0.1, 0.25, 0.5, 1.0]),
    "branch_cap": ("max_branches", [1, 2, 3, 4, 5]),
    "safety_shortlist": ("safety_shortlist_size", [0, 1, 2, 3]),
    "candidate_budget": ("candidate_budget", [1, 2, 3, 4, 6, 8, 12]),
    "boundary_overlap": ("boundary_epsilon", [0.0, 0.1, 0.25, 0.5]),
}


def run(benchmark: Path, scene_root: Path, frontier: Path, out: Path) -> dict[str, Any]:
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark))
    dev = _frozen_split(queries)["dev"]
    scenes = _load_scenes(scene_root)
    source = json.loads(frontier.read_text(encoding="utf-8"))
    base = RecallControlledConfig(**source["operating_point"]["config"])
    rows: list[dict[str, Any]] = []
    for group, (field, values) in SWEEPS.items():
        for value in values:
            config = replace(base, **{field: value})
            per_query = [search_prepared_scene(scenes[str(query["scene_id"])], query, config) for query in dev]
            summary = _summary(per_query, label=f"{group}_{value}")
            rows.append({
                "group": group,
                "field": field,
                "value": value,
                "query_count": summary["query_count"],
                "hit_at_1": summary["hit_at_1"],
                "hit_at_3": summary["hit_at_3"],
                "mrr": summary["mrr"],
                "mean_records_checked": summary["mean_records_checked"],
                "mean_context_tokens": summary["mean_serialized_context_tokens"],
                "mean_selected_branch_ratio": summary["mean_selected_branch_ratio"],
                "fallback_rate": summary["fallback_rate"],
            })
    payload = {
        "schema_version": "beyond_proximity.ieee_sensitivity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "split": "frozen development positives only",
        "query_count": len(dev),
        "base_config": source["operating_point"]["config"],
        "sweeps": rows,
        "test_set_used": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--frontier", type=Path, default=ROOT / "outputs/ieee_revision/recall_frontier_v1/frontier_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/sensitivity_v1")
    args = parser.parse_args()
    result = run(args.benchmark.resolve(), args.scene_root.resolve(), args.frontier.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "query_count": result["query_count"], "settings": len(result["sweeps"])}, indent=2))


if __name__ == "__main__":
    main()
