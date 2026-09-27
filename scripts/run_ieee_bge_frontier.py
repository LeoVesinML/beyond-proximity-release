#!/usr/bin/env python3
"""Run the held-out BGE efficiency--quality frontier on the five captured scenes."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder
from backend.query.graph_vs_flat import filter_five_scene_queries, hit_at_k, load_benchmark_queries, query_has_view_gt
from backend.query.recall_controlled import (
    PreparedScene,
    RecallControlledConfig,
    _choose_branches,
    _soft_branch_views,
    reciprocal_rank,
)
from backend.query.tokens import count_tokens
from scripts.run_ieee_recall_frontier import (
    MINIMUM_BRANCH_SAVINGS,
    MINIMUM_RECORD_SAVINGS,
    NONINFERIORITY_MARGIN,
    SEED,
    _choose_on_dev,
    _frozen_split,
    _load_scenes,
    _summary,
)


@dataclass
class BGEIndex:
    scene: PreparedScene
    view_matrix: np.ndarray
    branch_matrix: np.ndarray


def _softmax(ranked: list[tuple[float, str]], temperature: float) -> dict[str, float]:
    maximum = ranked[0][0] if ranked else 0.0
    weights = {item_id: math.exp((score - maximum) / temperature) for score, item_id in ranked}
    total = sum(weights.values()) or 1.0
    return {item_id: value / total for item_id, value in weights.items()}


def _rank(scores: dict[str, float]) -> list[tuple[float, str]]:
    return sorted(((float(score), key) for key, score in scores.items()), reverse=True)


def _indices(scenes: dict[str, PreparedScene], encoder: FastEmbedSemanticEncoder) -> dict[str, BGEIndex]:
    result = {}
    for scene_id, scene in scenes.items():
        view_matrix = encoder.encode([scene.view_texts[view_id] for view_id in scene.view_ids])
        branch_matrix = encoder.encode([scene.branch_texts[branch_id] for branch_id in scene.branch_ids])
        result[scene_id] = BGEIndex(scene=scene, view_matrix=view_matrix, branch_matrix=branch_matrix)
    return result


def _search(
    index: BGEIndex,
    query: dict[str, Any],
    vector: np.ndarray,
    config: RecallControlledConfig | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    scene = index.scene
    view_values = index.view_matrix @ vector
    view_scores = {view_id: float(view_values[pos]) for pos, view_id in enumerate(scene.view_ids)}
    ranked_global = _rank(view_scores)
    expected = {str(value) for value in query.get("expected_view_ids", []) if value}
    gt = query_has_view_gt(query)
    if config is None:
        checked = [view_id for _, view_id in ranked_global]
        ranked = checked
        context_texts = [scene.view_texts[view_id] for view_id in checked]
        return {
            "query_id": str(query["query_id"]), "scene_id": str(query["scene_id"]),
            "query": str(query["query"]), "query_type": query.get("query_type"),
            "method": "flat_bge", "records_checked": len(checked), "checked_view_ids": checked,
            "ranked_view_ids": ranked, "serialized_context_chars": sum(map(len, context_texts)),
            "serialized_context_tokens": sum(count_tokens(text) for text in context_texts),
            "proxy_view_comparisons": len(scene.view_ids), "proxy_branch_comparisons": 0,
            "gt_available": gt, "expected_view_ids": sorted(expected),
            "target_in_candidates": bool(expected & set(checked)),
            "hit_at_1": hit_at_k(ranked, sorted(expected), 1),
            "hit_at_3": hit_at_k(ranked, sorted(expected), 3),
            "mrr": reciprocal_rank(ranked, expected),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 6),
            "top_leaf_score": round(ranked_global[0][0], 6),
        }

    memberships = _soft_branch_views(scene, config.boundary_epsilon)
    if config.branch_score_mode == "node_text":
        branch_values = index.branch_matrix @ vector
        branch_scores = {
            branch_id: float(branch_values[pos]) for pos, branch_id in enumerate(scene.branch_ids)
        }
    else:
        branch_scores = {}
        for branch_id in scene.branch_ids:
            values = sorted((view_scores[value] for value in memberships[branch_id]), reverse=True)
            branch_scores[branch_id] = (
                values[0] if config.branch_score_mode == "max_leaf_proxy"
                else float(np.mean(values[:3]))
            ) if values else -1.0
    ranked_branches = _rank(branch_scores)
    probabilities = _softmax(ranked_branches, config.score_temperature)
    selected = _choose_branches(scene, ranked_branches, probabilities, config, expected, memberships)
    candidates = set().union(*(memberships[branch_id] for branch_id in selected)) if selected else set()
    safety = [view_id for _, view_id in ranked_global[: config.safety_shortlist_size]]
    candidates.update(safety)
    ranked_candidates = [(score, view_id) for score, view_id in ranked_global if view_id in candidates]
    if config.candidate_budget is not None:
        ranked_candidates = ranked_candidates[: config.candidate_budget]
    ranked = [view_id for _, view_id in ranked_candidates]
    context_texts = [scene.view_texts[view_id] for view_id in ranked]
    context_texts.extend(scene.branch_texts.values())
    probabilities_ordered = [probabilities[branch_id] for _, branch_id in ranked_branches]
    margin = probabilities_ordered[0] - probabilities_ordered[1] if len(probabilities_ordered) > 1 else 1.0
    fallback = config.fallback_to_flat and (
        (config.confidence_margin_delta > 0 and margin < config.confidence_margin_delta)
        or (config.minimum_leaf_score is not None and ranked_candidates and ranked_candidates[0][0] < config.minimum_leaf_score)
    )
    if fallback:
        ranked = [view_id for _, view_id in ranked_global]
        context_texts = [scene.view_texts[view_id] for view_id in ranked]
    return {
        "query_id": str(query["query_id"]), "scene_id": str(query["scene_id"]),
        "query": str(query["query"]), "query_type": query.get("query_type"),
        "config": asdict(config), "selected_branches": selected,
        "branch_count": len(scene.branch_ids), "selected_branch_count": len(selected),
        "selected_branch_ratio": len(selected) / len(scene.branch_ids),
        "fallback_triggered": fallback, "fallback_reasons": ["bge_confidence"] if fallback else [],
        "records_checked": len(ranked), "checked_view_ids": ranked, "ranked_view_ids": ranked,
        "serialized_context_chars": sum(map(len, context_texts)),
        "serialized_context_tokens": sum(count_tokens(text) for text in context_texts),
        "proxy_view_comparisons": len(scene.view_ids), "proxy_branch_comparisons": len(scene.branch_ids),
        "gt_available": gt, "expected_view_ids": sorted(expected),
        "target_in_candidates": bool(expected & set(ranked)),
        "hit_at_1": hit_at_k(ranked, sorted(expected), 1),
        "hit_at_3": hit_at_k(ranked, sorted(expected), 3),
        "mrr": reciprocal_rank(ranked, expected),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 6),
        "top_leaf_score": round(ranked_candidates[0][0], 6) if ranked_candidates else -1.0,
    }


def _run(queries: list[dict[str, Any]], indices: dict[str, BGEIndex], vectors: dict[str, np.ndarray], config: RecallControlledConfig | None) -> list[dict[str, Any]]:
    return [_search(indices[str(row["scene_id"])], row, vectors[str(row["query_id"])], config) for row in queries]


def run(benchmark: Path, scene_root: Path, out: Path) -> dict[str, Any]:
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark))
    split = _frozen_split(queries)
    scenes = _load_scenes(scene_root)
    encoder = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    indices = _indices(scenes, encoder)
    encoded_queries = encoder.encode([str(row["query"]) for row in queries])
    vectors = {str(row["query_id"]): encoded_queries[pos] for pos, row in enumerate(queries)}
    flat_dev = _run(split["dev"], indices, vectors, None)
    flat_dev_summary = _summary(flat_dev, label="flat_bge_dev")

    configs = [
        RecallControlledConfig(
            selection_mode="adaptive_mass", branch_score_mode=mode,
            branch_mass_threshold=tau, score_temperature=0.1,
            candidate_budget=6, safety_shortlist_size=safety,
            fallback_to_flat=False,
        )
        for mode in ("node_text", "max_leaf_proxy", "top3_leaf_proxy")
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9)
        for safety in (0, 1, 2, 3)
    ]
    candidates = [(config, _run(split["dev"], indices, vectors, config)) for config in configs]
    chosen, _chosen_rows, assessed = _choose_on_dev(candidates, flat_dev_summary, flat_dev)

    flat_test = _run(split["test"], indices, vectors, None)
    flat_summary = _summary(flat_test, label="flat_bge_test")
    frontier = []
    per_query = []
    runs = {}
    for budget in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, None):
        config = replace(chosen, candidate_budget=budget)
        rows = _run(split["test"], indices, vectors, config)
        label = f"budget_{budget if budget is not None else 'unlimited'}"
        summary = _summary(rows, label=label, flat_rows=flat_test)
        summary["candidate_budget"] = budget
        summary["config"] = asdict(config)
        summary["meets_m7_noninferiority"] = bool(
            summary["paired_hit_at_3_difference"]["ci_95"]["low"] >= NONINFERIORITY_MARGIN
            and summary["record_savings_vs_flat"] >= MINIMUM_RECORD_SAVINGS
            and summary["mean_selected_branch_ratio"] <= 1 - MINIMUM_BRANCH_SAVINGS
        )
        frontier.append(summary)
        runs[label] = rows
        per_query.extend({"frontier_label": label, **row} for row in rows)
    eligible = [row for row in frontier if row["meets_m7_noninferiority"]]
    operating = min(eligible, key=lambda row: (row["mean_records_checked"], -row["hit_at_3"])) if eligible else max(
        frontier, key=lambda row: row["hit_at_3"] + 0.5 * row["hit_at_1"] - 0.15 * row["record_ratio_vs_flat"]
    )
    oracle_config = replace(
        RecallControlledConfig(**operating["config"]), selection_mode="oracle",
        safety_shortlist_size=0, fallback_to_flat=False,
    )
    oracle_rows = _run(split["test"], indices, vectors, oracle_config)
    result = {
        "schema_version": "beyond_proximity.ieee_bge_frontier.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "model_name": DEFAULT_LOCAL_MODEL,
        "split_counts": {key: len(value) for key, value in split.items()},
        "tuning": {"final_config": asdict(chosen), "candidate_count": len(assessed), "candidates": assessed},
        "flat_test": flat_summary, "frontier": frontier, "operating_point": operating,
        "oracle_branch": _summary(oracle_rows, label="oracle_bge_test", flat_rows=flat_test),
        "decision": {"m7_operating_point_found": bool(eligible), "selected_frontier_label": operating["label"]},
        "accounting_note": "All view proxies are compared with BGE; serialized records and branch coverage are reported separately.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "bge_frontier_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "frontier_per_query.json").write_text(json.dumps(per_query, indent=2) + "\n", encoding="utf-8")
    (out / "operating_point_per_query.json").write_text(json.dumps(runs[operating["label"]], indent=2) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "benchmark": str(benchmark), "scene_root": str(scene_root), "model": DEFAULT_LOCAL_MODEL,
        "seed": SEED, "tuning_split": "50 frozen dev positives", "test_split": "75 frozen positives",
    }, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/bge_frontier_v1")
    args = parser.parse_args()
    result = run(args.benchmark.resolve(), args.scene_root.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "decision": result["decision"], "operating_point": result["operating_point"]}, indent=2))


if __name__ == "__main__":
    main()
