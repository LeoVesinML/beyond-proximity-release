#!/usr/bin/env python3
"""Run the frozen dev/test recall-controlled retrieval study for IEEE Access."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    resolve_scene_dir,
)
from backend.query.recall_controlled import (
    PreparedScene,
    RecallControlledConfig,
    paired_bootstrap_difference,
    prepare_scene,
    search_flat_prepared_scene,
    search_prepared_scene,
    wilson_interval,
)

SEED = 20260901
DEV_PER_SCENE_TYPE = 2
NONINFERIORITY_MARGIN = -0.05
MINIMUM_RECORD_SAVINGS = 0.25
MINIMUM_BRANCH_SAVINGS = 0.20
TUNING_QUALITY_TOLERANCE = 0.05


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _frozen_split(queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positives = [row for row in queries if row.get("expected_view_ids")]
    negatives = [row for row in queries if not row.get("expected_view_ids")]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        groups[(str(row["scene_id"]), str(row["query_type"]))].append(row)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: str(row["query_id"]))
        dev.extend(ordered[:DEV_PER_SCENE_TYPE])
        test.extend(ordered[DEV_PER_SCENE_TYPE:])
    return {
        "dev": sorted(dev, key=lambda row: str(row["query_id"])),
        "test": sorted(test, key=lambda row: str(row["query_id"])),
        "negative_audit": sorted(negatives, key=lambda row: str(row["query_id"])),
    }


def _load_scenes(scene_root: Path) -> dict[str, PreparedScene]:
    scenes: dict[str, PreparedScene] = {}
    for benchmark_scene_id in sorted(DEFAULT_SCENE_MAP):
        scene_dir = resolve_scene_dir(benchmark_scene_id, scene_root)
        scenes[benchmark_scene_id] = prepare_scene(load_scene_bundle(scene_dir))
    return scenes


def _run_method(
    queries: Iterable[dict[str, Any]],
    scenes: dict[str, PreparedScene],
    config: RecallControlledConfig | None,
) -> list[dict[str, Any]]:
    rows = []
    for query in queries:
        scene = scenes[str(query["scene_id"])]
        if config is None:
            rows.append(search_flat_prepared_scene(scene, query))
        else:
            rows.append(search_prepared_scene(scene, query, config))
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return fmean(values) if values else 0.0


def _summary(
    rows: list[dict[str, Any]],
    *,
    label: str,
    flat_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("gt_available")]
    result: dict[str, Any] = {
        "label": label,
        "query_count": len(rows),
        "quality_denominator": len(eligible),
        "hit_at_1": round(_mean(eligible, "hit_at_1"), 6) if eligible else None,
        "hit_at_3": round(_mean(eligible, "hit_at_3"), 6) if eligible else None,
        "mrr": round(_mean(eligible, "mrr"), 6) if eligible else None,
        "mean_records_checked": round(_mean(rows, "records_checked"), 6),
        "mean_serialized_context_tokens": round(_mean(rows, "serialized_context_tokens"), 6),
        "mean_elapsed_ms": round(_mean(rows, "elapsed_ms"), 6),
        "fallback_rate": round(_mean(rows, "fallback_triggered"), 6)
        if rows and "fallback_triggered" in rows[0]
        else 0.0,
        "mean_selected_branch_ratio": round(_mean(rows, "selected_branch_ratio"), 6)
        if rows and "selected_branch_ratio" in rows[0]
        else None,
    }
    if eligible:
        for metric in ("hit_at_1", "hit_at_3"):
            successes = sum(bool(row[metric]) for row in eligible)
            low, high = wilson_interval(successes, len(eligible))
            result[f"{metric}_wilson_95"] = {
                "low": round(low, 6),
                "high": round(high, 6),
            }
    if flat_rows:
        flat_by_id = {str(row["query_id"]): row for row in flat_rows}
        paired = [row for row in eligible if str(row["query_id"]) in flat_by_id]
        for metric in ("hit_at_1", "hit_at_3", "mrr"):
            result[f"paired_{metric}_difference"] = paired_bootstrap_difference(
                [float(row[metric]) for row in paired],
                [float(flat_by_id[str(row["query_id"])][metric]) for row in paired],
                seed=SEED,
            )
        flat_records = _mean(flat_rows, "records_checked")
        flat_tokens = _mean(flat_rows, "serialized_context_tokens")
        result["record_ratio_vs_flat"] = round(result["mean_records_checked"] / flat_records, 6)
        result["record_savings_vs_flat"] = round(1 - result["record_ratio_vs_flat"], 6)
        result["context_token_ratio_vs_flat"] = round(
            result["mean_serialized_context_tokens"] / flat_tokens, 6
        )
        result["context_token_savings_vs_flat"] = round(
            1 - result["context_token_ratio_vs_flat"], 6
        )
    return result


def _choose_on_dev(
    candidates: list[tuple[RecallControlledConfig, list[dict[str, Any]]]],
    flat_summary: dict[str, Any],
    flat_rows: list[dict[str, Any]],
) -> tuple[RecallControlledConfig, list[dict[str, Any]], list[dict[str, Any]]]:
    assessed = []
    for config, rows in candidates:
        summary = _summary(rows, label="candidate", flat_rows=flat_rows)
        summary["config"] = asdict(config)
        summary["meets_tuning_quality_floor"] = bool(
            summary["hit_at_3"] >= flat_summary["hit_at_3"] - TUNING_QUALITY_TOLERANCE
            and summary["hit_at_1"] >= flat_summary["hit_at_1"] - TUNING_QUALITY_TOLERANCE
            and summary["mean_selected_branch_ratio"] <= 1 - MINIMUM_BRANCH_SAVINGS
        )
        assessed.append(summary)
    eligible = [row for row in assessed if row["meets_tuning_quality_floor"]]
    pool = eligible or assessed
    if eligible:
        winner = min(
            pool,
            key=lambda row: (
                row["record_ratio_vs_flat"],
                -row["hit_at_3"],
                -row["hit_at_1"],
                row["mean_serialized_context_tokens"],
            ),
        )
    else:
        winner = max(
            pool,
            key=lambda row: (
                row["hit_at_3"] + 0.5 * row["hit_at_1"] - 0.15 * row["record_ratio_vs_flat"],
                row["hit_at_3"],
                row["hit_at_1"],
            ),
        )
    winner_config = RecallControlledConfig(**winner["config"])
    winner_rows = next(rows for config, rows in candidates if config == winner_config)
    return winner_config, winner_rows, assessed


def _tune(
    dev: list[dict[str, Any]],
    scenes: dict[str, PreparedScene],
    flat_rows: list[dict[str, Any]],
) -> tuple[RecallControlledConfig, dict[str, Any], list[dict[str, Any]]]:
    flat_summary = _summary(flat_rows, label="flat_dev")
    current = RecallControlledConfig(
        selection_mode="fixed_top2",
        branch_score_mode="node_text",
        candidate_budget=6,
        safety_shortlist_size=0,
        fallback_to_flat=False,
        boundary_epsilon=0.0,
    )
    stage_rows: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "selection_rule": {
            "quality_floor": "both hit@1 and hit@3 no more than 0.05 below flat on dev, with at least 20% of branches pruned on average",
            "if_floor_met": "minimize record ratio, then maximize hit@3 and hit@1",
            "if_floor_unmet": "maximize hit@3 + 0.5*hit@1 - 0.15*record_ratio",
        },
        "flat_dev": flat_summary,
        "stages": [],
    }

    stages: list[tuple[str, list[RecallControlledConfig]]] = []
    stages.append(("M1_adaptive_branch_mass", [
        replace(current, selection_mode="adaptive_mass", branch_mass_threshold=tau)
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9)
    ]))
    for stage_name, candidates in stages:
        evaluated = [(config, _run_method(dev, scenes, config)) for config in candidates]
        current, current_rows, assessed = _choose_on_dev(evaluated, flat_summary, flat_rows)
        trace["stages"].append({"stage": stage_name, "winner": asdict(current), "candidates": assessed})
        stage_rows.append({"stage": stage_name, **_summary(current_rows, label=stage_name, flat_rows=flat_rows)})

    ordered_leaf_scores = sorted(float(row["top_leaf_score"]) for row in flat_rows)
    leaf_p10 = ordered_leaf_scores[int(0.10 * (len(ordered_leaf_scores) - 1))]
    leaf_p25 = ordered_leaf_scores[int(0.25 * (len(ordered_leaf_scores) - 1))]
    remaining = [
        ("M4_leaf_proxy", [replace(current, branch_score_mode=mode) for mode in (
            "node_text", "max_leaf_proxy", "top3_leaf_proxy"
        )]),
        ("M3_global_safety_shortlist", [replace(current, safety_shortlist_size=size) for size in (0, 1, 2, 3)]),
        ("M2_confidence_fallback", [
            replace(
                current,
                confidence_margin_delta=delta,
                minimum_leaf_score=threshold,
                fallback_to_flat=delta > 0 or threshold is not None,
            )
            for delta in (0.0, 0.05, 0.1)
            for threshold in (None, leaf_p10, leaf_p25)
        ]),
        ("M5_soft_membership", [replace(current, boundary_epsilon=epsilon) for epsilon in (0.0, 0.5, 1.0, 2.0, 4.0)]),
    ]
    for stage_name, initial_candidates in remaining:
        # Rebase each stage on the winner of the preceding stage.
        if stage_name == "M4_leaf_proxy":
            candidates = [replace(current, branch_score_mode=item.branch_score_mode) for item in initial_candidates]
        elif stage_name == "M3_global_safety_shortlist":
            candidates = [replace(current, safety_shortlist_size=item.safety_shortlist_size) for item in initial_candidates]
        elif stage_name == "M2_confidence_fallback":
            candidates = [replace(
                current,
                confidence_margin_delta=item.confidence_margin_delta,
                minimum_leaf_score=item.minimum_leaf_score,
                fallback_to_flat=item.fallback_to_flat,
            ) for item in initial_candidates]
        else:
            candidates = [replace(current, boundary_epsilon=item.boundary_epsilon) for item in initial_candidates]
        evaluated = [(config, _run_method(dev, scenes, config)) for config in candidates]
        current, current_rows, assessed = _choose_on_dev(evaluated, flat_summary, flat_rows)
        trace["stages"].append({"stage": stage_name, "winner": asdict(current), "candidates": assessed})
        stage_rows.append({"stage": stage_name, **_summary(current_rows, label=stage_name, flat_rows=flat_rows)})
    trace["final_config"] = asdict(current)
    return current, trace, stage_rows


def _failure_taxonomy(
    graph_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    scenes: dict[str, PreparedScene],
) -> dict[str, Any]:
    oracle_by_id = {str(row["query_id"]): row for row in oracle_rows}
    failures = []
    counts: Counter[str] = Counter()
    for row in graph_rows:
        if not row.get("gt_available") or row.get("hit_at_3"):
            continue
        oracle = oracle_by_id[str(row["query_id"])]
        selected = set(row.get("selected_branches", []))
        expected = set(row.get("expected_view_ids", []))
        scene = scenes[str(row["scene_id"])]
        expected_branches = {
            branch_id for branch_id, views in scene.branch_views.items() if expected & views
        }
        if not oracle.get("hit_at_3"):
            category = "leaf_scorer_failure_in_oracle_branch"
        elif not selected & expected_branches:
            category = "wrong_branch_gate"
        elif not row.get("target_in_candidates"):
            category = "candidate_budget_or_membership"
        elif not row.get("ranked_view_ids"):
            category = "zero_lexical_overlap"
        else:
            category = "leaf_ranking_error"
        counts[category] += 1
        failures.append({
            "query_id": row["query_id"],
            "scene_id": row["scene_id"],
            "query": row["query"],
            "category": category,
            "selected_branches": row.get("selected_branches", []),
            "expected_branches": sorted(expected_branches),
            "checked_view_ids": row.get("checked_view_ids", []),
            "expected_view_ids": row.get("expected_view_ids", []),
            "oracle_hit_at_3": oracle.get("hit_at_3"),
        })
    return {
        "miss_count": len(failures),
        "category_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def run(benchmark: Path, scene_root: Path, out: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark))
    split = _frozen_split(queries)
    scenes = _load_scenes(scene_root)
    flat_dev = _run_method(split["dev"], scenes, None)
    chosen, tuning, tuning_stages = _tune(split["dev"], scenes, flat_dev)

    flat_test = _run_method(split["test"], scenes, None)
    flat_test_summary = _summary(flat_test, label="flat_test")
    budgets: list[int | None] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, None]
    frontier_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    frontier_runs: dict[str, list[dict[str, Any]]] = {}
    for budget in budgets:
        config = replace(chosen, candidate_budget=budget)
        rows = _run_method(split["test"], scenes, config)
        label = f"budget_{budget if budget is not None else 'unlimited'}"
        summary = _summary(rows, label=label, flat_rows=flat_test)
        summary["candidate_budget"] = budget
        summary["config"] = asdict(config)
        ci = summary["paired_hit_at_3_difference"]["ci_95"]
        summary["meets_m7_noninferiority"] = bool(
            ci["low"] >= NONINFERIORITY_MARGIN
            and summary["record_savings_vs_flat"] >= MINIMUM_RECORD_SAVINGS
            and summary["mean_selected_branch_ratio"] <= 1 - MINIMUM_BRANCH_SAVINGS
        )
        frontier_rows.append(summary)
        frontier_runs[label] = rows
        per_query.extend({"frontier_label": label, **row} for row in rows)

    eligible = [row for row in frontier_rows if row["meets_m7_noninferiority"]]
    operating = min(
        eligible,
        key=lambda row: (row["mean_records_checked"], -row["hit_at_3"], -row["hit_at_1"]),
    ) if eligible else max(
        frontier_rows,
        key=lambda row: (
            row["hit_at_3"] + 0.5 * row["hit_at_1"] - 0.15 * row["record_ratio_vs_flat"],
            row["hit_at_3"],
        ),
    )
    operating_rows = frontier_runs[operating["label"]]
    operating_config = RecallControlledConfig(**operating["config"])

    oracle_config = replace(
        operating_config,
        selection_mode="oracle",
        safety_shortlist_size=0,
        confidence_margin_delta=0.0,
        fallback_to_flat=False,
    )
    oracle_rows = _run_method(split["test"], scenes, oracle_config)
    oracle_summary = _summary(oracle_rows, label="oracle_branch_test", flat_rows=flat_test)

    legacy_config = RecallControlledConfig(
        selection_mode="fixed_top2",
        branch_score_mode="node_text",
        candidate_budget=operating_config.candidate_budget,
        safety_shortlist_size=0,
        fallback_to_flat=False,
    )
    ablation_configs = [
        ("legacy", legacy_config),
        ("M1_adaptive_mass", replace(legacy_config, selection_mode="adaptive_mass", branch_mass_threshold=chosen.branch_mass_threshold)),
        ("M1+M4_leaf_proxy", replace(legacy_config, selection_mode="adaptive_mass", branch_mass_threshold=chosen.branch_mass_threshold, branch_score_mode=chosen.branch_score_mode)),
        ("M1+M4+M3_safety", replace(legacy_config, selection_mode="adaptive_mass", branch_mass_threshold=chosen.branch_mass_threshold, branch_score_mode=chosen.branch_score_mode, safety_shortlist_size=chosen.safety_shortlist_size)),
        ("M1+M4+M3+M2_fallback", replace(legacy_config, selection_mode="adaptive_mass", branch_mass_threshold=chosen.branch_mass_threshold, branch_score_mode=chosen.branch_score_mode, safety_shortlist_size=chosen.safety_shortlist_size, confidence_margin_delta=chosen.confidence_margin_delta, minimum_leaf_score=chosen.minimum_leaf_score, fallback_to_flat=chosen.fallback_to_flat)),
        ("selected_M1+M4+M3", operating_config),
    ]
    ablations = []
    for label, config in ablation_configs:
        rows = _run_method(split["test"], scenes, config)
        item = _summary(rows, label=label, flat_rows=flat_test)
        item["config"] = asdict(config)
        ablations.append(item)

    negative_rows = _run_method(split["negative_audit"], scenes, operating_config)
    negative_audit = {
        "query_count": len(negative_rows),
        "accuracy": None,
        "reason": "Negative correctness is not scored without independently exhaustive absence GT.",
        "mean_records_checked": round(_mean(negative_rows, "records_checked"), 6),
        "mean_context_tokens": round(_mean(negative_rows, "serialized_context_tokens"), 6),
        "fallback_rate": round(_mean(negative_rows, "fallback_triggered"), 6),
    }
    taxonomy = _failure_taxonomy(operating_rows, oracle_rows, scenes)
    decision = {
        "route": "option_a_method_paper" if eligible else "option_b_empirical_analysis",
        "criterion": {
            "paired_hit_at_3_difference_ci_lower_bound_at_least": NONINFERIORITY_MARGIN,
            "minimum_record_savings_vs_flat": MINIMUM_RECORD_SAVINGS,
            "minimum_branch_savings": MINIMUM_BRANCH_SAVINGS,
        },
        "m7_operating_point_found": bool(eligible),
        "selected_frontier_label": operating["label"],
        "selected_reason": "lowest checked-record cost among qualifying points" if eligible else "highest predeclared fallback utility; no point met M7",
    }
    payload = {
        "schema_version": "beyond_proximity.ieee_recall_frontier.v1",
        "generated_at": generated_at,
        "status": "complete",
        "benchmark": str(benchmark.relative_to(ROOT)),
        "scene_root": str(scene_root.relative_to(ROOT)),
        "seed": SEED,
        "split": {
            key: [str(row["query_id"]) for row in values]
            for key, values in split.items()
        },
        "split_counts": {key: len(values) for key, values in split.items()},
        "tuning": tuning,
        "flat_test": flat_test_summary,
        "frontier": frontier_rows,
        "operating_point": operating,
        "oracle_branch": oracle_summary,
        "ablations": ablations,
        "negative_audit": negative_audit,
        "failure_taxonomy": taxonomy,
        "decision": decision,
        "cost_accounting": {
            "serialized_context": "view records retained for leaf ranking plus all root child summaries needed for branch scoring",
            "proxy_view_comparisons": "cached lexical leaf proxies evaluated for every scene view; reported separately and never called sublinear",
            "construction_cost": "excluded from per-query retrieval and reported separately in existing construction artifacts",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_config.json", {
        "generated_at": generated_at,
        "benchmark": payload["benchmark"],
        "scene_root": payload["scene_root"],
        "seed": SEED,
        "dev_per_scene_query_type": DEV_PER_SCENE_TYPE,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "minimum_record_savings": MINIMUM_RECORD_SAVINGS,
        "minimum_branch_savings": MINIMUM_BRANCH_SAVINGS,
        "tuning_quality_tolerance": TUNING_QUALITY_TOLERANCE,
        "candidate_budgets": budgets,
    })
    _write_json(out / "split.json", payload["split"])
    _write_json(out / "tuning.json", tuning)
    _write_json(out / "frontier_summary.json", payload)
    _write_json(out / "frontier_per_query.json", per_query)
    _write_json(out / "operating_point_per_query.json", operating_rows)
    _write_json(out / "oracle_branch_per_query.json", oracle_rows)
    _write_json(out / "negative_audit_per_query.json", negative_rows)
    _write_json(out / "failure_taxonomy.json", taxonomy)
    _write_csv(out / "frontier.csv", [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in frontier_rows])
    _write_csv(out / "ablations.csv", [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in ablations])
    _write_csv(out / "tuning_stages.csv", [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in tuning_stages])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/recall_frontier_v1")
    args = parser.parse_args()
    result = run(args.benchmark.resolve(), args.scene_root.resolve(), args.out.resolve())
    print(json.dumps({
        "status": result["status"],
        "split_counts": result["split_counts"],
        "decision": result["decision"],
        "operating_point": result["operating_point"],
    }, indent=2))


if __name__ == "__main__":
    main()
