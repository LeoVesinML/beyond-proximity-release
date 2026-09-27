#!/usr/bin/env python3
"""Revision-2 analyses of the lexical and BGE recall frontiers.

Adds, relative to ``run_ieee_recall_frontier.py`` and ``run_ieee_bge_frontier.py``:

* budget ``k`` selected on the development split only (the original script
  selected the operating budget on the test frontier);
* flat top-``k`` under the same scorer as the fair baseline;
* exact cl100k_base token counts, split into view records and branch summaries;
* a per-query identity check between Algorithm 1 and flat top-3;
* the component ablation on the development and test splits;
* an abstention rule for the 25 negative queries.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import FastEmbedSemanticEncoder
from backend.query.graph_vs_flat import filter_five_scene_queries, load_benchmark_queries
from backend.query.recall_controlled import (
    RecallControlledConfig,
    paired_bootstrap_difference,
    reciprocal_rank,
    search_flat_prepared_scene,
    search_prepared_scene,
    wilson_interval,
)
from backend.query.tokens import count_tokens
from scripts.run_ieee_bge_frontier import _indices, _search
from scripts.run_ieee_recall_frontier import (
    MINIMUM_BRANCH_SAVINGS,
    MINIMUM_RECORD_SAVINGS,
    NONINFERIORITY_MARGIN,
    SEED,
    _frozen_split,
    _load_scenes,
)

BUDGETS = [1, 2, 3, 4, 6, 8, 12]
LEXICAL = RecallControlledConfig(
    selection_mode="adaptive_mass", branch_score_mode="max_leaf_proxy",
    branch_mass_threshold=0.7, score_temperature=0.25, max_branches=5,
    safety_shortlist_size=1, fallback_to_flat=False,
)
# The dev tuning of run_ieee_bge_frontier.py selected branch-summary (node_text)
# scoring for BGE, not max-leaf scoring; see outputs/ieee_revision/bge_frontier_v1.
BGE = RecallControlledConfig(
    selection_mode="adaptive_mass", branch_score_mode="node_text",
    branch_mass_threshold=0.5, score_temperature=0.1, max_branches=5,
    safety_shortlist_size=1, fallback_to_flat=False,
)


def _positives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("gt_available")]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = _positives(rows)
    out = {
        "n": len(pos),
        "hit_at_1": fmean(float(r["hit_at_1"]) for r in pos),
        "hit_at_3": fmean(float(r["hit_at_3"]) for r in pos),
        "mrr": fmean(float(r["mrr"]) for r in pos),
        "retained": fmean(float(r["records_checked"]) for r in rows),
        "view_tokens": fmean(float(r["view_tokens"]) for r in rows),
        "branch_tokens": fmean(float(r["branch_tokens"]) for r in rows),
    }
    out["context_tokens"] = out["view_tokens"] + out["branch_tokens"]
    low, high = wilson_interval(sum(bool(r["hit_at_3"]) for r in pos), len(pos))
    out["hit_at_3_wilson"] = [low, high]
    return out


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    b_by = {r["query_id"]: r for r in b}
    pa = [r for r in _positives(a) if r["query_id"] in b_by]
    return {
        metric: paired_bootstrap_difference(
            [float(r[metric]) for r in pa], [float(b_by[r["query_id"]][metric]) for r in pa], seed=SEED
        )
        for metric in ("hit_at_1", "hit_at_3", "mrr")
    }


class Track:
    """Wraps one scorer so that flat, flat top-k, and Algorithm 1 share leaf scores."""

    def __init__(self, name: str, scenes, queries, encoder=None):
        self.name = name
        self.scenes = scenes
        if name == "bge":
            self.indices = _indices(scenes, encoder)
            vectors = encoder.encode([str(q["query"]) for q in queries])
            self.vectors = {str(q["query_id"]): vectors[i] for i, q in enumerate(queries)}

    def _annotate(self, row: dict[str, Any], scene, with_branches: bool) -> dict[str, Any]:
        row["view_tokens"] = sum(count_tokens(scene.view_texts[v]) for v in row["checked_view_ids"])
        row["branch_tokens"] = (
            sum(count_tokens(scene.branch_texts[b]) for b in scene.branch_ids) if with_branches else 0
        )
        return row

    def flat(self, query: dict[str, Any]) -> dict[str, Any]:
        scene = self.scenes[str(query["scene_id"])]
        if self.name == "lexical":
            row = search_flat_prepared_scene(scene, query)
        else:
            row = _search(self.indices[str(query["scene_id"])], query, self.vectors[str(query["query_id"])], None)
        return self._annotate(row, scene, with_branches=False)

    def flat_top_k(self, query: dict[str, Any], k: int) -> dict[str, Any]:
        row = dict(self.flat(query))
        scene = self.scenes[str(query["scene_id"])]
        ranked = row["ranked_view_ids"][:k]
        expected = set(row["expected_view_ids"])
        row.update(
            checked_view_ids=row["checked_view_ids"][:k], ranked_view_ids=ranked,
            records_checked=min(k, row["records_checked"]),
            hit_at_1=bool(expected & set(ranked[:1])), hit_at_3=bool(expected & set(ranked[:3])),
            mrr=reciprocal_rank(ranked, expected),
        )
        return self._annotate(row, scene, with_branches=False)

    def algorithm1(self, query: dict[str, Any], config: RecallControlledConfig) -> dict[str, Any]:
        scene = self.scenes[str(query["scene_id"])]
        if self.name == "lexical":
            row = search_prepared_scene(scene, query, config)
        else:
            row = _search(self.indices[str(query["scene_id"])], query, self.vectors[str(query["query_id"])], config)
        return self._annotate(row, scene, with_branches=True)


def _frontier(track: Track, queries, base: RecallControlledConfig) -> dict[str, Any]:
    flat_rows = [track.flat(q) for q in queries]
    result = {"flat": {"summary": _metrics(flat_rows), "rows": flat_rows}, "budgets": {}}
    for k in BUDGETS:
        cfg = replace(base, candidate_budget=k)
        algo = [track.algorithm1(q, cfg) for q in queries]
        topk = [track.flat_top_k(q, k) for q in queries]
        summary = _metrics(algo)
        paired_flat = _paired(algo, flat_rows)
        summary["paired_vs_exhaustive"] = paired_flat
        summary["paired_vs_flat_top_k"] = _paired(algo, topk)
        summary["record_savings_vs_exhaustive"] = 1 - summary["retained"] / result["flat"]["summary"]["retained"]
        summary["mean_selected_branch_ratio"] = fmean(r["selected_branch_ratio"] for r in algo)
        summary["qualifies"] = bool(
            paired_flat["hit_at_3"]["ci_95"]["low"] >= NONINFERIORITY_MARGIN
            and summary["record_savings_vs_exhaustive"] >= MINIMUM_RECORD_SAVINGS
            and summary["mean_selected_branch_ratio"] <= 1 - MINIMUM_BRANCH_SAVINGS
        )
        topk_by = {r["query_id"]: r for r in topk}
        summary["hit3_disagreements_vs_flat_top_k"] = sorted(
            r["query_id"] for r in _positives(algo) if bool(r["hit_at_3"]) != bool(topk_by[r["query_id"]]["hit_at_3"])
        )
        summary["hit1_disagreements_vs_flat_top_k"] = sorted(
            r["query_id"] for r in _positives(algo) if bool(r["hit_at_1"]) != bool(topk_by[r["query_id"]]["hit_at_1"])
        )
        result["budgets"][k] = {
            "algorithm1": summary,
            "flat_top_k": _metrics(topk),
            "algorithm1_rows": algo,
            "flat_top_k_rows": topk,
        }
    qualifying = [k for k in BUDGETS if result["budgets"][k]["algorithm1"]["qualifies"]]
    result["selected_budget"] = min(qualifying) if qualifying else None
    return result


def _ablation(track: Track, queries, base: RecallControlledConfig, k: int) -> list[dict[str, Any]]:
    legacy = RecallControlledConfig(
        selection_mode="fixed_top2", branch_score_mode="node_text", candidate_budget=k,
        safety_shortlist_size=0, fallback_to_flat=False,
    )
    stages = [
        ("fixed two-branch gate", legacy),
        ("+ adaptive mass", replace(legacy, selection_mode="adaptive_mass",
                                    branch_mass_threshold=base.branch_mass_threshold,
                                    score_temperature=base.score_temperature)),
        ("+ configured branch score", replace(legacy, selection_mode="adaptive_mass",
                                            branch_mass_threshold=base.branch_mass_threshold,
                                            score_temperature=base.score_temperature,
                                            branch_score_mode=base.branch_score_mode)),
        ("+ safety shortlist (Algorithm 1)", replace(base, candidate_budget=k)),
    ]
    out = []
    for label, cfg in stages:
        rows = [track.algorithm1(q, cfg) for q in queries]
        m = _metrics(rows)
        out.append({"stage": label, "hit_at_1": m["hit_at_1"], "hit_at_3": m["hit_at_3"],
                    "mrr": m["mrr"], "retained": m["retained"], "n": m["n"]})
    return out


def _abstention(track: Track, dev, test, negatives, config) -> dict[str, Any]:
    """Abstain when the best proxy score is at or below a threshold set on dev positives.

    The threshold is the 5th percentile of the dev positives' top leaf score, so that
    about 95% of dev positives are answered. Because the safety shortlist keeps the
    global top record, the top score and hence the decision are the same for flat
    search and Algorithm 1.
    """
    def top(q):
        return float(track.flat(q)["top_leaf_score"])
    dev_scores = sorted(top(q) for q in dev)
    threshold = dev_scores[int(0.05 * (len(dev_scores) - 1))]
    cfg = replace(config, candidate_budget=3)
    test_rows = [track.algorithm1(q, cfg) for q in test]
    answered = [r for r in test_rows if top_score(track, r) > threshold]
    neg_scores = [top(q) for q in negatives]
    return {
        "rule": "abstain if top leaf score <= 5th percentile of dev-positive top scores",
        "threshold": threshold,
        "test_positive_answer_rate": len(answered) / len(test_rows),
        "test_positive_hit_at_3_with_abstention": fmean(
            float(r["hit_at_3"]) if top_score(track, r) > threshold else 0.0 for r in test_rows
        ),
        "negative_count": len(negatives),
        "negative_accuracy": sum(s <= threshold for s in neg_scores) / len(neg_scores),
        "negative_top_scores": neg_scores,
    }


def top_score(track: Track, row: dict[str, Any]) -> float:
    return float(row["top_leaf_score"])


def _strip(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "flat": res["flat"]["summary"],
        "selected_budget": res["selected_budget"],
        "budgets": {k: {"algorithm1": v["algorithm1"], "flat_top_k": v["flat_top_k"]} for k, v in res["budgets"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/frontier")
    args = parser.parse_args()
    queries = filter_five_scene_queries(load_benchmark_queries(args.benchmark))
    split = _frozen_split(queries)
    scenes = _load_scenes(args.scene_root)
    encoder = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    summary: dict[str, Any] = {"tokenizer": "tiktoken cl100k_base", "split_counts": {k: len(v) for k, v in split.items()}}
    per_query: dict[str, Any] = {}
    for name, base in (("lexical", LEXICAL), ("bge", BGE)):
        track = Track(name, scenes, queries, encoder)
        dev = _frontier(track, split["dev"], base)
        test = _frontier(track, split["test"], base)
        k_dev = dev["selected_budget"]
        summary[name] = {
            "config": asdict(base),
            "dev_selected_budget": k_dev,
            "test_selected_budget_original_protocol": test["selected_budget"],
            "dev": _strip(dev),
            "test": _strip(test),
            "ablation_dev": _ablation(track, split["dev"], base, 3),
            "ablation_test": _ablation(track, split["test"], base, 3),
            "abstention": _abstention(track, split["dev"], split["test"], split["negative_audit"], base),
        }
        per_query[name] = {
            "test_k3_algorithm1": test["budgets"][3]["algorithm1_rows"],
            "test_k3_flat_top_k": test["budgets"][3]["flat_top_k_rows"],
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (args.out / "per_query.json").write_text(json.dumps(per_query, indent=1, default=str) + "\n", encoding="utf-8")
    for name in ("lexical", "bge"):
        s = summary[name]
        t3 = s["test"]["budgets"][3]
        print(name, "k_dev", s["dev_selected_budget"], "k_test", s["test_selected_budget_original_protocol"])
        print("  flat", {k: round(v, 3) for k, v in s["test"]["flat"].items() if isinstance(v, float)})
        print("  alg1", {k: round(v, 3) for k, v in t3["algorithm1"].items() if isinstance(v, float)})
        print("  topk", {k: round(v, 3) for k, v in t3["flat_top_k"].items() if isinstance(v, float)})
        print("  disagree hit3", t3["algorithm1"]["hit3_disagreements_vs_flat_top_k"],
              "hit1", t3["algorithm1"]["hit1_disagreements_vs_flat_top_k"])
        print("  mrr vs topk", t3["algorithm1"]["paired_vs_flat_top_k"]["mrr"]["estimate"],
              t3["algorithm1"]["paired_vs_flat_top_k"]["mrr"]["ci_95"])
        print("  abstain", {k: v for k, v in s["abstention"].items() if k != "negative_top_scores"})
        print("  ablation dev", [(a["stage"], round(a["hit_at_3"], 3), round(a["retained"], 2)) for a in s["ablation_dev"]])
        print("  ablation test", [(a["stage"], round(a["hit_at_3"], 3), round(a["retained"], 2)) for a in s["ablation_test"]])


if __name__ == "__main__":
    main()
