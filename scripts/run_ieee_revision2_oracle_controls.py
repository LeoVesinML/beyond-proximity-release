#!/usr/bin/env python3
"""Revision-2 oracle-map controls (ScanNet core / extended, Replica).

1. Same-scorer comparison: the predicted-map flat BGE scorer
   (scripts/run_predicted_map_pair.py: full query text vs object text, BGE-small,
   argmax) applied to the ORACLE object maps.
2. Algorithm 1 (backend/query/recall_controlled.py semantics: cached leaf
   proxies, max-leaf branch scores, cumulative softmax mass up to a cap,
   global top-m safety shortlist, shared leaf scorer, budget k) over the
   oracle maps with the same object scorer used by the v3 "graph" rows
   (backend.query.grounding lexical score + gated relation reranking).
   The oracle trees are flat (root -> object nodes), so the hierarchy is the
   deterministic spatial median split used for predicted maps (<=12 objects).
3. B11 constant sweeps (relation boost, fallback confidence threshold) on the
   ScanNet core queries with the unchanged v3 pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_grounding_variants as rgv  # noqa: E402
from backend.query.agent_protocol import FastEmbedSemanticEncoder  # noqa: E402
from backend.query.grounding import (  # noqa: E402
    GroundingSearchResult, _fresh, _rank_lexical, build_candidates, lexical_score,
    parse_relation_query, rerank_relations, search_grounding,
)
from backend.query.recall_controlled import wilson_interval  # noqa: E402
from scripts.evaluate_results import best_bbox_iou_3d, box_bounds, expected_3d_boxes  # noqa: E402
from scripts.run_experiment import load_scene  # noqa: E402
from scripts.run_predicted_map_pair import _select_branches, _split_spatial  # noqa: E402

CONFIG_DIR = ROOT / "outputs/ieee_revision2/predicted_maps/configs"


def _center(candidate: Any) -> np.ndarray:
    lo_hi = box_bounds(candidate.bbox_3d) if candidate.bbox_3d else None
    if lo_hi is None:
        return np.zeros(3)
    return (np.asarray(lo_hi[0]) + np.asarray(lo_hi[1])) / 2.0


def _softmax_mass_select(scores: np.ndarray, temperature: float, mass: float, cap: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    maximum = float(scores.max())
    weights = np.exp((scores - maximum) / temperature)
    probs = weights / weights.sum()
    chosen, cumulative = [], 0.0
    for index in order:
        chosen.append(index)
        cumulative += float(probs[index])
        if cumulative >= mass or len(chosen) >= cap:
            break
    return chosen


def make_algorithm1_search(k: int, m: int, temperature: float, mass: float, cap: int, max_leaf: int):
    def search(bundle: dict[str, Any], query: str, **options: Any) -> GroundingSearchResult:
        canonical = build_candidates(bundle["views"], bundle["tree"])
        parsed = parse_relation_query(query)
        target = parsed.target or query
        proxies = np.asarray([lexical_score(c, target)[0] for c in canonical], float)
        centers = np.vstack([_center(c) for c in canonical])
        branches = _split_spatial(list(range(len(canonical))), centers, max_leaf_size=max_leaf)
        branch_scores = np.asarray([max(proxies[i] for i in b) for b in branches])
        chosen = _softmax_mass_select(branch_scores, temperature, mass, cap)
        universe = {i for b in chosen for i in branches[b]}
        global_order = sorted(range(len(canonical)), key=lambda i: (-proxies[i], canonical[i].object_id))
        safety = [i for i in global_order if proxies[i] > 0][:m]
        universe.update(safety)
        initial = [canonical[i] for i in sorted(universe)]
        ranked = _rank_lexical(_fresh(initial), target)
        ranked, unresolved, rejected = rerank_relations(ranked, _fresh(canonical), parsed, use_relation=True)
        status = "ok" if ranked and ranked[0].score > 0 else "no_match"
        return GroundingSearchResult(
            status=status, variant="algorithm1", query=query, parsed_query=parsed,
            candidates=ranked[:k], candidate_count=len(canonical), searched_candidate_count=len(initial),
            metadata={"initial_candidate_ids": [c.object_id for c in initial],
                      "same_input_candidate_ids": [c.object_id for c in canonical],
                      "algorithm1": {"branch_count": len(branches), "selected_branches": chosen,
                                     "safety_ids": [canonical[i].object_id for i in safety],
                                     "proxy_comparisons": len(canonical), "k": k, "m": m,
                                     "temperature": temperature, "mass": mass, "cap": cap,
                                     "relation_unresolved": unresolved}},
        )
    return search


def run_with_search(track: str, out: Path, run_name: str, options: dict[str, Any], search_fn=None) -> dict[str, Any]:
    config = json.loads((CONFIG_DIR / f"{track}.json").read_text())
    benchmark = json.loads(Path(config["benchmark_path"]).read_text())
    original = rgv.search_grounding
    if search_fn is not None:
        rgv.search_grounding = search_fn
    try:
        summary = rgv.run_variant(run_name=run_name, options=options, config=config, benchmark=benchmark,
                                  output_root=out, embedder=None, embedding_model="", embedding_cache=ROOT / ".cache/fastembed",
                                  limit=None)
    finally:
        rgv.search_grounding = original
    (out / run_name / "actual_search_options.json").write_text(json.dumps(
        {"options_passed_to_search": options, "custom_search": search_fn is not None,
         "note": "run_config.json fixed_search_policy is hard-coded metadata in run_grounding_variants.py; "
                 "the values actually used are the defaults overridden by options_passed_to_search"}, indent=2))
    return compact(summary)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    m = summary["metrics"]
    return {
        "n": m["query_count"],
        "recall_at_1": m["recall_at_1_exact_object_id"]["value"],
        "recall_at_3": m["recall_at_3_exact_object_id"]["value"],
        "acc_at_0_25": m["acc_at_0_25"]["value"], "acc_at_0_25_num_den": [m["acc_at_0_25"]["numerator"], m["acc_at_0_25"]["denominator"]],
        "acc_at_0_5": m["acc_at_0_5"]["value"],
        "negative_accuracy": m["negative_accuracy"]["value"],
        "objects_per_query": m["efficiency"]["checked_objects"]["mean"],
        "estimated_tokens_per_query": m["efficiency"]["estimated_tokens"]["mean"],
    }


def same_scorer_bge(track: str, encoder: FastEmbedSemanticEncoder, text_mode: str) -> dict[str, Any]:
    config = json.loads((CONFIG_DIR / f"{track}.json").read_text())
    queries = json.loads(Path(config["benchmark_path"]).read_text())["queries"]
    paths = {s["scene_id"]: Path(s["path"]) for s in config["scenes"]}
    rows = []
    for scene_id in sorted({q["scene_id"] for q in queries}):
        loaded = load_scene(paths[scene_id])
        cands = build_candidates(loaded["existing_views"], loaded["existing_tree"])
        texts = [c.label if text_mode == "label" else c.text for c in cands]
        vecs = encoder.encode(texts)
        centers = np.vstack([_center(c) for c in cands])
        branches = _split_spatial(list(range(len(cands))), centers)
        sq = [q for q in queries if q["scene_id"] == scene_id and any(box_bounds(b) for b in expected_3d_boxes(q))]
        qv = encoder.encode([str(q["query"]) for q in sq])
        for q, v in zip(sq, qv):
            sims = vecs @ v
            order = np.argsort(-sims, kind="stable")
            # Algorithm-1 style hierarchy from run_predicted_map_pair (mass 0.7, T 0.1, cap 5, m=1, k=3)
            bscores = np.asarray([max(float(sims[i]) for i in b) for b in branches])
            sel = _select_branches(bscores)
            universe = {i for b in sel for i in branches[b]} | {int(order[0])}
            hier = [int(i) for i in order if int(i) in universe][:3]
            ids = {str(x) for x in q.get("expected_object_ids", [])}
            def iou(i):
                return best_bbox_iou_3d(cands[i].bbox_3d, q) or 0.0
            rows.append({"query_id": q["query_id"], "flat_top1": cands[int(order[0])].object_id,
                         "flat_top1_iou": iou(int(order[0])), "flat_r1": cands[int(order[0])].object_id in ids,
                         "flat_r3": any(cands[int(i)].object_id in ids for i in order[:3]),
                         "hier_top1_iou": iou(hier[0]), "hier_checked": len(universe), "objects": len(cands)})
    n = len(rows)
    def rate(key, thr=None):
        k = sum((r[key] >= thr) if thr is not None else bool(r[key]) for r in rows)
        lo, hi = wilson_interval(k, n)
        return {"value": round(k / n, 6), "numerator": k, "denominator": n, "wilson_95": [round(lo, 6), round(hi, 6)]}
    return {"n": n, "object_text": text_mode,
            "flat_bge_acc_at_0_25": rate("flat_top1_iou", 0.25), "flat_bge_acc_at_0_5": rate("flat_top1_iou", 0.5),
            "flat_bge_recall_at_1_id": rate("flat_r1"), "flat_bge_recall_at_3_id": rate("flat_r3"),
            "hier_bge_acc_at_0_25": rate("hier_top1_iou", 0.25),
            "mean_objects_flat": round(float(np.mean([r["objects"] for r in rows])), 4),
            "mean_universe_hier": round(float(np.mean([r["hier_checked"] for r in rows])), 4),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/predicted_maps/oracle_controls")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema_version": "beyond_proximity.ieee_revision2.oracle_controls.v1"}

    encoder = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    result["same_scorer_flat_bge_oracle"] = {
        f"{track}:{mode}": same_scorer_bge(track, encoder, mode)
        for track in ("scannet_core", "replica") for mode in ("label", "candidate_text")}

    alg = {}
    for track in ("scannet_core", "scannet_extended", "replica"):
        alg[track] = {
            "algorithm1_k3_m1_T0.25_mass0.7_cap5": run_with_search(
                track, out / "algorithm1" / track, "algorithm1_k3_m1", {"variant": "graph"},
                make_algorithm1_search(k=3, m=1, temperature=0.25, mass=0.7, cap=5, max_leaf=12)),
            "v3_graph_reference": run_with_search(track, out / "algorithm1" / track, "v3_graph", {"variant": "graph"}),
            "v3_flat_lexical_reference": run_with_search(track, out / "algorithm1" / track, "v3_flat_lexical", {"variant": "flat_lexical"}),
        }
    result["B8_algorithm1_oracle"] = alg

    sweeps: dict[str, Any] = {"relation_boost (confidence_threshold=0.45)": {}, "confidence_threshold (relation_boost=0.20)": {}}
    for beta in (0.20, 0.35, 0.50):
        sweeps["relation_boost (confidence_threshold=0.45)"][str(beta)] = {
            v: run_with_search("scannet_core", out / "b11" / f"boost_{beta}", v, {"variant": v, "relation_boost": beta})
            for v in ("graph", "graph_fallback", "flat_lexical")}
    for thr in (0.35, 0.45, 0.55):
        sweeps["confidence_threshold (relation_boost=0.20)"][str(thr)] = {
            v: run_with_search("scannet_core", out / "b11" / f"threshold_{thr}", v, {"variant": v, "confidence_threshold": thr})
            for v in ("graph", "graph_fallback", "flat_lexical")}
    result["B11_sweeps_scannet_core"] = sweeps
    (out / "oracle_controls_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    printable = json.loads(json.dumps(result))
    for v in printable["same_scorer_flat_bge_oracle"].values():
        v.pop("rows")
    print(json.dumps(printable, indent=1))


if __name__ == "__main__":
    main()
