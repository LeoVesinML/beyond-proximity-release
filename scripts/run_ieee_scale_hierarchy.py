#!/usr/bin/env python3
"""Recall-controlled scene hierarchy (Algorithm 1) on the ReferIt3D scale pools.

Reuses the record loader, embedding cache, query sampler, and quality function of
``scripts/run_ieee_public_scale_ann.py`` (imported, not modified) so pools, query
records, leakage control, and relevance are identical to public_scale_ann_v1.

No oracle scan ID is used. Every record's branch is its ScanNet scan ID. Three
branch-proxy variants are evaluated with the adaptive softmax mass gate of
``backend/query/recall_controlled.py`` (``_choose_branches``, adaptive_mass):

* ``maxleaf``   - branch score = max exact BGE cosine over the branch's records
                  (exact, O(N) proxy work).
* ``summary``   - branch score = cosine to the normalized mean embedding of the
                  branch's records (O(#branches)); safety shortlist = HNSW top-m.
* ``hnswproxy`` - branch score = max similarity among the HNSW top-L results that
                  fall in the branch (branches with no HNSW hit get zero mass);
                  safety shortlist = HNSW top-m.

Final ranking: exact BGE cosine inside the candidate set C, top k=3.
Configuration is chosen on a tuning query subset disjoint from all evaluation
queries, then frozen for the evaluation queries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_ieee_public_scale_ann as base  # noqa: E402  (imported unchanged)
from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL  # noqa: E402
from backend.query.recall_controlled import paired_bootstrap_difference, wilson_interval  # noqa: E402

SEED = 20260901
TAUS = (0.5, 0.7, 0.9)
TEMPERATURES = (0.01, 0.02, 0.05, 0.1)
CAPS = (1, 3, 5, 10)
SHORTLISTS = (0, 1, 3)
VARIANTS = ("maxleaf", "summary", "hnswproxy")
HNSW_SHORTLIST_COUNT = 12  # same search count as the hnsw_dense baseline
HNSW_PROXY_COUNT = 64
TOP_K = 3
TIMING_REPS = 5
NEAR_TIE = 1e-6
MAX_CAP = max(CAPS)


# --------------------------------------------------------------------------- gate
def top_branches(branch_scores: np.ndarray, kth: int) -> np.ndarray:
    """Top-kth branch indices by (score desc, branch index asc); deterministic under ties."""
    if kth <= 0:
        return np.empty(0, dtype=np.int64)
    if kth < len(branch_scores):
        threshold = np.partition(branch_scores, len(branch_scores) - kth)[len(branch_scores) - kth]
        pool = np.nonzero(branch_scores >= threshold)[0]
    else:
        pool = np.arange(len(branch_scores))
    order = np.lexsort((pool, -branch_scores[pool]))[:kth]
    return pool[order]


def gate(branch_scores: np.ndarray, temperature: float, tau: float, cap: int) -> list[int]:
    """Adaptive softmax mass gate, same stopping rule as recall_controlled._choose_branches.

    Branches are visited in descending score order; a branch is added, its softmax
    probability accumulated, and traversal stops once cumulative mass >= tau or the
    cap is reached. Branches with -inf score (empty after query exclusion, or not
    reached by HNSW) receive zero probability and are never selected.
    """
    finite = np.isfinite(branch_scores)
    count = int(finite.sum())
    if count == 0:
        return []
    kth = min(cap, count)
    top = top_branches(branch_scores, kth)
    maximum = branch_scores[top[0]]
    weights = np.exp((branch_scores[finite] - maximum) / temperature)
    total = float(weights.sum()) or 1.0
    cumulative = np.cumsum(np.exp((branch_scores[top] - maximum) / temperature) / total)
    reached = np.nonzero(cumulative >= tau)[0]
    stop = int(reached[0]) + 1 if len(reached) else kth
    return [int(value) for value in top[: min(stop, cap)]]


def gate_prefix_lengths(branch_scores: np.ndarray) -> tuple[np.ndarray, dict[tuple[float, float, int], int]]:
    """Top-MAX_CAP branch order plus the selected prefix length for every (T, tau, B)."""
    finite = np.isfinite(branch_scores)
    count = int(finite.sum())
    kth = min(MAX_CAP, count)
    if kth == 0:
        return np.empty(0, dtype=np.int64), {key: 0 for key in itertools.product(TEMPERATURES, TAUS, CAPS)}
    top = top_branches(branch_scores, kth)
    maximum = branch_scores[top[0]]
    lengths: dict[tuple[float, float, int], int] = {}
    finite_scores = branch_scores[finite]
    for temperature in TEMPERATURES:
        total = float(np.exp((finite_scores - maximum) / temperature).sum()) or 1.0
        cumulative = np.cumsum(np.exp((branch_scores[top] - maximum) / temperature) / total)
        for tau in TAUS:
            reached = np.nonzero(cumulative >= tau)[0]
            for cap in CAPS:
                limit = min(cap, kth)
                stop = int(reached[0]) + 1 if len(reached) else limit
                lengths[(temperature, tau, cap)] = min(stop, limit)
    return top, lengths


# --------------------------------------------------------------------------- pool
class Pool:
    def __init__(self, records: list[dict[str, Any]], embeddings: np.ndarray, size: int):
        from usearch.index import Index

        self.size = size
        self.matrix = embeddings[:size]  # view of the shared in-memory float32 array
        scene_ids = [str(row["scene_id"]) for row in records[:size]]
        self.scene_names = sorted(set(scene_ids))
        scene_index = {name: i for i, name in enumerate(self.scene_names)}
        self.scene_of = np.array([scene_index[name] for name in scene_ids], dtype=np.int64)
        order = np.argsort(self.scene_of, kind="stable")
        self.order = order  # sorted position -> original index
        self.position = np.empty(size, dtype=np.int64)
        self.position[order] = np.arange(size)
        self.sorted_matrix = np.ascontiguousarray(self.matrix[order])
        counts = np.bincount(self.scene_of, minlength=len(self.scene_names))
        self.counts = counts
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.ends = (self.starts + counts).astype(np.int64)
        self.sums = np.add.reduceat(self.sorted_matrix, self.starts, axis=0).astype(np.float32)
        norms = np.linalg.norm(self.sums, axis=1, keepdims=True)
        self.summaries = np.ascontiguousarray(self.sums / np.maximum(norms, 1e-12))
        object_ids: dict[tuple[str, str], int] = {}
        self.object_of_sorted = np.array([
            object_ids.setdefault((str(records[int(i)]["scene_id"]), str(records[int(i)]["object_id"])), len(object_ids))
            for i in order
        ], dtype=np.int64)
        self.object_ids = object_ids
        started = time.perf_counter()
        self.ann = Index(ndim=int(self.matrix.shape[1]), metric="cos", dtype="f32",
                         connectivity=16, expansion_add=128, expansion_search=64)
        self.ann.add(np.arange(size, dtype=np.uint64), self.matrix, threads=0)
        self.ann_build_seconds = time.perf_counter() - started

    # --- proxies (shared by grid and timed paths) ---------------------------
    def summary_scores(self, vector: np.ndarray, query_pos: int) -> np.ndarray:
        scores = self.summaries @ vector
        own = int(self.scene_of[self.order[query_pos]])
        # Leave-one-out: remove the query's own vector from its branch summary so the
        # query cannot vote for its own scene (equivalent to indexing without it).
        if self.counts[own] <= 1:
            scores[own] = -np.inf
        else:
            reduced = self.sums[own] - vector
            scores[own] = float(reduced @ vector) / max(float(np.linalg.norm(reduced)), 1e-12)
        return scores

    def hnsw(self, vector: np.ndarray, query_index: int, count: int) -> tuple[np.ndarray, np.ndarray, int]:
        matches = self.ann.search(vector, count=min(count, self.size))
        keys = np.asarray(matches.keys, dtype=np.int64)
        sims = 1.0 - np.asarray(matches.distances, dtype=np.float32)
        keep = keys != query_index
        return keys[keep], sims[keep], int(matches.computed_distances)

    def hnsw_proxy_scores(self, keys: np.ndarray, sims: np.ndarray) -> np.ndarray:
        scores = np.full(len(self.scene_names), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.scene_of[keys], sims)
        return scores


# --------------------------------------------------------------------------- timed pipelines
def timed_flat(pool: Pool, vector: np.ndarray, query_index: int) -> tuple[list[int], float]:
    started = time.perf_counter()
    ranked = base._top_indices(pool.matrix @ vector, TOP_K, query_index)
    return ranked, (time.perf_counter() - started) * 1000


def _rank_candidates(pool: Pool, positions: np.ndarray, scores: np.ndarray) -> list[int]:
    """Top-k of C by (score desc, sorted position asc), same tie rule as canonical_top."""
    if len(positions) == 0:
        return []
    if len(positions) > TOP_K:
        threshold = np.partition(scores, len(scores) - TOP_K)[len(scores) - TOP_K]
        keep = scores >= threshold
        positions, scores = positions[keep], scores[keep]
    order = np.lexsort((positions, -scores))[:TOP_K]
    return [int(pool.order[positions[i]]) for i in order]


def timed_hierarchy(pool: Pool, variant: str, config: dict[str, Any], vector: np.ndarray,
                    query_index: int) -> dict[str, Any]:
    """Run Algorithm 1 end to end for one query; returns ranked original indices and counters."""
    temperature, tau, cap, m = config["T"], config["tau"], config["B"], config["m"]
    query_pos = int(pool.position[query_index])
    started = time.perf_counter()
    hnsw_distances = 0
    if variant == "maxleaf":
        leaf = pool.sorted_matrix @ vector
        leaf[query_pos] = -np.inf
        branch = np.maximum.reduceat(leaf, pool.starts)
        chosen = gate(branch, temperature, tau, cap)
        parts = [np.arange(pool.starts[b], pool.ends[b]) for b in chosen]
        if m:
            short = canonical_top(leaf, np.arange(pool.size), m)
            parts.append(short)
        positions = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        positions = positions[positions != query_pos]
        scores = leaf[positions]
        proxy = pool.size
        leaf_count = int(sum(pool.counts[b] for b in chosen)) + int(
            sum(1 for p in (short if m else []) if int(pool.scene_of[pool.order[p]]) not in set(chosen)))
    else:
        if variant == "summary":
            branch = pool.summary_scores(vector, query_pos)
            keys, sims, hnsw_distances = (pool.hnsw(vector, query_index, HNSW_SHORTLIST_COUNT)
                                          if m else (np.empty(0, np.int64), None, 0))
            proxy = len(pool.scene_names)
        else:
            keys, sims, hnsw_distances = pool.hnsw(vector, query_index, HNSW_PROXY_COUNT)
            branch = pool.hnsw_proxy_scores(keys, sims)
            proxy = 0
        chosen = gate(branch, temperature, tau, cap)
        blocks = [np.arange(pool.starts[b], pool.ends[b]) for b in chosen]
        short_pos = pool.position[keys[:m]] if m else np.empty(0, dtype=np.int64)
        chosen_set = set(chosen)
        extra = np.array([p for p in short_pos if int(pool.scene_of[pool.order[p]]) not in chosen_set], dtype=np.int64)
        positions = np.concatenate(blocks + [extra]) if (blocks or len(extra)) else np.empty(0, dtype=np.int64)
        positions = positions[positions != query_pos]
        scores = pool.sorted_matrix[positions] @ vector
        leaf_count = int(sum(pool.counts[b] for b in chosen)) + len(extra)
    ranked = _rank_candidates(pool, positions, scores)
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "ranked": ranked,
        "latency_ms": elapsed,
        "proxy_records": proxy,
        "hnsw_computed_distances": hnsw_distances,
        "leaf_records": leaf_count,
        "selected_branches": len(chosen),
        "query_scene_selected": int(pool.scene_of[query_index]) in set(chosen),
    }


# --------------------------------------------------------------------------- grid (untimed)
def grid_query(pool: Pool, vector: np.ndarray, query_index: int) -> dict[str, Any]:
    """All-configuration outcomes for one query from one exact score vector (untimed)."""
    query_pos = int(pool.position[query_index])
    target = int(pool.object_of_sorted[query_pos])
    leaf = pool.sorted_matrix @ vector
    leaf[query_pos] = -np.inf

    flat_top = canonical_top(leaf, np.arange(pool.size))
    ordered4 = canonical_top(leaf, np.arange(pool.size), TOP_K + 1)
    top4_scores = leaf[ordered4]
    top_cache: dict[int, np.ndarray] = {}

    def scene_top(b: int) -> np.ndarray:
        if b not in top_cache:
            block = np.arange(pool.starts[b], pool.ends[b])
            block = block[block != query_pos]
            top_cache[b] = canonical_top(leaf, block)
        return top_cache[b]

    hnsw_keys, hnsw_sims, hnsw_dist = pool.hnsw(vector, query_index, HNSW_SHORTLIST_COUNT)
    proxy_keys, proxy_sims, proxy_dist = pool.hnsw(vector, query_index, HNSW_PROXY_COUNT)
    variants = {
        "maxleaf": (np.maximum.reduceat(leaf, pool.starts), flat_top),
        "summary": (pool.summary_scores(vector, query_pos), pool.position[hnsw_keys]),
        "hnswproxy": (pool.hnsw_proxy_scores(proxy_keys, proxy_sims), pool.position[proxy_keys]),
    }
    out: dict[str, Any] = {
        "flat": _outcome(pool, list(flat_top), leaf, target),
        "flat_ranked": [int(pool.order[p]) for p in flat_top],
        # near-tie diagnostics: gap between rank-3 and rank-4 score, and rank-1 vs rank-2
        "gap_3_4": float(top4_scores[2] - top4_scores[3]) if len(top4_scores) > 3 else None,
        "gap_1_2": float(top4_scores[0] - top4_scores[1]) if len(top4_scores) > 1 else None,
        "flat_all_other_scene": all(int(pool.scene_of[pool.order[p]]) != int(pool.scene_of[query_index])
                                    for p in flat_top),
        "hnsw_distances": {"summary": hnsw_dist, "hnswproxy": proxy_dist},
        "variants": {},
    }
    for name, (branch, shortlist) in variants.items():
        top, lengths = gate_prefix_lengths(branch)
        by_prefix: dict[tuple[int, int], tuple[bool, bool, int]] = {}
        for n in sorted(set(lengths.values())):
            chosen = [int(b) for b in top[:n]]
            chosen_set = set(chosen)
            base_positions = [int(p) for b in chosen for p in scene_top(b)]
            base_leaf = int(sum(pool.counts[b] for b in chosen))
            for m in SHORTLISTS:
                extra = [int(p) for p in shortlist[:m]
                         if int(pool.scene_of[pool.order[p]]) not in chosen_set and int(p) != query_pos]
                cands = np.array(list(dict.fromkeys(base_positions + extra)), dtype=np.int64)
                hit1, hit3 = _outcome(pool, [int(p) for p in canonical_top(leaf, cands)], leaf, target)
                by_prefix[(n, m)] = (hit1, hit3, base_leaf + len(extra))
        out["variants"][name] = {"lengths": lengths, "by_prefix": by_prefix}
    return out


def canonical_top(leaf: np.ndarray, positions: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Top-k positions by (score desc, sorted position asc); deterministic under ties.

    All quality numbers use this ranking over one canonical score vector per query
    (sorted_matrix @ q), so flat and hierarchical methods break ties identically.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return positions
    values = leaf[positions]
    if len(positions) > k:
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        keep = values >= threshold
        positions, values = positions[keep], values[keep]
    order = np.lexsort((positions, -values))[:k]
    return positions[order]


def _outcome(pool: Pool, positions: list[int], leaf: np.ndarray, target: int) -> tuple[bool, bool]:
    objects = [int(pool.object_of_sorted[p]) for p in positions]
    return (bool(objects and objects[0] == target), bool(target in objects))


def config_list() -> list[dict[str, Any]]:
    return [{"tau": tau, "T": t, "B": b, "m": m}
            for tau, t, b, m in itertools.product(TAUS, TEMPERATURES, CAPS, SHORTLISTS)]


def config_key(config: dict[str, Any]) -> str:
    return f"tau={config['tau']},T={config['T']},B={config['B']},m={config['m']}"


def grid_metrics(results: list[dict[str, Any]], variant: str, config: dict[str, Any]) -> dict[str, float]:
    hit1 = hit3 = leaf = 0.0
    for item in results:
        data = item["variants"][variant]
        n = data["lengths"][(config["T"], config["tau"], config["B"])]
        h1, h3, lf = data["by_prefix"][(n, config["m"])]
        hit1 += h1
        hit3 += h3
        leaf += lf
    count = len(results)
    return {"hit_at_1": hit1 / count, "hit_at_3": hit3 / count, "mean_leaf_records": leaf / count}


# --------------------------------------------------------------------------- helpers
def eligible_sorted(records: list[dict[str, Any]], size: int) -> list[int]:
    return base._query_indices(records, size, size)  # full hash-ordered eligible list


def summarize(values: list[dict[str, Any]], key: str) -> float:
    return round(fmean(float(row[key]) for row in values), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr3d", type=Path, default=ROOT / "external/datasets/scanrefer_public/nr3d.csv")
    parser.add_argument("--sr3d-plus", type=Path, default=ROOT / "external/datasets/scanrefer_public/sr3d/sr3d+.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "external/datasets/scanrefer_public/referit3d_bge_small.npy")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/scale_hierarchy")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--sizes", default="1000,10000,50000,100000,155536")
    parser.add_argument("--queries-per-size", type=int, default=500)
    parser.add_argument("--tuning-per-size", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    load_start = os.getloadavg()

    all_records = base._load_records(args.nr3d.resolve(), args.sr3d_plus.resolve())
    sizes_requested = [int(v) for v in args.sizes.split(",") if v.strip()]
    maximum = min(max(sizes_requested), len(all_records))
    records = all_records[:maximum]
    embeddings = np.ascontiguousarray(np.asarray(
        base._embedding_cache(records, args.cache.resolve(), args.model), dtype=np.float32))
    sizes = sorted({min(v, len(records)) for v in sizes_requested if v > 1})

    # ---- query sets (C4) ---------------------------------------------------
    eval_sets = {size: base._query_indices(records, size, args.queries_per_size) for size in sizes}
    eligible_counts = {}
    eval_union = {records[i]["record_key"] for s in sizes for i in eval_sets[s]}
    tuning_sets: dict[int, list[int]] = {}
    for size in sizes:
        eligible = eligible_sorted(records, size)
        eligible_counts[size] = len(eligible)
        tuning_sets[size] = [i for i in eligible[args.queries_per_size:]
                             if records[i]["record_key"] not in eval_union][: args.tuning_per_size]
    tuning_union = {records[i]["record_key"] for s in sizes for i in tuning_sets[s]}
    assert not (tuning_union & eval_union), "tuning and evaluation queries overlap"
    query_accounting = {
        "per_pool": [{
            "collection_size": s,
            "eligible_queries": eligible_counts[s],
            "evaluation_queries": len(eval_sets[s]),
            "tuning_queries": len(tuning_sets[s]),
        } for s in sizes],
        "evaluation_query_pool_pairs_total": sum(len(eval_sets[s]) for s in sizes),
        "unique_evaluation_query_records": len(eval_union),
        "tuning_query_pool_pairs_total": sum(len(tuning_sets[s]) for s in sizes),
        "unique_tuning_query_records": len(tuning_union),
        "tuning_evaluation_overlap": len(tuning_union & eval_union),
        "rule": ("per pool, eligible = records whose (scan_id, target_id) has >=2 expressions inside "
                 "that pool; eligible records are ordered by sha256(seed:query:record_key) and the "
                 "first min(500, eligible) are evaluated (base._query_indices)."),
        "tuning_rule": ("per pool, the eligible records after the first 500 in the same hash order, "
                        "excluding any record that is an evaluation query in any pool, first 500 kept"),
    }
    print(json.dumps(query_accounting, indent=2), flush=True)

    configs = config_list()
    tuning_results: list[dict[str, Any]] = []
    eval_results: dict[int, list[dict[str, Any]]] = {}
    pools: dict[int, Pool] = {}
    build_rows = []
    for size in sizes:
        pool = Pool(records, embeddings, size)
        build_rows.append({"collection_size": size, "branches": len(pool.scene_names),
                           "mean_records_per_branch": round(size / len(pool.scene_names), 3),
                           "hnsw_build_seconds": round(pool.ann_build_seconds, 3)})
        for qi in tuning_sets[size]:
            item = grid_query(pool, np.asarray(embeddings[qi], dtype=np.float32), qi)
            item.update({"collection_size": size, "query_index": qi})
            tuning_results.append(item)
        eval_results[size] = []
        for qi in eval_sets[size]:
            item = grid_query(pool, np.asarray(embeddings[qi], dtype=np.float32), qi)
            item.update({"collection_size": size, "query_index": qi})
            eval_results[size].append(item)
        pools[size] = pool  # kept so timing uses the very same HNSW graph as the grid
        print(f"grid done size={size} branches={len(pool.scene_names)}", flush=True)

    # ---- tuning & selection (evaluation queries not consulted) -------------
    flat_tune_hit3 = fmean(float(r["flat"][1]) for r in tuning_results)
    flat_tune_hit1 = fmean(float(r["flat"][0]) for r in tuning_results)
    tuning_table = []
    selected: dict[str, dict[str, Any]] = {}
    selected_none: list[dict[str, str]] = []
    for variant in VARIANTS:
        rows = []
        for config in configs:
            metrics = grid_metrics(tuning_results, variant, config)
            rows.append({"variant": variant, **config, **{k: round(v, 6) for k, v in metrics.items()}})
        tuning_table.extend(rows)
        cfg = lambda r: {k: r[k] for k in ("tau", "T", "B", "m")}
        by_acc = lambda r: (-r["hit_at_3"], -r["hit_at_1"], r["mean_leaf_records"])
        by_cost = lambda r: (r["mean_leaf_records"], -r["hit_at_3"], -r["hit_at_1"])
        flat3 = round(flat_tune_hit3, 6)
        choices = {
            # declared before the first run:
            "accuracy": sorted(rows, key=by_acc)[0],
            "budget": (sorted([r for r in rows if r["hit_at_3"] >= flat3], key=by_cost) or [None])[0],
            # added after inspecting tuning-only results of the first run (evaluation not viewed):
            "gate_only": sorted([r for r in rows if r["m"] == 0], key=by_acc)[0],
            "budget_1pp": (sorted([r for r in rows if r["hit_at_3"] >= flat3 - 0.01], key=by_cost) or [None])[0],
        }
        for rule, row in choices.items():
            if row is None:
                selected_none.append({"variant": variant, "rule": rule})
                continue
            duplicate = next((key for key, sel in selected.items()
                              if sel["variant"] == variant and sel["config"] == cfg(row)), None)
            if duplicate:
                selected[duplicate].setdefault("also_selected_by", []).append(rule)
                continue
            selected[f"{variant}_{rule}"] = {"variant": variant, "rule": rule, "config": cfg(row), "tuning": row}
    print(json.dumps({"flat_tuning_hit3": flat_tune_hit3, "selected": selected}, indent=2, default=str), flush=True)

    # ---- timed evaluation with frozen configs ------------------------------
    per_query: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    mismatches = Counter()
    for size in sizes:
        pool = pools[size]
        grid_by_q = {r["query_index"]: r for r in eval_results[size]}
        size_rows = []
        # warm-up (not recorded)
        for qi in eval_sets[size][:5]:
            v = np.asarray(embeddings[qi], dtype=np.float32)
            timed_flat(pool, v, qi)
            for sel in selected.values():
                timed_hierarchy(pool, sel["variant"], sel["config"], v, qi)
        for qi in eval_sets[size]:
            v = np.asarray(embeddings[qi], dtype=np.float32)
            query = records[qi]
            g = grid_by_q[qi]
            # Interleaved repetitions; per-query latency = min over TIMING_REPS runs
            # (CPU is shared with other jobs), first-run latency kept as well.
            flat_times, hier_times = [], defaultdict(list)
            for _rep in range(TIMING_REPS):
                ranked, flat_ms = timed_flat(pool, v, qi)
                flat_times.append(flat_ms)
                for name, sel in selected.items():
                    res = timed_hierarchy(pool, sel["variant"], sel["config"], v, qi)
                    hier_times[name].append((res["latency_ms"], res))
            timed_quality = base._quality(ranked, records, query)
            if (timed_quality["hit_at_1"], timed_quality["hit_at_3"]) != g["flat"]:
                mismatches["flat_dense_exact"] += 1
            size_rows.append({"method": "flat_dense_exact", "collection_size": size, "query_index": qi,
                              "query_record_key": query["record_key"], "hit_at_1": g["flat"][0],
                              "hit_at_3": g["flat"][1],
                              "timed_path_hit_at_1": timed_quality["hit_at_1"],
                              "timed_path_hit_at_3": timed_quality["hit_at_3"],
                              "latency_ms": round(min(flat_times), 6),
                              "first_run_latency_ms": round(flat_times[0], 6),
                              "proxy_records": 0, "leaf_records": size, "hnsw_computed_distances": 0,
                              "selected_branches": None, "query_scene_selected": None,
                              "top3_all_other_scene": g["flat_all_other_scene"],
                              "gap_3_4": g["gap_3_4"], "gap_1_2": g["gap_1_2"],
                              "ranked_record_keys": [records[i]["record_key"] for i in g["flat_ranked"]]})
            for name, sel in selected.items():
                times = [t for t, _ in hier_times[name]]
                res = hier_times[name][0][1]
                q = base._quality(res["ranked"], records, query)
                data = g["variants"][sel["variant"]]
                n = data["lengths"][(sel["config"]["T"], sel["config"]["tau"], sel["config"]["B"])]
                expected = data["by_prefix"][(n, sel["config"]["m"])]
                if (q["hit_at_1"], q["hit_at_3"]) != expected[:2]:
                    mismatches[name] += 1
                if res["leaf_records"] != expected[2]:
                    mismatches[name + "_leaf_count"] += 1
                size_rows.append({"method": f"hier_{name}", "collection_size": size, "query_index": qi,
                                  "query_record_key": query["record_key"], "hit_at_1": expected[0],
                                  "hit_at_3": expected[1],
                                  "timed_path_hit_at_1": q["hit_at_1"], "timed_path_hit_at_3": q["hit_at_3"],
                                  "latency_ms": round(min(times), 6),
                                  "first_run_latency_ms": round(times[0], 6),
                                  "proxy_records": res["proxy_records"], "leaf_records": res["leaf_records"],
                                  "hnsw_computed_distances": res["hnsw_computed_distances"],
                                  "selected_branches": res["selected_branches"],
                                  "query_scene_selected": res["query_scene_selected"],
                                  "top3_all_other_scene": None,
                                  "ranked_record_keys": [records[i]["record_key"] for i in res["ranked"]]})
        per_query.extend(size_rows)
        flat_rows = [r for r in size_rows if r["method"] == "flat_dense_exact"]
        misses = [r for r in flat_rows if not r["hit_at_3"]]
        for method in ["flat_dense_exact"] + [f"hier_{n}" for n in selected]:
            rows = [r for r in size_rows if r["method"] == method]
            count = len(rows)
            item = {
                "method": method, "collection_size": size, "query_count": count,
                "hit_at_1": summarize(rows, "hit_at_1"), "hit_at_3": summarize(rows, "hit_at_3"),
                "median_latency_ms": round(float(np.median([r["latency_ms"] for r in rows])), 6),
                "mean_latency_ms": summarize(rows, "latency_ms"),
                "median_first_run_latency_ms": round(float(np.median([r["first_run_latency_ms"] for r in rows])), 6),
                "timed_path_hit_at_3": summarize(rows, "timed_path_hit_at_3"),
                "mean_proxy_records": summarize(rows, "proxy_records"),
                "mean_hnsw_computed_distances": summarize(rows, "hnsw_computed_distances"),
                "mean_leaf_records": summarize(rows, "leaf_records"),
                # distinct vector comparisons: max-leaf reuses its O(N) proxy scores as leaf scores
                "mean_scored_records": round(fmean(float(
                    r["proxy_records"] + r["hnsw_computed_distances"]
                    + (0 if method.startswith("hier_maxleaf") else r["leaf_records"])) for r in rows), 6),
                "median_leaf_records": float(np.median([r["leaf_records"] for r in rows])),
            }
            for metric in ("hit_at_1", "hit_at_3"):
                lo, hi = wilson_interval(sum(bool(r[metric]) for r in rows), count)
                item[f"{metric}_wilson_95"] = {"low": round(lo, 6), "high": round(hi, 6)}
            if method != "flat_dense_exact":
                sel = selected[method[len("hier_"):]]
                item["variant"] = sel["variant"]
                item["selection_rule"] = sel["rule"]
                item["config"] = sel["config"]
                item["mean_selected_branches"] = summarize(rows, "selected_branches")
                item["query_scene_selected_rate"] = round(fmean(float(r["query_scene_selected"]) for r in rows), 6)
                for metric in ("hit_at_1", "hit_at_3"):
                    item[f"paired_{metric}_difference_vs_flat"] = paired_bootstrap_difference(
                        [float(r[metric]) for r in rows], [float(r[metric]) for r in flat_rows],
                        samples=args.bootstrap_samples, seed=SEED)
            else:
                item["near_tie_rank3_rank4_queries"] = sum(1 for r in rows if r["gap_3_4"] is not None and r["gap_3_4"] < NEAR_TIE)
                item["near_tie_rank1_rank2_queries"] = sum(1 for r in rows if r["gap_1_2"] is not None and r["gap_1_2"] < NEAR_TIE)
                item["flat_misses_at_3"] = len(misses)
                item["flat_misses_top3_all_other_scene"] = sum(bool(r["top3_all_other_scene"]) for r in misses)
                item["flat_misses_top3_all_other_scene_fraction"] = (
                    round(item["flat_misses_top3_all_other_scene"] / len(misses), 6) if misses else None)
            summaries.append(item)
        print(f"timed done size={size}", flush=True)

    # ---- eval-set sensitivity over the full grid (reported, never used for selection)
    sensitivity = []
    for size in sizes:
        flat_h3 = fmean(float(r["flat"][1]) for r in eval_results[size])
        for variant in VARIANTS:
            for config in configs:
                m = grid_metrics(eval_results[size], variant, config)
                sensitivity.append({"collection_size": size, "variant": variant, **config,
                                    "flat_hit_at_3": round(flat_h3, 6), **{k: round(v, 6) for k, v in m.items()}})

    all_misses = [r for r in per_query if r["method"] == "flat_dense_exact" and not r["hit_at_3"]]
    manifest = {
        "nr3d": {"path": str(args.nr3d.resolve().relative_to(ROOT)), "sha256": base._sha256(args.nr3d.resolve()), "source_url": base.SOURCE_URLS["nr3d"]},
        "sr3d_plus": {"path": str(args.sr3d_plus.resolve().relative_to(ROOT)), "sha256": base._sha256(args.sr3d_plus.resolve()), "source_url": base.SOURCE_URLS["sr3d_plus"]},
        "embedding_cache": {"path": str(args.cache.resolve().relative_to(ROOT)), "sha256": base._sha256(args.cache.resolve())},
        "deduplicated_records": len(records),
        "scene_count": len({r["scene_id"] for r in records}),
        "object_count": len({(r["scene_id"], r["object_id"]) for r in records}),
    }
    result = {
        "schema_version": "beyond_proximity.ieee_scale_hierarchy.v1",
        "generated_at": generated_at,
        "status": "complete",
        "model_name": args.model,
        "seed": SEED,
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.platform(), "numpy": np.__version__, "cpu_count": os.cpu_count(),
                     "load_average_start": load_start, "load_average_end": os.getloadavg(),
                     "timing": f"per query min over {TIMING_REPS} interleaved repetitions after 5 warm-up queries; single-thread Python, numpy default BLAS threads"},
        "data_manifest": manifest,
        "query_accounting": query_accounting,
        "grid": {"tau": TAUS, "T": TEMPERATURES, "B": CAPS, "m": SHORTLISTS,
                 "hnsw_shortlist_search_count": HNSW_SHORTLIST_COUNT, "hnsw_proxy_search_count": HNSW_PROXY_COUNT,
                 "hnsw_params": {"connectivity": 16, "expansion_add": 128, "expansion_search": 64, "metric": "cos"}},
        "tuning": {"flat_hit_at_1": round(flat_tune_hit1, 6), "flat_hit_at_3": round(flat_tune_hit3, 6),
                   "queries": len(tuning_results),
                   "selection_rules": {
                       "accuracy": "max tuning hit@3, ties by hit@1 then fewer mean leaf records",
                       "budget": "fewest mean leaf records among configs with tuning hit@3 >= flat tuning hit@3",
                       "gate_only": "accuracy rule restricted to m=0 (no safety shortlist); added after tuning-only inspection",
                       "budget_1pp": "fewest mean leaf records with tuning hit@3 >= flat tuning hit@3 - 0.01; added after tuning-only inspection",
                   },
                   "selected": selected,
                   "rules_with_no_qualifying_config": selected_none},
        "protocol": {
            "quality_source": ("hit@1/hit@3 for every method (flat included) come from one canonical exact "
                               "score vector per query (pool sorted_matrix @ q) with deterministic tie-breaking "
                               "(score desc, position asc), so flat and hierarchy break near-ties identically"),
            "timed_path": ("latency and scored-record counts come from separate end-to-end executions; "
                           "timed_path_hit_* re-derive quality from those executions (flat uses the unchanged "
                           "base._top_indices); disagreements are counted in timed_vs_grid_mismatches"),
            "oracle_scan_id_used": False,
            "query_exclusion": ("query record excluded from leaves; summary variant removes the query vector "
                                "from its own branch mean (leave-one-out); HNSW results drop the query key"),
            "record_count_convention": "leaf_records counts all records of selected branches (query included, as in scene_scoped_dense_exact) plus shortlist records outside them",
        },
        "timed_vs_grid_mismatches": dict(mismatches),
        "pools": build_rows,
        "summaries": summaries,
        "flat_miss_scene_analysis_all_pools": {
            "flat_misses_at_3": len(all_misses),
            "top3_all_other_scene": sum(bool(r["top3_all_other_scene"]) for r in all_misses),
            "fraction": round(sum(bool(r["top3_all_other_scene"]) for r in all_misses) / len(all_misses), 6) if all_misses else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(per_query, indent=1) + "\n", encoding="utf-8")
    for name, rows in (("tuning_grid.csv", tuning_table), ("eval_grid_sensitivity.csv", sensitivity)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    curve_fields = ["method", "collection_size", "query_count", "hit_at_1", "hit_at_3", "median_latency_ms",
                    "mean_proxy_records", "mean_hnsw_computed_distances", "mean_leaf_records", "mean_scored_records"]
    with (out / "scale_hierarchy_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields + ["config"])
        writer.writeheader()
        for row in summaries:
            writer.writerow({**{k: row[k] for k in curve_fields}, "config": config_key(row["config"]) if "config" in row else ""})
    print(json.dumps({"mismatches": dict(mismatches), "summaries": [
        {k: s[k] for k in ("method", "collection_size", "hit_at_1", "hit_at_3", "median_latency_ms",
                           "mean_proxy_records", "mean_leaf_records")} for s in summaries]}, indent=1))


if __name__ == "__main__":
    main()
