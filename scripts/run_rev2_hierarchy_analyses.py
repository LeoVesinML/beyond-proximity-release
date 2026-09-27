#!/usr/bin/env python3
"""IEEE revision 2: hierarchy-construction audit, uncertainty, Algorithm 1 grid, and sensitivity.

Writes everything to outputs/ieee_revision2/hierarchy/. Every number in the
accompanying RESULTS.md is read from files this script saves.

Sections (see RESULTS.md):
  T1 vocabulary / label audit            -> label_audit.json
  T2 constructor inputs + wall time      -> construction_audit.json
  T3 lexical-vs-dense hit@1 explanation  -> hit1_invariance_audit.json
  T4 E3 fixed two-branch CLIP gate CIs   -> e3_two_branch_ci.json
  T5 grid paired CIs                     -> grid_paired_ci.json
  T6 Algorithm 1 hierarchy x scorer grid -> alg1_grid_summary.json, alg1_grid_aggregate.csv, alg1_grid_per_query.json
  T7 construction-constant sensitivity   -> sensitivity.csv, sensitivity.json
  T8 structural agreement figure data    -> structural_agreement.csv, structural_agreement.json
"""
from __future__ import annotations

import argparse
import builtins
import csv
import io
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.geometry.spatial_graph import build_spatial_clusters
from backend.io.view_store import list_view_summaries
from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder
from backend.query.graph_vs_flat import filter_five_scene_queries, load_benchmark_queries, load_scene_bundle
from backend.query.model_client import _view_text
from backend.query.recall_controlled import (
    PreparedScene,
    RecallControlledConfig,
    paired_bootstrap_difference,
    search_flat_prepared_scene,
    search_prepared_scene,
)
from backend.query.affordance import tokenize
from scripts.evaluate_hierarchy_construction import _manual_zones, _positions, _zone_assignment_metrics
from scripts.run_ieee_bge_frontier import BGEIndex, _search as dense_alg1_search
from scripts.run_ieee_hierarchy_scorer_grid import _lexical_scores, _matched_rank, _rank
from scripts.run_matched_hierarchy_controls import _disjoint_projection, _random_partition, _reference_zones
from scripts.run_raw_rgbd_hierarchy_benchmark import (
    CLUSTER_LABELS,
    ClipEncoder,
    deterministic_kmeans,
    l2_normalize,
    load_raw_scene,
    rank_method,
    silhouette_score,
    standardize,
)

SCENE_MAP = {
    "ConferenceHall": "ConferenceHall-capture-pilot",
    "Museume": "Museume-capture",
    "Theater": "Theater-capture",
    "outdoor-street": "outdoor-street-capture",
    "outdoor-drone": "outdoor-drone-capture",
}
VARIANTS = ("manual_reference", "pose_only", "pose_semantic_merge", "raw_rgbd")
SCORERS = ("lexical", "bge", "clip")
BUDGETS = (3, 6, 12)
BOOT_SEED = 20260901
RANDOM_PARTITION_SEED = 20260724  # same seeding as scripts/run_matched_hierarchy_controls.py
ALG1_CONFIGS = {
    # Paper operating configs (outputs/ieee_revision/recall_frontier_v1 final_config; bge_frontier_v1 final_config
    # except branch scoring, which is forced to max_leaf_proxy for every scorer so the same branch rule applies to
    # hierarchies without branch text).
    "lexical": dict(branch_mass_threshold=0.7, score_temperature=0.25),
    "bge": dict(branch_mass_threshold=0.5, score_temperature=0.1),
    # CLIP: T = 0.01 = 1 / exp(logit_scale) of the pretrained CLIP ViT-B/32 (logit scale 100), i.e. the model's own
    # contrastive softmax temperature; tau and B, m follow the BGE config. Not tuned on test.
    "clip": dict(branch_mass_threshold=0.5, score_temperature=0.01),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- file-read audit
class ReadRecorder:
    """Records every file opened through builtins.open / io.open while active."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._orig_builtin = builtins.open
        self._orig_io = io.open

    def __enter__(self) -> "ReadRecorder":
        recorder = self
        original = self._orig_builtin

        def patched(file, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                recorder.paths.append(str(Path(file).resolve()))
            except TypeError:
                pass
            return original(file, *args, **kwargs)

        builtins.open = patched  # type: ignore[assignment]
        io.open = patched  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        builtins.open = self._orig_builtin  # type: ignore[assignment]
        io.open = self._orig_io  # type: ignore[assignment]

    def categories(self, scene_root: Path) -> dict[str, dict[str, int]]:
        return {"open_calls": self._categories(self.paths, scene_root),
                "unique_files": self._categories(sorted(set(self.paths)), scene_root)}

    @staticmethod
    def _categories(paths: list[str], scene_root: Path) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        root = str(scene_root.resolve())
        for path in paths:
            if not path.startswith(root):
                if "manual_zone_gt" in path:
                    counts["manual_zone_gt"] += 1
                elif "huggingface" in path or "site-packages" in path:
                    counts["library_or_model_files"] += 1
                else:
                    counts["other"] += 1
                continue
            tail = path[len(root) + 1:].split("/")
            sub = tail[1] if len(tail) > 1 else ""
            if sub == "transforms.json":
                counts["transforms.json"] += 1
            elif sub == "views":
                counts["views/*.json (authored ViewJSON)"] += 1
            elif sub == "images":
                counts["images/* (RGB)"] += 1
            elif sub == "depths":
                counts["depths/* (depth arrays)"] += 1
            elif sub == "tree":
                counts["tree/*"] += 1
            else:
                counts[f"scene/{sub}"] += 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- constructors
def pose_only_clusters(scene_dir: Path, alpha: float) -> list[list[str]]:
    return build_spatial_clusters(_positions(scene_dir), scale_factor=alpha)


def pose_semantic_clusters(scene_dir: Path, alpha: float) -> list[list[str]]:
    positions = _positions(scene_dir)
    summaries = list_view_summaries(str(scene_dir))
    return build_spatial_clusters(positions, scale_factor=alpha, view_summaries=summaries)


def as_zones(clusters: list[list[str]], prefix: str) -> list[dict[str, Any]]:
    return [
        {"zone_id": f"{prefix}_zone_{i:02d}", "cluster_id": f"{prefix}_zone_{i:02d}", "view_ids": sorted(c)}
        for i, c in enumerate(clusters, 1)
    ]


def raw_cluster(
    clip_embeddings: np.ndarray,
    pose: np.ndarray,
    depth: np.ndarray,
    *,
    w_p: float = 0.5,
    w_d: float = 0.2,
    kmax_divisor: int = 3,
    min_views: int | None = None,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    """Parametric copy of construct_raw_hierarchy's clustering (defaults reproduce it exactly).

    min_views=None reproduces the original code, which does NOT enforce a per-cluster minimum; with an integer the
    silhouette search only admits K whose k-means solution has every cluster >= min_views (K=1 if none qualifies).
    """
    parts = [clip_embeddings]
    if w_p:
        parts.append(w_p * standardize(pose))
    if w_d:
        parts.append(w_d * standardize(depth))
    combined = standardize(np.concatenate(parts, axis=1))
    n = len(combined)
    upper = max(2, n // kmax_divisor)
    candidates = []
    for k in range(2, upper + 1):
        labels, _c, _it = deterministic_kmeans(combined, k)
        sizes = np.bincount(labels, minlength=k)
        admissible = min_views is None or int(sizes.min()) >= min_views
        candidates.append({
            "k": k,
            "silhouette": round(silhouette_score(combined, labels), 8),
            "min_size": int(sizes.min()),
            "admissible": bool(admissible),
            "labels": labels,
        })
    admissible = [row for row in candidates if row["admissible"]]
    if not admissible:
        return np.zeros(n, dtype=np.int64), 1, [{k: v for k, v in r.items() if k != "labels"} for r in candidates]
    best = max(admissible, key=lambda row: (row["silhouette"], -row["k"]))
    labels, _c, _it = deterministic_kmeans(combined, best["k"])
    return labels, best["k"], [{k: v for k, v in r.items() if k != "labels"} for r in candidates]


def labels_to_zones(labels: np.ndarray, view_ids: list[str], prefix: str = "raw") -> list[dict[str, Any]]:
    zones = []
    for cluster in sorted(set(int(v) for v in labels)):
        members = [view_ids[i] for i in np.flatnonzero(labels == cluster)]
        zones.append({"zone_id": f"{prefix}_cluster_{cluster + 1:02d}", "cluster_id": f"{prefix}_cluster_{cluster + 1:02d}", "view_ids": members})
    return zones


# --------------------------------------------------------------------------- metrics helpers
def quality(rank: list[str], expected: set[str]) -> tuple[float, float, float]:
    h1 = float(bool(rank[:1] and rank[0] in expected))
    h3 = float(bool(set(rank[:3]) & expected))
    rr = next((1.0 / i for i, v in enumerate(rank, 1) if v in expected), 0.0)
    return h1, h3, rr


def macro_f1(zones_by_scene: dict[str, list[dict[str, Any]]], manual: dict[str, list[dict[str, Any]]], views: dict[str, set[str]]) -> tuple[float, dict[str, float]]:
    per = {s: float(_zone_assignment_metrics(z, manual[s], views[s])["pairwise_f1"]) for s, z in zones_by_scene.items()}
    return round(fmean(per.values()), 6), per


def boot(left: list[float], right: list[float]) -> dict[str, Any]:
    return paired_bootstrap_difference(left, right, samples=10_000, seed=BOOT_SEED)


# --------------------------------------------------------------------------- main
def run(args: argparse.Namespace) -> dict[str, Any]:
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_root = args.scene_root.resolve()
    torch.set_num_threads(args.threads)

    queries = filter_five_scene_queries(load_benchmark_queries(args.benchmark.resolve()))
    positives = [q for q in queries if q.get("expected_view_ids")]
    split = json.loads(args.split.read_text(encoding="utf-8"))
    dev_ids = set(split["dev"]) if isinstance(split["dev"][0], str) else {str(r["query_id"]) for r in split["dev"]}
    test_ids = set(split["test"]) if isinstance(split["test"][0], str) else {str(r["query_id"]) for r in split["test"]}
    manual_raw = _manual_zones(args.manual_zones.resolve())
    manual = {s: _reference_zones(manual_raw[s]) for s in SCENE_MAP.values()}
    bundles = {b: load_scene_bundle(scene_root / c) for b, c in SCENE_MAP.items()}
    views = {SCENE_MAP[b]: set(bundles[b]["views"]) for b in SCENE_MAP}
    summary: dict[str, Any] = {
        "schema_version": "beyond_proximity.ieee_revision2_hierarchy.v1",
        "generated_at": utc_now(),
        "query_count": len(queries),
        "positive_count": len(positives),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "bootstrap": {"samples": 10_000, "seed": BOOT_SEED, "implementation": "backend.query.recall_controlled.paired_bootstrap_difference"},
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__, "torch_threads": args.threads, "device": "cpu"},
    }

    # ------------------------------------------------------------------ T1 label audit
    saved_raw = json.loads(args.raw_hierarchy.read_text(encoding="utf-8"))
    outdoor_terms = {"facade", "garden", "outdoor plaza", "street", "walkway"}
    label_audit = {
        "vocabulary": list(CLUSTER_LABELS),
        "vocabulary_size": len(CLUSTER_LABELS),
        "prompt_template": saved_raw["semantic_labeling"]["prompt_template"],
        "explicitly_outdoor_categories": [v for v in CLUSTER_LABELS if v in outdoor_terms],
        "indoor_or_ambiguous_categories": [v for v in CLUSTER_LABELS if v not in outdoor_terms],
        "labels_by_scene": {
            s["scene_id"]: [
                {"cluster_id": c["cluster_id"], "members": c["member_count"], "predicted_label": c["predicted_label"],
                 "top3": [(p["label"], p["score"]) for p in c["predicted_labels"]]}
                for c in s["clusters"]
            ]
            for s in saved_raw["scenes"]
        },
        "label_use_trace": {
            "assigned_at": "scripts/run_raw_rgbd_hierarchy_benchmark.py:construct_raw_hierarchy (argmax of CLIP text('a photograph of a {label}') . cluster CLIP centroid, AFTER k-means)",
            "e3_retrieval": "rank_method -> cluster_centroids(clusters, image embeddings) uses only cluster['view_ids']; predicted_label is never read",
            "grid_retrieval": "run_ieee_hierarchy_scorer_grid._zones_by_scene copies predicted_label into zone_name; _matched_rank reads only zone['view_ids'] and zone_id (tie-break)",
            "alg1_this_run": "max_leaf_proxy branch scoring reads only branch membership",
            "conclusion": "Category names label branches only; they do not affect cluster membership or any retrieval result in the saved runs.",
        },
    }
    write_json(out / "label_audit.json", label_audit)
    summary["T1_labels"] = {"file": "label_audit.json"}

    # ------------------------------------------------------------------ encoders
    t0 = time.perf_counter()
    clip = ClipEncoder(args.clip_model, cache_dir=args.model_cache.resolve(), threads=args.threads)
    clip_load_s = time.perf_counter() - t0
    logit_scale = float(clip.model.logit_scale.exp().item())

    # ------------------------------------------------------------------ T2 construction audit + wall time
    construction: dict[str, Any] = {"alpha": 3.5, "repeats": {}, "scenes": {}}
    reads: dict[str, dict[str, int]] = {}
    for name, fn in (("pose_only", pose_only_clusters), ("pose_semantic_merge", pose_semantic_clusters)):
        with ReadRecorder() as rec:
            for c in SCENE_MAP.values():
                fn(scene_root / c, 3.5)
        reads[name] = rec.categories(scene_root)
    # raw: construction path from load_raw_scene -> CLIP -> k-means -> labels (as in construct_raw_hierarchy)
    raw_records: dict[str, list[dict[str, Any]]] = {}
    with ReadRecorder() as rec:
        label_emb, _calls, _tok = clip.encode_queries([f"a photograph of a {l}" for l in CLUSTER_LABELS], batch_size=len(CLUSTER_LABELS))
        for c in SCENE_MAP.values():
            raw_records[c] = load_raw_scene(scene_root / c)
            emb, _ = clip.encode_images(raw_records[c], batch_size=8)
    reads["raw_rgbd"] = rec.categories(scene_root)
    construction["file_reads_during_construction"] = reads

    timing_rows = []
    reps_pose = 20
    for name, fn in (("pose_only", pose_only_clusters), ("pose_semantic_merge", pose_semantic_clusters)):
        totals = []
        for _ in range(reps_pose):
            started = time.perf_counter()
            for c in SCENE_MAP.values():
                fn(scene_root / c, 3.5)
            totals.append(time.perf_counter() - started)
        timing_rows.append({"constructor": name, "repeats": reps_pose, "median_total_s_5_scenes": round(statistics.median(totals), 6),
                            "min_total_s": round(min(totals), 6), "max_total_s": round(max(totals), 6)})
    raw_totals = []
    raw_embedding_cache: dict[str, np.ndarray] = {}
    raw_default_labels: dict[str, np.ndarray] = {}
    for rep in range(args.raw_timing_repeats):
        started = time.perf_counter()
        label_emb, _calls, _tok = clip.encode_queries([f"a photograph of a {l}" for l in CLUSTER_LABELS], batch_size=len(CLUSTER_LABELS))
        for c in SCENE_MAP.values():
            records = load_raw_scene(scene_root / c)
            emb, _ = clip.encode_images(records, batch_size=8)
            pose = np.asarray([r["pose_xyz"] for r in records])
            depth = np.asarray([r["depth_descriptor"] for r in records], dtype=np.float64)
            labels, k, _cand = raw_cluster(emb, pose, depth)
            for cl in range(k):
                idx = np.flatnonzero(labels == cl)
                centroid = l2_normalize(emb[idx].mean(axis=0, keepdims=True))[0]
                _ = int(np.argmax(label_emb @ centroid))
            raw_embedding_cache[c] = emb
            raw_default_labels[c] = labels
        raw_totals.append(time.perf_counter() - started)
    timing_rows.append({"constructor": "raw_rgbd", "repeats": args.raw_timing_repeats, "median_total_s_5_scenes": round(statistics.median(raw_totals), 6),
                        "min_total_s": round(min(raw_totals), 6), "max_total_s": round(max(raw_totals), 6),
                        "excludes": f"one-time CLIP model load {clip_load_s:.3f} s", "device": "cpu", "torch_threads": args.threads})
    construction["timing"] = timing_rows
    construction["clip_model_load_s"] = round(clip_load_s, 6)
    construction["saved_raw_run_construction_latency_s"] = round(saved_raw["totals"]["construction_latency_ms"] / 1000, 6)

    # reproduce the saved raw hierarchy
    saved_by_scene = {s["scene_id"]: sorted(sorted(c["view_ids"]) for c in s["clusters"]) for s in saved_raw["scenes"]}
    view_ids_scene = {c: [r["view_id"] for r in raw_records[c]] for c in SCENE_MAP.values()}
    repro = {}
    for c in SCENE_MAP.values():
        mine = sorted(sorted(z["view_ids"]) for z in labels_to_zones(raw_default_labels[c], view_ids_scene[c]))
        repro[c] = mine == saved_by_scene[c]
    construction["raw_default_reproduces_saved_clusters"] = repro
    construction["cluster_structure"] = {}
    for c in SCENE_MAP.values():
        po = pose_only_clusters(scene_root / c, 3.5)
        ps = pose_semantic_clusters(scene_root / c, 3.5)
        construction["cluster_structure"][c] = {
            "views": len(view_ids_scene[c]),
            "pose_only_sizes": sorted((len(x) for x in po), reverse=True),
            "pose_semantic_sizes": sorted((len(x) for x in ps), reverse=True),
            "raw_rgbd_saved_sizes": sorted((len(x) for x in saved_by_scene[c]), reverse=True),
            "raw_rgbd_singletons": sum(1 for x in saved_by_scene[c] if len(x) == 1),
        }
    write_json(out / "construction_audit.json", construction)
    summary["T2_construction"] = {"timing": timing_rows, "file_reads": reads, "raw_reproduces_saved": repro}

    # ------------------------------------------------------------------ embeddings for scoring
    clip_q, _qc, _qt = clip.encode_queries([str(q["query"]) for q in queries], batch_size=32)
    clip_q_by_id = {str(q["query_id"]): clip_q[i] for i, q in enumerate(queries)}
    bge = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    view_texts = {b: {v: _view_text(view) for v, view in bundles[b]["views"].items()} for b in SCENE_MAP}
    bge_ids = {b: sorted(view_texts[b]) for b in SCENE_MAP}
    bge_view = {b: bge.encode([view_texts[b][v] for v in bge_ids[b]]) for b in SCENE_MAP}
    bge_q = bge.encode([str(q["query"]) for q in queries])
    bge_q_by_id = {str(q["query_id"]): bge_q[i] for i, q in enumerate(queries)}
    clip_ids = {b: view_ids_scene[SCENE_MAP[b]] for b in SCENE_MAP}
    assert all(clip_ids[b] == bge_ids[b] for b in SCENE_MAP), "view id order mismatch"

    def grid_scores(q: dict[str, Any], scorer: str) -> dict[str, float]:
        b = str(q["scene_id"])
        if scorer == "lexical":
            return _lexical_scores(view_texts[b], str(q["query"]))
        if scorer == "bge":
            vals = bge_view[b] @ bge_q_by_id[str(q["query_id"])]
            return {v: float(vals[i]) for i, v in enumerate(bge_ids[b])}
        vals = raw_embedding_cache[SCENE_MAP[b]] @ clip_q_by_id[str(q["query_id"])]
        return {v: float(vals[i]) for i, v in enumerate(clip_ids[b])}

    base_zones = {
        c: {
            "manual_reference": manual[c],
            "pose_only": as_zones(pose_only_clusters(scene_root / c, 3.5), "pose_only"),
            "pose_semantic_merge": as_zones(pose_semantic_clusters(scene_root / c, 3.5), "pose_semantic_merge"),
            "raw_rgbd": [
                {"zone_id": str(cl["cluster_id"]), "cluster_id": str(cl["cluster_id"]), "view_ids": list(cl["view_ids"])}
                for s in saved_raw["scenes"] if s["scene_id"] == c for cl in s["clusters"]
            ],
        }
        for c in SCENE_MAP.values()
    }

    # sanity: reproduce saved grid cells with my embeddings
    grid_saved = json.loads((args.grid_dir / "hierarchy_scorer_grid_summary.json").read_text(encoding="utf-8"))
    grid_repro = []
    for cell in grid_saved["aggregate"]:
        rows = []
        for q in positives:
            ranked, _cc = _matched_rank(grid_scores(q, cell["scorer"]), base_zones[SCENE_MAP[str(q["scene_id"])]][cell["variant"]], int(cell["budget"]))
            rows.append(quality(ranked, set(q["expected_view_ids"])))
        grid_repro.append({"variant": cell["variant"], "scorer": cell["scorer"], "budget": cell["budget"],
                           "saved_hit_at_1": cell["hit_at_1"], "recomputed_hit_at_1": round(fmean(r[0] for r in rows), 6),
                           "saved_hit_at_3": cell["hit_at_3"], "recomputed_hit_at_3": round(fmean(r[1] for r in rows), 6)})
    grid_repro_ok = all(r["saved_hit_at_1"] == r["recomputed_hit_at_1"] and r["saved_hit_at_3"] == r["recomputed_hit_at_3"] for r in grid_repro)
    write_json(out / "grid_reproduction_check.json", {"all_cells_match": grid_repro_ok, "cells": grid_repro})
    summary["grid_reproduction_all_cells_match"] = grid_repro_ok

    # ------------------------------------------------------------------ T3 hit@1 invariance audit
    grid_rows = json.loads((args.grid_dir / "per_query.json").read_text(encoding="utf-8"))
    by_cell = defaultdict(dict)
    for r in grid_rows:
        by_cell[(r["scorer"], int(r["budget"]))].setdefault(r["query_id"], {})[r["variant"]] = r
    t3: dict[str, Any] = {"per_scorer": {}}
    qmap = {str(q["query_id"]): q for q in positives}
    for scorer in SCORERS:
        top_is_global_argmax = 0
        cells = 0
        tie_at_top = 0
        differing_queries: set[str] = set()
        differing_with_tie: set[str] = set()
        for qid, q in qmap.items():
            scores = grid_scores(q, scorer)
            glob = _rank(scores)
            best = scores[glob[0]]
            tied = [v for v, s in scores.items() if s == best]
            if len(tied) > 1:
                tie_at_top += 1
            tops = set()
            for budget in BUDGETS:
                for variant in VARIANTS:
                    r = by_cell[(scorer, budget)][qid][variant]
                    cells += 1
                    top_is_global_argmax += int(r["ranked_view_ids"][0] == glob[0])
                    tops.add(bool(r["hit_at_1"]))
            if len(tops) > 1:
                differing_queries.add(qid)
                if len(tied) > 1:
                    differing_with_tie.add(qid)
        t3["per_scorer"][scorer] = {
            "positive_queries": len(qmap),
            "queries_with_tied_top_score": tie_at_top,
            "cells_checked": cells,
            "cells_where_top1_equals_flat_top1": top_is_global_argmax,
            "queries_whose_hit1_differs_across_hierarchies_or_budgets": len(differing_queries),
            "of_which_have_tied_top_score": len(differing_with_tie),
        }
    uncovered = {c: sorted(views[c] - {v for z in manual[c] for v in z["view_ids"]}) for c in SCENE_MAP.values()}
    t3["manual_views_not_in_any_zone"] = uncovered
    t3["mechanism"] = (
        "scripts/run_ieee_hierarchy_scorer_grid.py:_matched_rank orders branches by max leaf score (zone_score = max member score), "
        "adds whole branches until the union has >= k views, and then ranks the union with the SAME global leaf order. The first "
        "branch therefore contains the flat argmax view whenever the argmax is unique and covered by some branch, so the rank-1 "
        "view (and hit@1) equals flat retrieval for continuous BGE/CLIP scores. The lexical grid score is overlap + 1e-3*overlap/|q|, "
        "a monotone function of the integer token overlap, so several views tie at the maximum; branch ties are broken by zone_id "
        "and view ties by view_id (both descending), so the rank-1 view depends on which tied branch is expanded first. Hit@1 is "
        "not computed with a different branch score; the variation is purely tie-breaking."
    )
    write_json(out / "hit1_invariance_audit.json", t3)
    summary["T3_hit1"] = t3["per_scorer"]

    # ------------------------------------------------------------------ T4 E3 two-branch gate CIs
    e3_rows = json.loads((args.raw_run_dir / "per_query_results.json").read_text(encoding="utf-8"))
    e3_pos = [r for r in e3_rows if r["methods"]["flat_clip"]["quality_eligible"]]
    def col(method: str, metric: str) -> list[float]:
        return [float(r["methods"][method][metric]) for r in e3_pos]
    e3 = {"n": len(e3_pos), "source": rel(args.raw_run_dir / "per_query_results.json"), "means": {}, "paired": {}}
    for m in ("flat_clip", "manual_hierarchy_clip", "raw_rgbd_hierarchy_clip"):
        e3["means"][m] = {k: round(fmean(col(m, k)), 6) for k in ("hit_at_1", "hit_at_3")}
    for other in ("manual_hierarchy_clip", "flat_clip"):
        for metric in ("hit_at_1", "hit_at_3"):
            e3["paired"][f"raw_minus_{other}_{metric}"] = boot(col("raw_rgbd_hierarchy_clip", metric), col(other, metric))
    # target outside the two best-ranked branches
    outside = {}
    for m in ("manual_hierarchy_clip", "raw_rgbd_hierarchy_clip"):
        outside[m] = sum(1 for r in e3_pos if not (set(r["methods"][m]["expected_view_ids"]) & set(r["methods"][m]["candidate_view_ids"])))
    # recompute the same gate for pose-only / pose+semantic with the recomputed CLIP embeddings (and re-check manual/raw)
    recomputed_outside = {}
    recomputed_means = {}
    for label, variant in (("manual_hierarchy_clip", "manual_reference"), ("raw_rgbd_hierarchy_clip", "raw_rgbd"),
                           ("pose_only_clip", "pose_only"), ("pose_semantic_merge_clip", "pose_semantic_merge"), ("flat_clip", None)):
        cnt, h1s, h3s, sizes = 0, [], [], []
        for q in positives:
            c = SCENE_MAP[str(q["scene_id"])]
            zones = None if variant is None else [{"cluster_id": z["zone_id"], "view_ids": z["view_ids"]} for z in base_zones[c][variant]]
            ranked, cand, _sel = rank_method(clip_q_by_id[str(q["query_id"])], raw_embedding_cache[c], view_ids_scene[c], clusters=zones)
            exp = set(q["expected_view_ids"])
            cnt += int(not (exp & set(cand)))
            h1, h3, _ = quality(ranked, exp)
            h1s.append(h1); h3s.append(h3); sizes.append(len(cand))
        recomputed_outside[label] = cnt
        recomputed_means[label] = {"hit_at_1": round(fmean(h1s), 6), "hit_at_3": round(fmean(h3s), 6), "mean_views_checked": round(fmean(sizes), 6)}
    e3["target_outside_two_best_branches_saved_run"] = outside
    e3["target_outside_two_best_branches_recomputed"] = recomputed_outside
    e3["recomputed_gate_means"] = recomputed_means
    write_json(out / "e3_two_branch_ci.json", e3)
    summary["T4_e3"] = e3

    # ------------------------------------------------------------------ T5 grid paired CIs (saved grid per-query)
    t5 = {"source": rel(args.grid_dir / "per_query.json"), "seed": BOOT_SEED, "saved_grid_seed": grid_saved["protocol"]["seed"], "cells": []}
    for scorer, variant in (("bge", "raw_rgbd"), ("lexical", "pose_only")):
        for budget in BUDGETS:
            cell = by_cell[(scorer, budget)]
            ids = sorted(qmap)
            entry = {"scorer": scorer, "variant": variant, "budget": budget, "n": len(ids)}
            for metric in ("hit_at_1", "hit_at_3"):
                left = [float(cell[i][variant][metric]) for i in ids]
                right = [float(cell[i]["manual_reference"][metric]) for i in ids]
                entry[f"{variant}_{metric}"] = round(fmean(left), 6)
                entry[f"manual_{metric}"] = round(fmean(right), 6)
                entry[f"paired_{metric}_minus_manual"] = boot(left, right)
                saved = next(a for a in grid_saved["aggregate"] if a["variant"] == variant and a["scorer"] == scorer and a["budget"] == budget)
                entry[f"saved_grid_ci_{metric}_seed{grid_saved['protocol']['seed']}"] = saved[f"paired_{metric}_vs_manual"]
            t5["cells"].append(entry)
    write_json(out / "grid_paired_ci.json", t5)
    summary["T5_grid_ci"] = t5["cells"]

    # ------------------------------------------------------------------ T6 Algorithm 1 grid
    def prepared(b: str, zones: list[dict[str, Any]]) -> PreparedScene:
        ids = bge_ids[b]
        branch_ids = [str(z["zone_id"]) for z in zones]
        return PreparedScene(
            scene_id=b, view_ids=ids, view_texts=view_texts[b],
            view_tokens={v: tokenize(view_texts[b][v]) for v in ids},
            branch_ids=branch_ids,
            branch_texts={str(z["zone_id"]): str(z.get("zone_name") or z["zone_id"]) for z in zones},
            branch_tokens={str(z["zone_id"]): tokenize(str(z.get("zone_name") or "")) for z in zones},
            branch_views={str(z["zone_id"]): {str(v) for v in z["view_ids"] if str(v) in view_texts[b]} for z in zones},
            positions={},
        )

    def alg1(scorer: str, variant: str | None, budget: int) -> list[dict[str, Any]]:
        rows = []
        for q in positives:
            b = str(q["scene_id"])
            c = SCENE_MAP[b]
            zones = base_zones[c][variant] if variant else [{"zone_id": "all", "view_ids": sorted(views[c])}]
            scene = prepared(b, zones)
            if scorer == "lexical":
                if variant is None:
                    res = search_flat_prepared_scene(scene, q)
                else:
                    cfg = RecallControlledConfig(selection_mode="adaptive_mass", branch_score_mode="max_leaf_proxy", max_branches=5,
                                                 candidate_budget=budget, safety_shortlist_size=1, fallback_to_flat=False, **ALG1_CONFIGS[scorer])
                    res = search_prepared_scene(scene, q, cfg)
            else:
                matrix = bge_view[b] if scorer == "bge" else raw_embedding_cache[c]
                vec = bge_q_by_id[str(q["query_id"])] if scorer == "bge" else clip_q_by_id[str(q["query_id"])]
                index = BGEIndex(scene=scene, view_matrix=matrix, branch_matrix=np.zeros((len(zones), matrix.shape[1])))
                cfg = None if variant is None else RecallControlledConfig(
                    selection_mode="adaptive_mass", branch_score_mode="max_leaf_proxy", max_branches=5, candidate_budget=budget,
                    safety_shortlist_size=1, fallback_to_flat=False, **ALG1_CONFIGS[scorer])
                res = dense_alg1_search(index, q, vec, cfg)
            rows.append({
                "query_id": str(q["query_id"]), "scene_id": b, "scorer": scorer, "variant": variant or "flat", "budget": budget if variant else None,
                "split": "dev" if str(q["query_id"]) in dev_ids else ("test" if str(q["query_id"]) in test_ids else "other"),
                "hit_at_1": float(bool(res["hit_at_1"])), "hit_at_3": float(bool(res["hit_at_3"])), "mrr": float(res["mrr"]),
                "records_checked": int(res["records_checked"]), "selected_branch_count": res.get("selected_branch_count"),
                "branch_count": res.get("branch_count"), "target_in_candidates": bool(res["target_in_candidates"]),
                "ranked_view_ids": res["ranked_view_ids"][:3],
            })
        return rows

    alg1_rows: list[dict[str, Any]] = []
    agg: list[dict[str, Any]] = []
    for scorer in SCORERS:
        flat = alg1(scorer, None, 0)
        alg1_rows.extend(flat)
        agg.append({"scorer": scorer, "variant": "flat", "budget": None, "n": len(flat),
                    "hit_at_1": round(fmean(r["hit_at_1"] for r in flat), 6), "hit_at_3": round(fmean(r["hit_at_3"] for r in flat), 6),
                    "mean_records_checked": round(fmean(r["records_checked"] for r in flat), 6)})
        for budget in BUDGETS:
            cells = {v: alg1(scorer, v, budget) for v in VARIANTS}
            for v in VARIANTS:
                rows = cells[v]
                alg1_rows.extend(rows)
                entry = {"scorer": scorer, "variant": v, "budget": budget, "n": len(rows),
                         "hit_at_1": round(fmean(r["hit_at_1"] for r in rows), 6), "hit_at_3": round(fmean(r["hit_at_3"] for r in rows), 6),
                         "dev_hit_at_3": round(fmean(r["hit_at_3"] for r in rows if r["split"] == "dev"), 6),
                         "test_hit_at_3": round(fmean(r["hit_at_3"] for r in rows if r["split"] == "test"), 6),
                         "mean_records_checked": round(fmean(r["records_checked"] for r in rows), 6),
                         "mean_selected_branch_ratio": round(fmean(r["selected_branch_count"] / r["branch_count"] for r in rows), 6),
                         "target_in_candidates_rate": round(fmean(float(r["target_in_candidates"]) for r in rows), 6),
                         "config": {**ALG1_CONFIGS[scorer], "branch_score_mode": "max_leaf_proxy", "max_branches": 5, "safety_shortlist_size": 1, "candidate_budget": budget}}
                if v != "manual_reference":
                    for metric in ("hit_at_1", "hit_at_3"):
                        ci = boot([r[metric] for r in rows], [r[metric] for r in cells["manual_reference"]])
                        entry[f"paired_{metric}_minus_manual"] = ci["estimate"]
                        entry[f"paired_{metric}_ci_low"] = ci["ci_95"]["low"]
                        entry[f"paired_{metric}_ci_high"] = ci["ci_95"]["high"]
                agg.append(entry)
    write_json(out / "alg1_grid_per_query.json", alg1_rows)
    write_csv(out / "alg1_grid_aggregate.csv", agg)
    write_json(out / "alg1_grid_summary.json", {"configs": ALG1_CONFIGS, "clip_logit_scale": logit_scale, "aggregate": agg,
                                                "note": "Algorithm 1 = backend/query/recall_controlled.search_prepared_scene (lexical) and the dense "
                                                        "variant scripts/run_ieee_bge_frontier._search (BGE, CLIP); branches = hierarchy zones; "
                                                        "branch score = max leaf; fallback_to_flat False; 125 positives."})
    summary["T6_alg1"] = {"configs": ALG1_CONFIGS, "clip_logit_scale": round(logit_scale, 4), "aggregate": agg}

    # ------------------------------------------------------------------ T7 sensitivity
    def gate_quality(zones_by_scene: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
        per = []
        for q in positives:
            c = SCENE_MAP[str(q["scene_id"])]
            ranked, _ = _matched_rank(grid_scores(q, "clip"), zones_by_scene[c], 3)
            h1, h3, _ = quality(ranked, set(q["expected_view_ids"]))
            per.append((str(q["query_id"]), h1, h3))
        dev = [p for p in per if p[0] in dev_ids]
        return {"dev_hit_at_1": round(fmean(p[1] for p in dev), 6), "dev_hit_at_3": round(fmean(p[2] for p in dev), 6),
                "all_hit_at_1": round(fmean(p[1] for p in per), 6), "all_hit_at_3": round(fmean(p[2] for p in per), 6),
                "dev_n": len(dev), "all_n": len(per)}

    raw_inputs = {c: (raw_embedding_cache[c], np.asarray([r["pose_xyz"] for r in raw_records[c]]),
                      np.asarray([r["depth_descriptor"] for r in raw_records[c]], dtype=np.float64)) for c in SCENE_MAP.values()}
    sens_rows = []

    def add_raw(group: str, value: Any, **kw: Any) -> None:
        zones_by_scene, ks, singletons = {}, {}, 0
        for c in SCENE_MAP.values():
            emb, pose, depth = raw_inputs[c]
            labels, k, _cand = raw_cluster(emb, pose, depth, **kw)
            zones_by_scene[c] = labels_to_zones(labels, view_ids_scene[c])
            ks[c] = k
            singletons += sum(1 for z in zones_by_scene[c] if len(z["view_ids"]) == 1)
        f1, per = macro_f1(zones_by_scene, manual, views)
        sens_rows.append({"hierarchy": "raw_rgbd", "parameter": group, "value": value, "is_default": value == {"w_p": "0.5", "w_d": "0.2", "min_views": "none (original)", "K_max": "floor(n/3)"}[group],
                          "macro_pairwise_f1": f1, "cluster_counts": ks, "singleton_clusters": singletons, **gate_quality(zones_by_scene)})

    for wp in (0.0, 0.25, 0.5, 1.0):
        add_raw("w_p", str(wp), w_p=wp)
    for wd in (0.0, 0.1, 0.2, 0.5):
        add_raw("w_d", str(wd), w_d=wd)
    add_raw("min_views", "none (original)", min_views=None)
    for m in (2, 3, 4):
        add_raw("min_views", str(m), min_views=m)
    for div, lab in ((4, "floor(n/4)"), (3, "floor(n/3)"), (2, "floor(n/2)")):
        add_raw("K_max", lab, kmax_divisor=div)
    for alpha in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        zones_by_scene = {c: as_zones(pose_only_clusters(scene_root / c, alpha), "pose_only") for c in SCENE_MAP.values()}
        f1, per = macro_f1(zones_by_scene, manual, views)
        sens_rows.append({"hierarchy": "pose_only", "parameter": "alpha", "value": str(alpha), "is_default": alpha == 3.5,
                          "macro_pairwise_f1": f1, "cluster_counts": {c: len(z) for c, z in zones_by_scene.items()},
                          "singleton_clusters": sum(1 for z in zones_by_scene.values() for x in z if len(x["view_ids"]) == 1),
                          **gate_quality(zones_by_scene)})
    ref = {
        "flat_clip": gate_quality({c: [{"zone_id": "all", "view_ids": sorted(views[c])}] for c in SCENE_MAP.values()}),
        "manual_reference": gate_quality(manual),
    }
    write_csv(out / "sensitivity.csv", sens_rows)
    write_json(out / "sensitivity.json", {"gate": "budget-filling (run_ieee_hierarchy_scorer_grid._matched_rank), k=3, CLIP scorer", "rows": sens_rows, "references": ref,
                                           "min_views_note": "The original constructor does not enforce a per-cluster minimum; 'none (original)' is the shipped behaviour."})
    summary["T7_sensitivity"] = {"rows": sens_rows, "references": ref}

    # ------------------------------------------------------------------ T8 structural agreement
    struct_zones = {v: {c: base_zones[c][v] for c in SCENE_MAP.values()} for v in ("pose_only", "pose_semantic_merge", "raw_rgbd")}
    struct_rows = []
    trials = 1000
    for variant, zones_by_scene in struct_zones.items():
        f1, per = macro_f1(zones_by_scene, manual, views)
        per_trial = np.zeros((trials, len(SCENE_MAP)))
        per_scene_rand = {}
        for j, c in enumerate(SCENE_MAP.values()):
            sizes = [len(z["view_ids"]) for z in zones_by_scene[c]]
            rng = random.Random(RANDOM_PARTITION_SEED + sum(ord(ch) for ch in c + variant))
            vals = [float(_zone_assignment_metrics(_random_partition(views[c], sizes, rng), manual[c], views[c])["pairwise_f1"]) for _ in range(trials)]
            per_trial[:, j] = vals
            per_scene_rand[c] = round(fmean(vals), 6)
        macro_trials = per_trial.mean(axis=1)
        struct_rows.append({"hierarchy": variant, "macro_pairwise_f1": f1,
                            "random_size_matched_mean": round(float(macro_trials.mean()), 6),
                            "random_size_matched_p2_5": round(float(np.percentile(macro_trials, 2.5)), 6),
                            "random_size_matched_p97_5": round(float(np.percentile(macro_trials, 97.5)), 6),
                            "fraction_random_ge_observed": round(float((macro_trials >= f1 - 1e-12).mean()), 6),
                            "random_trials": trials, **{f"f1_{c}": per[c] for c in SCENE_MAP.values()},
                            **{f"random_mean_{c}": per_scene_rand[c] for c in SCENE_MAP.values()}})
    proj = {c: _disjoint_projection(manual_raw_as_ref(manual_raw[c]), views[c]) for c in SCENE_MAP.values()}
    pf1, pper = macro_f1(proj, manual, views)
    struct_rows.append({"hierarchy": "disjoint_manual_projection", "macro_pairwise_f1": pf1, **{f"f1_{c}": pper[c] for c in SCENE_MAP.values()}})
    write_csv(out / "structural_agreement.csv", struct_rows)
    reviewer = rel(ROOT / "outputs/reviewer_revision/matched_hierarchy_v1/structure.csv")
    write_json(out / "structural_agreement.json", {"rows": struct_rows, "metric": "pairwise F1 (scripts/evaluate_hierarchy_construction._zone_assignment_metrics), macro over 5 scenes",
                                                    "random": "cluster-size-matched random partition, 1000 trials/scene, seed 20260724+ord-sum(scene+variant) as in run_matched_hierarchy_controls; macro per trial; 95% range = 2.5-97.5 percentiles",
                                                    "cross_check_existing": reviewer, "excludes": "Cursor/MCP hierarchy"})
    summary["T8_structure"] = struct_rows

    write_json(out / "summary.json", summary)
    return summary


def manual_raw_as_ref(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # _disjoint_projection needs query_reference_count, which _reference_zones drops.
    return [{"zone_id": str(r["zone_id"]), "view_ids": [str(v) for v in r["view_ids"]], "query_reference_count": r.get("query_reference_count")} for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    p.add_argument("--manual-zones", type=Path, default=ROOT / "docs/benchmarks/manual_zone_gt_v2.json")
    p.add_argument("--split", type=Path, default=ROOT / "outputs/ieee_revision/recall_frontier_v1/split.json")
    p.add_argument("--raw-hierarchy", type=Path, default=ROOT / "outputs/raw_rgbd_hierarchy/five_scene_clip_v1/raw_hierarchy.json")
    p.add_argument("--raw-run-dir", type=Path, default=ROOT / "outputs/raw_rgbd_hierarchy/five_scene_clip_v1")
    p.add_argument("--grid-dir", type=Path, default=ROOT / "outputs/ieee_revision/hierarchy_scorer_grid_v1")
    p.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/hierarchy")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--model-cache", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--raw-timing-repeats", type=int, default=3)
    args = p.parse_args()
    s = run(args)
    print(json.dumps({k: s[k] for k in ("grid_reproduction_all_cells_match",)}, indent=2))


if __name__ == "__main__":
    main()
