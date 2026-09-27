#!/usr/bin/env python3
"""Run the matched hierarchy x scorer x budget grid requested for IEEE Access."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.geometry.spatial_graph import build_spatial_clusters
from backend.io.view_store import list_view_summaries
from backend.query.affordance import score_text_with_tokens, tokenize
from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder
from backend.query.graph_vs_flat import filter_five_scene_queries, load_benchmark_queries, load_scene_bundle
from backend.query.model_client import _view_text
from backend.query.recall_controlled import paired_bootstrap_difference
from scripts.evaluate_hierarchy_construction import _cluster_records, _manual_zones, _positions
from scripts.run_raw_rgbd_hierarchy_benchmark import ClipEncoder, load_raw_scene

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
SEED = 20260902


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in rows
        ])


def _zones_by_scene(
    scene_root: Path,
    manual_path: Path,
    raw_path: Path,
    scale_factor: float,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    references = _manual_zones(manual_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_by_scene = {str(row["scene_id"]): list(row["clusters"]) for row in raw["scenes"]}
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for captured_scene in SCENE_MAP.values():
        summaries = list_view_summaries(str(scene_root / captured_scene))
        positions = _positions(scene_root / captured_scene)
        result[captured_scene] = {
            "manual_reference": [
                {
                    "zone_id": str(row["zone_id"]),
                    "zone_name": str(row["zone_label"]),
                    "view_ids": [str(value) for value in row["view_ids"]],
                }
                for row in references[captured_scene]
            ],
            "pose_only": _cluster_records(
                build_spatial_clusters(positions, scale_factor=scale_factor),
                summaries,
                variant="pose_only",
            ),
            "pose_semantic_merge": _cluster_records(
                build_spatial_clusters(
                    positions,
                    scale_factor=scale_factor,
                    view_summaries=summaries,
                ),
                summaries,
                variant="pose_semantic_merge",
            ),
            "raw_rgbd": [
                {
                    "zone_id": str(row["cluster_id"]),
                    "zone_name": str(row["predicted_label"]),
                    "view_ids": [str(value) for value in row["view_ids"]],
                }
                for row in raw_by_scene[captured_scene]
            ],
        }
    return result


def _lexical_scores(texts: dict[str, str], query: str) -> dict[str, float]:
    terms = tokenize(query)
    result = {}
    for view_id, text in texts.items():
        overlap, ratio = score_text_with_tokens(text, terms)
        result[view_id] = float(overlap) + float(ratio) * 1e-3
    return result


def _rank(scores: dict[str, float]) -> list[str]:
    return [view_id for view_id, _score in sorted(scores.items(), key=lambda row: (row[1], row[0]), reverse=True)]


def _hierarchy_priority(scores: dict[str, float], zones: list[dict[str, Any]]) -> list[str]:
    global_rank = _rank(scores)
    zone_rows = []
    for zone in zones:
        members = [str(value) for value in zone.get("view_ids", []) if str(value) in scores]
        members = sorted(members, key=lambda value: (scores[value], value), reverse=True)
        zone_score = max((scores[value] for value in members), default=float("-inf"))
        zone_rows.append((zone_score, str(zone.get("zone_id") or ""), members))
    priority: list[str] = []
    for _score, _zone_id, members in sorted(zone_rows, reverse=True):
        priority.extend(members)
    priority.extend(global_rank)
    return list(dict.fromkeys(priority))


def _matched_rank(scores: dict[str, float], zones: list[dict[str, Any]], budget: int) -> tuple[list[str], int]:
    global_rank = _rank(scores)
    zone_rows = []
    for zone in zones:
        members = {str(value) for value in zone.get("view_ids", []) if str(value) in scores}
        zone_score = max((scores[value] for value in members), default=float("-inf"))
        zone_rows.append((zone_score, str(zone.get("zone_id") or ""), members))
    candidates: set[str] = set()
    for _score, _zone_id, members in sorted(zone_rows, reverse=True):
        candidates.update(members)
        if len(candidates) >= budget:
            break
    ranked = [value for value in global_rank if value in candidates]
    if len(ranked) < budget:
        ranked.extend(value for value in global_rank if value not in candidates)
    return ranked[: min(budget, len(global_rank))], len(candidates)


def _quality(rank: list[str], expected: set[str]) -> tuple[bool | None, bool | None, float | None]:
    if not expected:
        return None, None, None
    hit_one = bool(rank[:1] and rank[0] in expected)
    hit_three = bool(set(rank[:3]) & expected)
    reciprocal = next((1.0 / index for index, value in enumerate(rank, 1) if value in expected), 0.0)
    return hit_one, hit_three, reciprocal


def _summary(rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    eligible = [row for row in rows if row["hit_at_1"] is not None]
    result = {
        "variant": rows[0]["variant"],
        "scorer": rows[0]["scorer"],
        "budget": rows[0]["budget"],
        "query_count": len(rows),
        "quality_denominator": len(eligible),
        "hit_at_1": round(fmean(float(row["hit_at_1"]) for row in eligible), 6),
        "hit_at_3": round(fmean(float(row["hit_at_3"]) for row in eligible), 6),
        "mrr": round(fmean(float(row["mrr"]) for row in eligible), 6),
        "mean_retained": round(fmean(float(row["retained_count"]) for row in rows), 6),
    }
    if manual_rows is not None:
        manual_by_id = {str(row["query_id"]): row for row in manual_rows}
        for metric in ("hit_at_1", "hit_at_3", "mrr"):
            result[f"paired_{metric}_vs_manual"] = paired_bootstrap_difference(
                [float(row[metric]) for row in eligible],
                [float(manual_by_id[str(row["query_id"])][metric]) for row in eligible],
                seed=SEED,
            )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene_root = args.scene_root.resolve()
    queries = filter_five_scene_queries(load_benchmark_queries(args.benchmark.resolve()))
    zones = _zones_by_scene(
        scene_root,
        args.manual_zones.resolve(),
        args.raw_hierarchy.resolve(),
        args.spatial_scale_factor,
    )
    bundles: dict[str, dict[str, Any]] = {}
    view_texts: dict[str, dict[str, str]] = {}
    for benchmark_scene, captured_scene in SCENE_MAP.items():
        bundle = load_scene_bundle(scene_root / captured_scene)
        bundles[benchmark_scene] = bundle
        view_texts[benchmark_scene] = {
            view_id: _view_text(view) for view_id, view in bundle["views"].items()
        }

    bge = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    bge_view: dict[str, np.ndarray] = {}
    bge_ids: dict[str, list[str]] = {}
    for scene_id, texts in view_texts.items():
        ids = sorted(texts)
        bge_ids[scene_id] = ids
        bge_view[scene_id] = bge.encode([texts[value] for value in ids])
    bge_queries = bge.encode([str(row["query"]) for row in queries])

    clip = ClipEncoder(args.clip_model, cache_dir=args.model_cache.resolve(), threads=args.threads)
    clip_view: dict[str, np.ndarray] = {}
    clip_ids: dict[str, list[str]] = {}
    clip_calls = []
    for benchmark_scene, captured_scene in SCENE_MAP.items():
        records = load_raw_scene(scene_root / captured_scene)
        matrix, calls = clip.encode_images(records, batch_size=args.image_batch_size)
        clip_ids[benchmark_scene] = [str(row["view_id"]) for row in records]
        clip_view[benchmark_scene] = matrix
        clip_calls.extend({"scene_id": benchmark_scene, **row} for row in calls)
    clip_queries, clip_query_calls, _token_counts = clip.encode_queries(
        [str(row["query"]) for row in queries], batch_size=args.text_batch_size
    )

    query_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        scene_id = str(query["scene_id"])
        captured_scene = SCENE_MAP[scene_id]
        expected = {str(value) for value in query.get("expected_view_ids", []) if value}
        scores_by_scorer = {
            "lexical": _lexical_scores(view_texts[scene_id], str(query["query"])),
            "bge": {
                value: float(bge_view[scene_id][index] @ bge_queries[query_index])
                for index, value in enumerate(bge_ids[scene_id])
            },
            "clip": {
                value: float(clip_view[scene_id][index] @ clip_queries[query_index])
                for index, value in enumerate(clip_ids[scene_id])
            },
        }
        for scorer in SCORERS:
            scores = scores_by_scorer[scorer]
            for variant in VARIANTS:
                for budget in BUDGETS:
                    retained, candidate_count = _matched_rank(
                        scores, zones[captured_scene][variant], budget
                    )
                    hit_one, hit_three, reciprocal = _quality(retained, expected)
                    query_rows.append({
                        "query_id": str(query["query_id"]),
                        "scene_id": scene_id,
                        "captured_scene_id": captured_scene,
                        "query_type": str(query.get("query_type") or "unspecified"),
                        "variant": variant,
                        "scorer": scorer,
                        "budget": budget,
                        "candidate_count_before_budget": candidate_count,
                        "retained_count": len(retained),
                        "ranked_view_ids": retained,
                        "expected_view_ids": sorted(expected),
                        "hit_at_1": hit_one,
                        "hit_at_3": hit_three,
                        "mrr": reciprocal,
                    })

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[(row["variant"], row["scorer"], int(row["budget"]))].append(row)
    aggregate = []
    for scorer in SCORERS:
        for budget in BUDGETS:
            manual = grouped[("manual_reference", scorer, budget)]
            for variant in VARIANTS:
                aggregate.append(_summary(grouped[(variant, scorer, budget)], None if variant == "manual_reference" else manual))

    per_scene = []
    for scene_id in SCENE_MAP:
        for row in aggregate:
            selected = [
                value for value in grouped[(row["variant"], row["scorer"], row["budget"])]
                if value["scene_id"] == scene_id and value["hit_at_1"] is not None
            ]
            per_scene.append({
                "scene_id": scene_id,
                "variant": row["variant"],
                "scorer": row["scorer"],
                "budget": row["budget"],
                "quality_denominator": len(selected),
                "hit_at_1": round(fmean(float(value["hit_at_1"]) for value in selected), 6),
                "hit_at_3": round(fmean(float(value["hit_at_3"]) for value in selected), 6),
                "mrr": round(fmean(float(value["mrr"]) for value in selected), 6),
            })

    result = {
        "schema_version": "beyond_proximity.ieee_hierarchy_scorer_grid.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "query_count": len(queries),
        "quality_denominator": sum(bool(row.get("expected_view_ids")) for row in queries),
        "variants": list(VARIANTS),
        "scorers": list(SCORERS),
        "budgets": list(BUDGETS),
        "aggregate": aggregate,
        "per_scene": per_scene,
        "models": {
            "bge": {"model_id": DEFAULT_LOCAL_MODEL, "backend": bge.backend_name},
            "clip": {
                "model_id": args.clip_model,
                "revision": clip.revision,
                "device": "cpu",
                "image_encoder_calls": len(clip_calls),
                "query_encoder_calls": len(clip_query_calls),
            },
        },
        "protocol": {
            "hierarchy_priority": "For each budget independently, branches are ordered by maximum leaf score and expanded until their union covers the budget. The shared leaf scorer ranks that candidate union; the global order fills only uncovered capacity.",
            "matched_retention": "Every cell retains exactly min(budget, scene view count) records.",
            "raw_construction_boundary": "Raw RGB-D clusters were frozen before manual zones were read. Lexical and BGE cells use ViewJSON text only as a shared query-time leaf scorer, not as a raw-construction input.",
            "uncertainty": "Paired 10,000-resample bootstrap differences against the manual hierarchy for every non-manual cell.",
            "seed": SEED,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "hierarchy_scorer_grid_summary.json", result)
    _write_json(args.out / "per_query.json", query_rows)
    _write_csv(args.out / "aggregate.csv", aggregate)
    _write_csv(args.out / "per_scene.csv", per_scene)
    _write_json(args.out / "run_config.json", {
        "benchmark": str(args.benchmark.resolve()),
        "manual_zones": str(args.manual_zones.resolve()),
        "raw_hierarchy": str(args.raw_hierarchy.resolve()),
        "scene_root": str(scene_root),
        "spatial_scale_factor": args.spatial_scale_factor,
        "clip_model": args.clip_model,
        "budgets": list(BUDGETS),
        "seed": SEED,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--manual-zones", type=Path, default=ROOT / "docs/benchmarks/manual_zone_gt_v2.json")
    parser.add_argument("--raw-hierarchy", type=Path, default=ROOT / "outputs/raw_rgbd_hierarchy/five_scene_clip_v1/raw_hierarchy.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/hierarchy_scorer_grid_v1")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--spatial-scale-factor", type=float, default=3.5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
