#!/usr/bin/env python3
"""Run a same-input flat/graph BGE control on saved ConceptGraphs maps."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder
from backend.query.recall_controlled import paired_bootstrap_difference, wilson_interval
from scripts.evaluate_results import best_bbox_iou_3d


@dataclass
class PredictedObject:
    object_id: str
    text: str
    bbox: dict[str, Any]
    center: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_map(path: Path) -> list[PredictedObject]:
    original = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
    finally:
        pathlib.PosixPath = original
    result: list[PredictedObject] = []
    for index, raw in enumerate(payload.get("objects", [])):
        points = np.asarray(raw.get("bbox_np"), dtype=np.float64)
        if points.shape != (8, 3) or not np.isfinite(points).all():
            continue
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        if np.any(maximum <= minimum):
            continue
        captions = raw.get("captions") or []
        if isinstance(captions, str):
            captions = [captions]
        parts = [str(raw.get("class_name", "object"))]
        parts.extend(str(value) for value in captions[:3] if str(value).strip())
        result.append(PredictedObject(
            object_id=str(raw.get("id", f"object_{index}")),
            text=". ".join(dict.fromkeys(parts)),
            bbox={"min": minimum.tolist(), "max": maximum.tolist()},
            center=(minimum + maximum) / 2.0,
        ))
    return result


def _split_spatial(indices: list[int], centers: np.ndarray, max_leaf_size: int = 12) -> list[list[int]]:
    pending = [indices]
    leaves: list[list[int]] = []
    while pending:
        group = pending.pop(0)
        if len(group) <= max_leaf_size:
            leaves.append(sorted(group))
            continue
        values = centers[group]
        axis = int(np.argmax(np.ptp(values, axis=0)))
        ordered = sorted(group, key=lambda index: (float(centers[index, axis]), index))
        midpoint = len(ordered) // 2
        pending.extend((ordered[:midpoint], ordered[midpoint:]))
    return leaves


def _softmax(scores: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    shifted = (scores - float(scores.max())) / temperature
    weights = np.exp(np.clip(shifted, -80.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


def _select_branches(scores: np.ndarray, threshold: float = 0.7, cap: int = 5) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    probabilities = _softmax(scores)
    selected: list[int] = []
    mass = 0.0
    for index in order:
        selected.append(int(index))
        mass += float(probabilities[index])
        if mass >= threshold or len(selected) >= cap:
            break
    return selected


def _result_row(
    *, dataset: str, scene_id: str, query: dict[str, Any], objects: list[PredictedObject],
    similarities: np.ndarray, branches: list[list[int]], method: str,
    encoder: FastEmbedSemanticEncoder,
) -> dict[str, Any]:
    started = time.perf_counter()
    flat_order = np.argsort(-similarities, kind="stable")
    selected_branches: list[int] = []
    if method == "flat_bge_predicted_map":
        ranked = flat_order.tolist()
    else:
        branch_scores = np.asarray([
            max(float(similarities[index]) for index in branch) for branch in branches
        ])
        selected_branches = _select_branches(branch_scores)
        candidates = {index for branch_id in selected_branches for index in branches[branch_id]}
        candidates.add(int(flat_order[0]))  # M3 global safety shortlist, m=1.
        ranked = [int(index) for index in flat_order if int(index) in candidates][:3]
    top3 = ranked[:3]
    ious = [best_bbox_iou_3d(objects[index].bbox, query) or 0.0 for index in top3]
    top_iou = ious[0] if ious else 0.0
    max_iou = max(ious, default=0.0)
    context_tokens = sum(encoder.token_count(objects[index].text) for index in top3)
    return {
        "dataset": dataset,
        "scene_id": scene_id,
        "query_id": str(query["query_id"]),
        "query": str(query["query"]),
        "method": method,
        "ranked_object_ids": [objects[index].object_id for index in top3],
        "selected_branch_ids": selected_branches,
        "branch_count": len(branches),
        "proxy_object_comparisons": len(objects),
        "checked_objects": len(objects) if method == "flat_bge_predicted_map" else len(ranked),
        "serialized_top3_tokens": context_tokens,
        "top1_iou": round(top_iou, 6),
        "max_top3_iou": round(max_iou, 6),
        "acc_at_0_25": float(top_iou >= 0.25),
        "acc_at_0_50": float(top_iou >= 0.50),
        "recall_at_3_iou_0_25": float(max_iou >= 0.25),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
    }


def _summarize(rows: list[dict[str, Any]], flat_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    n = len(rows)
    result = {
        "method": rows[0]["method"],
        "query_count": n,
        "acc_at_0_25": round(sum(row["acc_at_0_25"] for row in rows) / n, 6),
        "acc_at_0_50": round(sum(row["acc_at_0_50"] for row in rows) / n, 6),
        "recall_at_3_iou_0_25": round(sum(row["recall_at_3_iou_0_25"] for row in rows) / n, 6),
        "mean_checked_objects": round(sum(row["checked_objects"] for row in rows) / n, 6),
        "mean_proxy_object_comparisons": round(sum(row["proxy_object_comparisons"] for row in rows) / n, 6),
        "mean_serialized_top3_tokens": round(sum(row["serialized_top3_tokens"] for row in rows) / n, 6),
        "mean_elapsed_ms": round(sum(row["elapsed_ms"] for row in rows) / n, 6),
    }
    successes = int(sum(row["acc_at_0_25"] for row in rows))
    low, high = wilson_interval(successes, n)
    result["acc_at_0_25_wilson_95"] = {"low": round(low, 6), "high": round(high, 6)}
    if flat_rows is not None:
        by_id = {row["query_id"]: row for row in flat_rows}
        paired = [(row, by_id[row["query_id"]]) for row in rows]
        for metric in ("acc_at_0_25", "acc_at_0_50", "recall_at_3_iou_0_25"):
            result[f"paired_{metric}_difference_vs_flat"] = paired_bootstrap_difference(
                [left[metric] for left, _ in paired],
                [right[metric] for _, right in paired],
                samples=10000,
                seed=20260902,
            )
        flat_checked = sum(right["checked_objects"] for _, right in paired) / n
        result["checked_object_reduction_vs_flat"] = round(
            1.0 - result["mean_checked_objects"] / flat_checked, 6
        )
    return result


def _map_path(root: Path, scene_id: str) -> Path:
    paths = list((root / "native_data" / scene_id).glob("exps/*mapping*/pcd_*.pkl.gz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one predicted map for {scene_id}, found {len(paths)}")
    return paths[0]


def run(scannet_root: Path, replica_root: Path, out: Path) -> dict[str, Any]:
    datasets = {
        "ScanNet": (scannet_root, ROOT / "docs/benchmarks/scannet_bbq_grounding_pilot_v1.json"),
        "Replica": (replica_root, ROOT / "docs/benchmarks/replica_scannet_pilot_queries.json"),
    }
    encoder = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    construction: list[dict[str, Any]] = []
    for dataset, (map_root, benchmark_path) in datasets.items():
        benchmark = _load_json(benchmark_path)
        scene_ids = _load_json(map_root / "run_config.json")["scene_ids"]
        queries = [row for row in benchmark["queries"] if row["scene_id"] in scene_ids]
        rows_by_method: dict[str, list[dict[str, Any]]] = {
            "flat_bge_predicted_map": [], "graph_bge_predicted_map": []
        }
        for scene_id in scene_ids:
            started = time.perf_counter()
            objects = _load_map(_map_path(map_root, scene_id))
            if not objects:
                continue
            centers = np.vstack([obj.center for obj in objects])
            branches = _split_spatial(list(range(len(objects))), centers)
            object_vectors = encoder.encode([obj.text for obj in objects])
            scene_queries = [row for row in queries if row["scene_id"] == scene_id]
            query_vectors = encoder.encode([str(row["query"]) for row in scene_queries])
            construction.append({
                "dataset": dataset, "scene_id": scene_id, "object_count": len(objects),
                "branch_count": len(branches), "max_leaf_size": max(map(len, branches)),
                "construction_ms": round((time.perf_counter() - started) * 1000.0, 6),
            })
            for query, vector in zip(scene_queries, query_vectors):
                similarities = object_vectors @ vector
                for method in rows_by_method:
                    rows_by_method[method].append(_result_row(
                        dataset=dataset, scene_id=scene_id, query=query, objects=objects,
                        similarities=similarities, branches=branches, method=method, encoder=encoder,
                    ))
        flat = rows_by_method["flat_bge_predicted_map"]
        graph = rows_by_method["graph_bge_predicted_map"]
        summaries.extend((_summarize(flat), _summarize(graph, flat)))
        all_rows.extend(flat + graph)
    payload = {
        "schema_version": "beyond_proximity.predicted_map_pair.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model_name": DEFAULT_LOCAL_MODEL,
        "protocol": {
            "map_source": "native ConceptGraphs predicted object maps",
            "hierarchy": "deterministic recursive spatial median split, maximum 12 objects per branch",
            "branch_gate": "max cached BGE leaf proxy, adaptive mass 0.7, cap 5",
            "safety_shortlist": 1,
            "retained_object_budget": 3,
            "quality": "official dataset 3D boxes; top-1 Acc@0.25/0.50 and top-3 Recall@IoU0.25",
            "accounting": "proxy comparisons, retained objects, and serialized top-3 tokens are separate",
        },
        "construction": construction,
        "summaries": summaries,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "predicted_map_pair_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "predicted_map_pair_per_query.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scannet-root", type=Path, default=ROOT / "outputs/baselines/conceptgraphs_scannet_full_v1")
    parser.add_argument("--replica-root", type=Path, default=ROOT / "outputs/baselines/conceptgraphs_replica_full_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/predicted_map_pair_v1")
    args = parser.parse_args()
    result = run(args.scannet_root.resolve(), args.replica_root.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
