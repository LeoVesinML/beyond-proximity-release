#!/usr/bin/env python3
"""Build a hierarchy from raw RGB-D and poses, then compare it with manual zones."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import filter_five_scene_queries, load_benchmark_queries


SCHEMA_VERSION = "semanticsplat.raw_rgbd_hierarchy_benchmark.v1"
RAW_HIERARCHY_SCHEMA = "semanticsplat.raw_rgbd_hierarchy.v1"
METHODS = ("flat_clip", "manual_hierarchy_clip", "raw_rgbd_hierarchy_clip")
SCENE_DIRS = (
    "ConferenceHall-capture-pilot",
    "Museume-capture",
    "Theater-capture",
    "outdoor-street-capture",
    "outdoor-drone-capture",
)
BENCHMARK_TO_CAPTURE = {
    "ConferenceHall": "ConferenceHall-capture-pilot",
    "Museume": "Museume-capture",
    "Theater": "Theater-capture",
    "outdoor-street": "outdoor-street-capture",
    "outdoor-drone": "outdoor-drone-capture",
}
CLUSTER_LABELS = (
    "auditorium",
    "banquet hall",
    "conference room",
    "corridor",
    "entrance",
    "exhibition gallery",
    "facade",
    "garden",
    "information desk",
    "lobby",
    "lounge",
    "museum display",
    "outdoor plaza",
    "reception desk",
    "service counter",
    "stage",
    "street",
    "theater seating",
    "walkway",
    "wayfinding area",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def l2_normalize(values: np.ndarray, axis: int = 1) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def standardize(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, keepdims=True)
    return (values - values.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)


def deterministic_kmeans(
    features: np.ndarray,
    cluster_count: int,
    *,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """K-means with deterministic farthest-point initialization."""
    if cluster_count < 1 or cluster_count > len(features):
        raise ValueError("cluster_count must be between 1 and the number of rows")
    centers = [0]
    while len(centers) < cluster_count:
        selected = features[np.asarray(centers)]
        distances = ((features[:, None, :] - selected[None, :, :]) ** 2).sum(axis=2)
        nearest = distances.min(axis=1)
        nearest[np.asarray(centers)] = -1
        centers.append(int(np.argmax(nearest)))
    centroids = features[np.asarray(centers)].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if iteration > 1 and np.array_equal(next_labels, labels):
            return labels, centroids, iteration
        labels = next_labels
        for cluster_index in range(cluster_count):
            members = features[labels == cluster_index]
            if len(members):
                centroids[cluster_index] = members.mean(axis=0)
    return labels, centroids, max_iterations


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(features):
        return -1.0
    distances = np.sqrt(
        np.maximum(
            ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )
    values: list[float] = []
    for index, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        own = own[own != index]
        a = float(distances[index, own].mean()) if len(own) else 0.0
        b = min(
            float(distances[index, np.flatnonzero(labels == other)].mean())
            for other in unique
            if other != label
        )
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


def select_cluster_count(features: np.ndarray) -> tuple[int, list[dict[str, Any]]]:
    # The search range scales with the scene instead of imposing a fixed
    # eight-cluster ceiling. At least three observations remain per cluster.
    upper = max(2, len(features) // 3)
    candidates: list[dict[str, Any]] = []
    for cluster_count in range(2, upper + 1):
        labels, _centroids, iterations = deterministic_kmeans(features, cluster_count)
        score = silhouette_score(features, labels)
        candidates.append(
            {
                "cluster_count": cluster_count,
                "silhouette": round(score, 8),
                "iterations": iterations,
            }
        )
    best = max(candidates, key=lambda row: (row["silhouette"], -row["cluster_count"]))
    return int(best["cluster_count"]), candidates


def depth_descriptor(path: Path) -> tuple[list[float], dict[str, Any]]:
    depth = np.load(path, mmap_mode="r")
    flat = np.asarray(depth).reshape(-1)
    stride = max(1, len(flat) // 100_000)
    sampled = np.asarray(flat[::stride], dtype=np.float32)
    valid = sampled[np.isfinite(sampled) & (sampled > 0)]
    if len(valid):
        quantiles = np.quantile(valid, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        valid_ratio = float(len(valid) / len(sampled))
    else:
        quantiles = [0.0] * 5
        valid_ratio = 0.0
    descriptor = [valid_ratio, *[float(value) for value in quantiles]]
    return descriptor, {
        "shape": list(depth.shape),
        "sample_count": int(len(sampled)),
        "valid_sample_count": int(len(valid)),
        "valid_ratio": round(valid_ratio, 8),
        "quantiles": [round(float(value), 8) for value in quantiles],
    }


def load_raw_scene(scene_dir: Path) -> list[dict[str, Any]]:
    """Load raw capture files only; this function must never access views/*.json."""
    transform_path = scene_dir / "transforms.json"
    data = json.loads(transform_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for frame in data.get("frames", []):
        view_id = str(frame["view_id"])
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        image_path = scene_dir / str(frame["file_path"])
        depth_path = scene_dir / str(frame["depth_file_path"])
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing raw RGB-D pair for {scene_dir.name}/{view_id}")
        descriptor, depth_stats = depth_descriptor(depth_path)
        records.append(
            {
                "view_id": view_id,
                "image_path": image_path,
                "depth_path": depth_path,
                "pose_xyz": matrix[:3, 3].astype(float).tolist(),
                "depth_descriptor": descriptor,
                "depth_stats": depth_stats,
            }
        )
    return sorted(records, key=lambda row: row["view_id"])


class ClipEncoder:
    def __init__(self, model_id: str, *, cache_dir: Path, threads: int) -> None:
        torch.set_num_threads(max(1, threads))
        self.model_id = model_id
        started = time.perf_counter()
        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode_images(
        self,
        records: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        embeddings: list[np.ndarray] = []
        calls: list[dict[str, Any]] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            images = []
            for record in batch:
                with Image.open(record["image_path"]) as source:
                    images.append(source.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
                values = self.model.visual_projection(vision.pooler_output)
            latency_ms = (time.perf_counter() - started) * 1000
            matrix = l2_normalize(values.detach().cpu().numpy())
            embeddings.append(matrix)
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "view_ids": [record["view_id"] for record in batch],
                    "image_count": len(batch),
                    "latency_ms": round(latency_ms, 6),
                    "device": "cpu",
                    "text_tokens": 0,
                }
            )
        return np.vstack(embeddings), calls

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
        embeddings: list[np.ndarray] = []
        calls: list[dict[str, Any]] = []
        token_counts: list[int] = []
        tokenizer = self.processor.tokenizer
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            counts = inputs["attention_mask"].sum(dim=1).tolist()
            token_counts.extend(int(value) for value in counts)
            started = time.perf_counter()
            with torch.inference_mode():
                text = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                values = self.model.text_projection(text.pooler_output)
            latency_ms = (time.perf_counter() - started) * 1000
            embeddings.append(l2_normalize(values.detach().cpu().numpy()))
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "query_indices": list(range(start, start + len(batch))),
                    "query_count": len(batch),
                    "input_tokens": int(sum(counts)),
                    "latency_ms": round(latency_ms, 6),
                    "token_method": f"transformers_native:{tokenizer.name_or_path}",
                }
            )
        return np.vstack(embeddings), calls, token_counts


def construct_raw_hierarchy(
    scene_root: Path,
    encoder: ClipEncoder,
    *,
    image_batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[str]]]:
    """Construct every scene hierarchy before any manual annotation is loaded."""
    output: dict[str, Any] = {
        "schema_version": RAW_HIERARCHY_SCHEMA,
        "generated_at": utc_now(),
        "status": "complete",
        "source_policy": {
            "allowed": ["transforms.json", "images/*", "depths/*"],
            "forbidden_during_construction": ["views/*.json", "tree/*.json", "manual_zone_gt_v2.json"],
            "manual_viewjson_files_read": 0,
            "manual_zone_files_read": 0,
        },
        "model": {
            "model_id": encoder.model_id,
            "revision": encoder.revision,
            "role": "frozen image feature extractor",
        },
        "scenes": [],
    }
    label_embeddings, label_calls, label_token_counts = encoder.encode_queries(
        [f"a photograph of a {label}" for label in CLUSTER_LABELS],
        batch_size=len(CLUSTER_LABELS),
    )
    output["semantic_labeling"] = {
        "vocabulary": list(CLUSTER_LABELS),
        "prompt_template": "a photograph of a {label}",
        "calls": label_calls,
        "input_tokens": sum(label_token_counts),
        "manual_scene_specific_labels": False,
    }
    image_embeddings: dict[str, np.ndarray] = {}
    view_ids_by_scene: dict[str, list[str]] = {}
    for scene_name in SCENE_DIRS:
        scene_dir = scene_root / scene_name
        started = time.perf_counter()
        records = load_raw_scene(scene_dir)
        clip_embeddings, image_calls = encoder.encode_images(
            records,
            batch_size=image_batch_size,
        )
        pose = standardize(np.asarray([record["pose_xyz"] for record in records]))
        depth = standardize(
            np.asarray([record["depth_descriptor"] for record in records], dtype=np.float64)
        )
        combined = np.concatenate((clip_embeddings, 0.5 * pose, 0.2 * depth), axis=1)
        combined = standardize(combined)
        selected_count, candidates = select_cluster_count(combined)
        labels, centroids, iterations = deterministic_kmeans(combined, selected_count)
        clusters = []
        for cluster_index in range(selected_count):
            indices = np.flatnonzero(labels == cluster_index)
            clip_centroid = l2_normalize(
                clip_embeddings[indices].mean(axis=0, keepdims=True)
            )[0]
            label_scores = label_embeddings @ clip_centroid
            predicted_labels = [
                {
                    "label": CLUSTER_LABELS[label_index],
                    "score": round(float(label_scores[label_index]), 8),
                }
                for label_index in np.argsort(-label_scores)[:3]
            ]
            clusters.append(
                {
                    "cluster_id": f"raw_cluster_{cluster_index + 1:02d}",
                    "predicted_label": predicted_labels[0]["label"],
                    "predicted_labels": predicted_labels,
                    "view_ids": [records[index]["view_id"] for index in indices],
                    "member_count": int(len(indices)),
                    "feature_centroid": centroids[cluster_index].round(8).tolist(),
                    "clip_centroid": clip_centroid.round(8).tolist(),
                }
            )
        image_embeddings[scene_name] = clip_embeddings
        view_ids_by_scene[scene_name] = [record["view_id"] for record in records]
        output["scenes"].append(
            {
                "scene_id": scene_name,
                "view_count": len(records),
                "rgb_image_count": len(records),
                "depth_map_count": len(records),
                "pose_count": len(records),
                "selected_cluster_count": selected_count,
                "cluster_selection": candidates,
                "kmeans_iterations": iterations,
                "feature_weights": {"clip": 1.0, "pose": 0.5, "depth": 0.2},
                "clusters": clusters,
                "views": [
                    {
                        "view_id": record["view_id"],
                        "image_path": record["image_path"].relative_to(ROOT).as_posix(),
                        "depth_path": record["depth_path"].relative_to(ROOT).as_posix(),
                        "pose_xyz": [round(float(value), 8) for value in record["pose_xyz"]],
                        "depth_stats": record["depth_stats"],
                    }
                    for record in records
                ],
                "image_encoder_calls": image_calls,
                "construction_latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    6,
                ),
            }
        )
    output["totals"] = {
        "scene_count": len(output["scenes"]),
        "rgb_image_count": sum(scene["rgb_image_count"] for scene in output["scenes"]),
        "depth_map_count": sum(scene["depth_map_count"] for scene in output["scenes"]),
        "pose_count": sum(scene["pose_count"] for scene in output["scenes"]),
        "image_encoder_calls": sum(
            len(scene["image_encoder_calls"]) for scene in output["scenes"]
        ),
        "semantic_label_encoder_calls": len(label_calls),
        "model_call_count": (
            sum(len(scene["image_encoder_calls"]) for scene in output["scenes"])
            + len(label_calls)
        ),
        "construction_latency_ms": round(
            sum(scene["construction_latency_ms"] for scene in output["scenes"]),
            6,
        ),
        "text_tokens": sum(label_token_counts),
    }
    return output, image_embeddings, view_ids_by_scene


def load_manual_zones(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(scene["captured_scene_id"]): [
            {
                "cluster_id": str(zone["zone_id"]),
                "view_ids": [str(value) for value in zone.get("view_ids", [])],
            }
            for zone in scene.get("zones", [])
        ]
        for scene in data.get("scenes", [])
    }


def pairwise_zone_metrics(
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    view_ids: Iterable[str],
) -> dict[str, Any]:
    def memberships(zones: list[dict[str, Any]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for zone in zones:
            for view_id in zone["view_ids"]:
                result.setdefault(view_id, set()).add(zone["cluster_id"])
        return result

    pred = memberships(predicted)
    ref = memberships(reference)
    tp = fp = fn = tn = 0
    ordered = sorted(view_ids)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            predicted_same = bool(pred.get(left, set()) & pred.get(right, set()))
            reference_same = bool(ref.get(left, set()) & ref.get(right, set()))
            if predicted_same and reference_same:
                tp += 1
            elif predicted_same:
                fp += 1
            elif reference_same:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pair_count": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pairwise_precision": round(precision, 6),
        "pairwise_recall": round(recall, 6),
        "pairwise_f1": round(f1, 6),
        "rand_index": round((tp + tn) / max(tp + fp + fn + tn, 1), 6),
    }


def cluster_centroids(
    clusters: list[dict[str, Any]],
    embeddings: np.ndarray,
    view_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    index = {view_id: position for position, view_id in enumerate(view_ids)}
    values = []
    ids = []
    for cluster in clusters:
        indices = [index[value] for value in cluster["view_ids"] if value in index]
        if not indices:
            continue
        values.append(l2_normalize(embeddings[indices].mean(axis=0, keepdims=True))[0])
        ids.append(cluster["cluster_id"])
    return np.asarray(values), ids


def expected_views(query: dict[str, Any]) -> list[str]:
    return [str(value) for value in query.get("expected_view_ids", []) if value]


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    target = set(expected)
    for index, value in enumerate(ranked, 1):
        if value in target:
            return 1.0 / index
    return 0.0


def rank_method(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    view_ids: list[str],
    *,
    clusters: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str], list[str]]:
    if clusters is None:
        candidate_ids = list(view_ids)
        selected_clusters: list[str] = []
    else:
        centroids, cluster_ids = cluster_centroids(clusters, embeddings, view_ids)
        cluster_scores = centroids @ query_embedding
        selected_clusters = [
            cluster_ids[index] for index in np.argsort(-cluster_scores)[:2]
        ]
        candidate_ids = sorted(
            {
                view_id
                for cluster in clusters
                if cluster["cluster_id"] in selected_clusters
                for view_id in cluster["view_ids"]
            }
        )
    view_index = {view_id: position for position, view_id in enumerate(view_ids)}
    scores = [
        (float(embeddings[view_index[view_id]] @ query_embedding), view_id)
        for view_id in candidate_ids
    ]
    ranked = [view_id for _score, view_id in sorted(scores, key=lambda row: (-row[0], row[1]))]
    return ranked, candidate_ids, selected_clusters


def optional_mean(values: Iterable[float | int | bool | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(mean(clean), 6) if clean else None


def method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    results = [row["methods"][method] for row in rows]
    quality = [result for result in results if result["quality_eligible"]]
    return {
        "query_count": len(results),
        "quality_denominator": len(quality),
        "hit_at_1": optional_mean(result["hit_at_1"] for result in quality),
        "hit_at_3": optional_mean(result["hit_at_3"] for result in quality),
        "mrr": optional_mean(result["reciprocal_rank"] for result in quality),
        "mean_views_checked": optional_mean(result["views_checked"] for result in results),
        "mean_query_input_tokens": optional_mean(
            result["query_input_tokens"] for result in results
        ),
        "total_query_input_tokens": sum(
            int(result["query_input_tokens"]) for result in results
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "query_id",
        "scene_id",
        "query",
        "method",
        "quality_eligible",
        "hit_at_1",
        "hit_at_3",
        "reciprocal_rank",
        "selected_view_id",
        "views_checked",
        "selected_cluster_ids",
        "query_input_tokens",
        "query_encoder_latency_ms",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                value = row["methods"][method]
                writer.writerow(
                    {
                        "query_id": row["query_id"],
                        "scene_id": row["scene_id"],
                        "query": row["query"],
                        "method": method,
                        **{key: value[key] for key in fields if key in value},
                        "selected_cluster_ids": "|".join(value["selected_cluster_ids"]),
                    }
                )


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Raw RGB-D Hierarchy Benchmark",
        "",
        f"- RGB images: {summary['construction']['rgb_image_count']}",
        f"- Depth maps: {summary['construction']['depth_map_count']}",
        f"- Camera poses: {summary['construction']['pose_count']}",
        f"- Construction model calls: {summary['construction']['model_call_count']}",
        f"- Construction label tokens: {summary['construction']['text_tokens']}",
        f"- Construction latency: {summary['construction']['construction_latency_ms'] / 1000:.2f} s",
        f"- Manual ViewJSON files read during construction: {summary['construction']['manual_viewjson_files_read']}",
        "",
        "| Method | hit@1 | hit@3 | MRR | Views checked | Query tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = summary["methods"][method]
        lines.append(
            f"| {method} | {value['hit_at_1']:.3f} | {value['hit_at_3']:.3f} | "
            f"{value['mrr']:.3f} | {value['mean_views_checked']:.2f} | "
            f"{value['mean_query_input_tokens']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- Macro pairwise F1 against manual zones: {summary['structure']['macro_pairwise_f1']:.3f}",
            f"- Macro Rand index against manual zones: {summary['structure']['macro_rand_index']:.3f}",
            "",
            "Construction uses raw RGB images, depth maps, and camera poses only. "
            "Manual zones and expected view IDs are loaded after construction for evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoder(
        args.model,
        cache_dir=args.model_cache.resolve(),
        threads=args.threads,
    )

    # Freeze the automatic hierarchy before any manual reference is opened.
    hierarchy, image_embeddings, view_ids_by_scene = construct_raw_hierarchy(
        args.scene_root.resolve(),
        encoder,
        image_batch_size=args.image_batch_size,
    )
    write_json(out_dir / "raw_hierarchy.json", hierarchy)

    manual_zones = load_manual_zones(args.manual_zones.resolve())
    raw_by_scene = {
        scene["scene_id"]: scene["clusters"] for scene in hierarchy["scenes"]
    }
    structure_by_scene = {}
    for scene_name in SCENE_DIRS:
        structure_by_scene[scene_name] = pairwise_zone_metrics(
            raw_by_scene[scene_name],
            manual_zones[scene_name],
            view_ids_by_scene[scene_name],
        )

    queries = filter_five_scene_queries(load_benchmark_queries(args.benchmark.resolve()))
    query_embeddings, query_calls, query_token_counts = encoder.encode_queries(
        [str(query["query"]) for query in queries],
        batch_size=args.text_batch_size,
    )
    call_by_index = {
        index: call
        for call in query_calls
        for index in call["query_indices"]
    }
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        scene_name = BENCHMARK_TO_CAPTURE[str(query["scene_id"])]
        embeddings = image_embeddings[scene_name]
        view_ids = view_ids_by_scene[scene_name]
        expected = expected_views(query)
        methods = {}
        variants = {
            "flat_clip": None,
            "manual_hierarchy_clip": manual_zones[scene_name],
            "raw_rgbd_hierarchy_clip": raw_by_scene[scene_name],
        }
        for method, clusters in variants.items():
            ranked, candidates, selected_clusters = rank_method(
                query_embeddings[index],
                embeddings,
                view_ids,
                clusters=clusters,
            )
            eligible = bool(expected)
            methods[method] = {
                "ranked_view_ids": ranked[:3],
                "selected_view_id": ranked[0] if ranked else None,
                "candidate_view_ids": candidates,
                "views_checked": len(candidates),
                "selected_cluster_ids": selected_clusters,
                "quality_eligible": eligible,
                "expected_view_ids": expected,
                "hit_at_1": bool(ranked[:1] and ranked[0] in expected) if eligible else None,
                "hit_at_3": bool(set(ranked[:3]) & set(expected)) if eligible else None,
                "reciprocal_rank": reciprocal_rank(ranked, expected) if eligible else None,
                "query_input_tokens": query_token_counts[index],
                "query_encoder_call": int(call_by_index[index]["call_index"]),
                "query_encoder_latency_ms": round(
                    float(call_by_index[index]["latency_ms"])
                    / int(call_by_index[index]["query_count"]),
                    6,
                ),
            }
        rows.append(
            {
                "query_id": str(query["query_id"]),
                "scene_id": str(query["scene_id"]),
                "captured_scene_id": scene_name,
                "query": str(query["query"]),
                "query_type": str(query.get("query_type") or "unspecified"),
                "methods": methods,
            }
        )

    method_values = {method: method_summary(rows, method) for method in METHODS}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "complete",
        "query_count": len(rows),
        "quality_query_count": sum(
            1 for row in rows if row["methods"]["flat_clip"]["quality_eligible"]
        ),
        "model": {
            "model_id": args.model,
            "revision": encoder.revision,
            "role": "frozen VLM image/text encoder",
            "framework": "transformers",
            "torch_version": torch.__version__,
            "load_seconds": round(encoder.load_seconds, 6),
            "device": "cpu",
        },
        "construction": {
            **hierarchy["totals"],
            "manual_viewjson_files_read": hierarchy["source_policy"][
                "manual_viewjson_files_read"
            ],
            "manual_zone_files_read": hierarchy["source_policy"]["manual_zone_files_read"],
        },
        "structure": {
            "by_scene": structure_by_scene,
            "macro_pairwise_f1": optional_mean(
                value["pairwise_f1"] for value in structure_by_scene.values()
            ),
            "macro_rand_index": optional_mean(
                value["rand_index"] for value in structure_by_scene.values()
            ),
            "reference": args.manual_zones.resolve().relative_to(ROOT).as_posix(),
            "reference_loaded_after_construction": True,
        },
        "query_encoder": {
            "model_call_count": len(query_calls),
            "input_tokens": sum(query_token_counts),
            "latency_ms": round(sum(call["latency_ms"] for call in query_calls), 6),
            "calls": query_calls,
        },
        "methods": method_values,
        "comparisons": {
            "raw_vs_flat_views_reduction_pct": round(
                100
                * (
                    1
                    - method_values["raw_rgbd_hierarchy_clip"]["mean_views_checked"]
                    / method_values["flat_clip"]["mean_views_checked"]
                ),
                6,
            ),
            "manual_vs_flat_views_reduction_pct": round(
                100
                * (
                    1
                    - method_values["manual_hierarchy_clip"]["mean_views_checked"]
                    / method_values["flat_clip"]["mean_views_checked"]
                ),
                6,
            ),
        },
        "inputs": {
            "benchmark": {
                "path": args.benchmark.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(args.benchmark.resolve()),
            },
            "manual_zones": {
                "path": args.manual_zones.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(args.manual_zones.resolve()),
                "evaluation_only": True,
            },
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": args.threads,
        },
        "git_commit": git_commit(),
    }
    write_json(out_dir / "per_query_results.json", rows)
    write_json(out_dir / "metrics_summary.json", summary)
    write_json(
        out_dir / "run_config.json",
        {
            "schema_version": "semanticsplat.raw_rgbd_hierarchy_run_config.v1",
            "model": args.model,
            "seed": 20260724,
            "image_batch_size": args.image_batch_size,
            "text_batch_size": args.text_batch_size,
            "threads": args.threads,
            "source_policy": hierarchy["source_policy"],
            "feature_weights": {"clip": 1.0, "pose": 0.5, "depth": 0.2},
            "cluster_count_selection": "maximum silhouette over k=2..floor(n/3)",
            "manual_reference_loaded_after_raw_hierarchy_write": True,
        },
    )
    write_csv(out_dir / "per_query_metrics.csv", rows)
    (out_dir / "metrics_summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "backend" / "data" / "scenes",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json",
    )
    parser.add_argument(
        "--manual-zones",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "manual_zone_gt_v2.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "raw_rgbd_hierarchy" / "five_scene_clip_v1",
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--threads", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
