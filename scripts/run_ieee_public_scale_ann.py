#!/usr/bin/env python3
"""Measure dense, scene-scoped, and HNSW retrieval on official ReferIt3D records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder
from backend.query.recall_controlled import paired_bootstrap_difference, reciprocal_rank, wilson_interval
from backend.query.tokens import count_tokens

SEED = 20260901
SOURCE_URLS = {
    "nr3d": "https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view",
    "sr3d_plus": "https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def _load_records(nr3d: Path, sr3d_plus: Path) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset, path in (("nr3d", nr3d), ("sr3d_plus", sr3d_plus)):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                text = " ".join(str(row["utterance"]).strip().lower().split())
                key = (str(row["scan_id"]), str(row["target_id"]), text)
                if not text or key in records:
                    continue
                records[key] = {
                    "scene_id": key[0],
                    "object_id": key[1],
                    "text": text,
                    "instance_type": str(row.get("instance_type") or "").strip(),
                    "dataset": dataset,
                    "record_key": f"{key[0]}::{key[1]}::{hashlib.sha1(text.encode()).hexdigest()[:12]}",
                }
    ordered = sorted(records.values(), key=lambda row: _stable_hash(str(row["record_key"])))
    for index, row in enumerate(ordered):
        row["index"] = index
    return ordered


def _embedding_cache(
    records: list[dict[str, Any]],
    cache: Path,
    model_name: str,
) -> np.ndarray:
    metadata = cache.with_suffix(".json")
    fingerprint = hashlib.sha256(
        "\n".join(str(row["record_key"]) for row in records).encode("utf-8")
    ).hexdigest()
    if cache.is_file() and metadata.is_file():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("fingerprint") == fingerprint and info.get("model_name") == model_name:
            values = np.load(cache, mmap_mode="r")
            if len(values) == len(records):
                return values

    cache.parent.mkdir(parents=True, exist_ok=True)
    encoder = FastEmbedSemanticEncoder(model_name=model_name, cache_dir=ROOT / ".cache/fastembed")
    started = time.perf_counter()
    embeddings = encoder.encode([str(row["text"]) for row in records]).astype(np.float32)
    np.save(cache, embeddings)
    metadata.write_text(json.dumps({
        "model_name": model_name,
        "record_count": len(records),
        "dimensions": int(embeddings.shape[1]),
        "fingerprint": fingerprint,
        "encode_seconds": round(time.perf_counter() - started, 6),
    }, indent=2) + "\n", encoding="utf-8")
    return np.load(cache, mmap_mode="r")


def _top_indices(scores: np.ndarray, count: int, excluded: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32).copy()
    if 0 <= excluded < len(scores):
        scores[excluded] = -np.inf
    count = min(count, max(0, len(scores) - 1))
    if count <= 0:
        return []
    selected = np.argpartition(scores, -count)[-count:]
    return [int(value) for value in selected[np.argsort(scores[selected])[::-1]]]


def _quality(ranked: list[int], records: list[dict[str, Any]], query: dict[str, Any]) -> dict[str, Any]:
    target = (str(query["scene_id"]), str(query["object_id"]))
    relevant = [
        index for index in ranked
        if (str(records[index]["scene_id"]), str(records[index]["object_id"])) == target
    ]
    binary = [
        "relevant" if (str(records[index]["scene_id"]), str(records[index]["object_id"])) == target else "other"
        for index in ranked
    ]
    return {
        "hit_at_1": bool(relevant and relevant[0] == ranked[0]),
        "hit_at_3": bool(relevant),
        "mrr": reciprocal_rank(binary, ["relevant"]),
    }


def _summarize(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    count = len(selected)
    summary: dict[str, Any] = {
        "method": method,
        "query_count": count,
        "hit_at_1": round(fmean(float(row["hit_at_1"]) for row in selected), 6),
        "hit_at_3": round(fmean(float(row["hit_at_3"]) for row in selected), 6),
        "mrr": round(fmean(float(row["mrr"]) for row in selected), 6),
        "mean_latency_ms": round(fmean(float(row["latency_ms"]) for row in selected), 6),
        "median_latency_ms": round(float(np.median([row["latency_ms"] for row in selected])), 6),
        "mean_scored_records": round(fmean(float(row["scored_records"]) for row in selected), 6)
        if all(row["scored_records"] is not None for row in selected)
        else None,
        "mean_serialized_top3_tokens": round(
            fmean(float(row["serialized_top3_tokens"]) for row in selected), 6
        ),
    }
    for metric in ("hit_at_1", "hit_at_3"):
        low, high = wilson_interval(sum(bool(row[metric]) for row in selected), count)
        summary[f"{metric}_wilson_95"] = {"low": round(low, 6), "high": round(high, 6)}
    return summary


def _query_indices(records: list[dict[str, Any]], size: int, limit: int) -> list[int]:
    object_counts = Counter(
        (str(row["scene_id"]), str(row["object_id"])) for row in records[:size]
    )
    eligible = [
        index for index, row in enumerate(records[:size])
        if object_counts[(str(row["scene_id"]), str(row["object_id"]))] >= 2
    ]
    return sorted(
        eligible,
        key=lambda index: _stable_hash(f"query:{records[index]['record_key']}"),
    )[:limit]


def run(
    nr3d: Path,
    sr3d_plus: Path,
    out: Path,
    cache: Path,
    model_name: str,
    sizes: list[int],
    queries_per_size: int,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    all_records = _load_records(nr3d, sr3d_plus)
    maximum = min(max(sizes), len(all_records))
    records = all_records[:maximum]
    embeddings = _embedding_cache(records, cache, model_name)
    sizes = sorted({min(value, len(records)) for value in sizes if value > 1})
    per_query: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []

    from usearch.index import Index

    for size in sizes:
        matrix = np.asarray(embeddings[:size], dtype=np.float32)
        scene_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records[:size]):
            scene_indices[str(row["scene_id"])].append(index)
        query_indices = _query_indices(records, size, queries_per_size)

        started = time.perf_counter()
        ann = Index(
            ndim=int(matrix.shape[1]),
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
        )
        ann.add(np.arange(size, dtype=np.uint64), matrix, threads=0)
        build_rows.append({
            "collection_size": size,
            "build_seconds": round(time.perf_counter() - started, 6),
            "dimensions": int(matrix.shape[1]),
            "connectivity": 16,
            "expansion_add": 128,
            "expansion_search": 64,
        })

        for query_index in query_indices:
            query = records[query_index]
            vector = np.asarray(embeddings[query_index], dtype=np.float32)

            started = time.perf_counter()
            flat_ranked = _top_indices(matrix @ vector, 3, query_index)
            flat_ms = (time.perf_counter() - started) * 1000

            scoped_candidates = scene_indices[str(query["scene_id"])]
            scoped_matrix = matrix[scoped_candidates]
            started = time.perf_counter()
            local_ranked = _top_indices(
                scoped_matrix @ vector,
                min(3, len(scoped_candidates) - 1),
                scoped_candidates.index(query_index),
            )
            scoped_ranked = [scoped_candidates[index] for index in local_ranked]
            scoped_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            matches = ann.search(vector, count=min(12, size))
            ann_ranked = [int(key) for key in matches.keys if int(key) != query_index][:3]
            ann_ms = (time.perf_counter() - started) * 1000

            for method, ranked, latency, scored in (
                ("flat_dense_exact", flat_ranked, flat_ms, size),
                ("scene_scoped_dense_exact", scoped_ranked, scoped_ms, len(scoped_candidates)),
                ("hnsw_dense", ann_ranked, ann_ms, None),
            ):
                per_query.append({
                    "collection_size": size,
                    "query_index": query_index,
                    "query_record_key": query["record_key"],
                    "scene_id": query["scene_id"],
                    "object_id": query["object_id"],
                    "dataset": query["dataset"],
                    "method": method,
                    **_quality(ranked, records, query),
                    "latency_ms": round(latency, 6),
                    "scored_records": scored,
                    "ranked_record_keys": [records[index]["record_key"] for index in ranked],
                    "serialized_top3_tokens": sum(count_tokens(records[index]["text"]) for index in ranked),
                })
        size_rows = [row for row in per_query if row["collection_size"] == size]
        methods = ["flat_dense_exact", "scene_scoped_dense_exact", "hnsw_dense"]
        size_summaries = [_summarize(size_rows, method) for method in methods]
        flat_by_query = {
            row["query_record_key"]: row for row in size_rows if row["method"] == "flat_dense_exact"
        }
        for item in size_summaries:
            item["collection_size"] = size
            if item["method"] != "flat_dense_exact":
                selected = [row for row in size_rows if row["method"] == item["method"]]
                for metric in ("hit_at_1", "hit_at_3", "mrr"):
                    item[f"paired_{metric}_difference_vs_flat"] = paired_bootstrap_difference(
                        [float(row[metric]) for row in selected],
                        [float(flat_by_query[str(row["query_record_key"])][metric]) for row in selected],
                        seed=SEED + size,
                    )
        summaries.extend(size_summaries)

    data_manifest = {
        "nr3d": {"path": str(nr3d), "sha256": _sha256(nr3d), "source_url": SOURCE_URLS["nr3d"]},
        "sr3d_plus": {"path": str(sr3d_plus), "sha256": _sha256(sr3d_plus), "source_url": SOURCE_URLS["sr3d_plus"]},
        "raw_rows": {
            "nr3d": sum(1 for _ in csv.DictReader(nr3d.open(encoding="utf-8", newline=""))),
            "sr3d_plus": sum(1 for _ in csv.DictReader(sr3d_plus.open(encoding="utf-8", newline=""))),
        },
        "deduplicated_records_available": len(all_records),
        "deduplicated_records_indexed": len(records),
        "scene_count": len({row["scene_id"] for row in records}),
        "object_count": len({(row["scene_id"], row["object_id"]) for row in records}),
        "instance_type_count": len({row["instance_type"] for row in records}),
    }
    result = {
        "schema_version": "beyond_proximity.ieee_public_scale_ann.v1",
        "generated_at": generated_at,
        "status": "complete",
        "model_name": model_name,
        "seed": SEED,
        "queries_per_size_requested": queries_per_size,
        "collection_sizes": sizes,
        "data_manifest": data_manifest,
        "protocol": {
            "record": "one deduplicated ReferIt3D referring expression linked to a ScanNet scene and target object",
            "query_leakage_control": "the query record itself is excluded; relevance requires another expression linked to the same scene/object",
            "scope": "scene_scoped_dense_exact assumes the active ScanNet scene ID is supplied by external localization; it does not infer the scene from language",
            "flat": "exact cosine search over all records in the measured collection",
            "hierarchical": "exact cosine search restricted to the known-scene branch",
            "ann": "USearch HNSW over the same normalized BGE vectors",
            "top_k": 3,
            "statistics": "Wilson intervals for proportions; paired 10,000-resample bootstrap differences",
        },
        "index_build": build_rows,
        "summaries": summaries,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale_ann_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(per_query, indent=2) + "\n", encoding="utf-8")
    with (out / "scale_ann_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in summaries]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "run_config.json").write_text(json.dumps({
        "generated_at": generated_at,
        "model_name": model_name,
        "seed": SEED,
        "collection_sizes": sizes,
        "queries_per_size": queries_per_size,
        "embedding_cache": str(cache),
    }, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr3d", type=Path, default=ROOT / "external/datasets/scanrefer_public/nr3d.csv")
    parser.add_argument("--sr3d-plus", type=Path, default=ROOT / "external/datasets/scanrefer_public/sr3d/sr3d+.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision/public_scale_ann_v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "external/datasets/scanrefer_public/referit3d_bge_small.npy")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--sizes", default="1000,10000,50000,100000,155536")
    parser.add_argument("--queries-per-size", type=int, default=500)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    result = run(
        args.nr3d.resolve(), args.sr3d_plus.resolve(), args.out.resolve(),
        args.cache.resolve(), args.model, sizes, args.queries_per_size,
    )
    print(json.dumps({
        "status": result["status"],
        "data_manifest": result["data_manifest"],
        "collection_sizes": result["collection_sizes"],
        "summaries": result["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
