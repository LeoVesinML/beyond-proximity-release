#!/usr/bin/env python3
"""Adapt multi-scene ConceptGraphs native exports to canonical query results."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.adapt_conceptgraphs_output import adapt_prediction
    from scripts.baseline_adapter_common import (
        BaselineAdapterError,
        QUERY_RESULT_SCHEMA,
        RUN_CONFIG_SCHEMA,
        load_object,
        require_nonempty_string,
        validate_native_export,
    )
except ModuleNotFoundError:  # Direct script execution adds scripts/ rather than repo root.
    from adapt_conceptgraphs_output import adapt_prediction
    from baseline_adapter_common import (
        BaselineAdapterError,
        QUERY_RESULT_SCHEMA,
        RUN_CONFIG_SCHEMA,
        load_object,
        require_nonempty_string,
        validate_native_export,
    )


CAPTURE_TO_BENCHMARK = {
    "ConferenceHall-capture-pilot": "ConferenceHall",
    "Museume-capture": "Museume",
    "Theater-capture": "Theater",
    "outdoor-street-capture": "outdoor-street",
    "outdoor-drone-capture": "outdoor-drone",
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run-id", default="conceptgraphs_full_v1")
    parser.add_argument("--mode", choices=("live", "cached_live"), default="live")
    parser.add_argument(
        "--dataset-scope",
        choices=("captured", "scannet", "replica"),
        default="captured",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_object(args.benchmark)
    queries = benchmark.get("queries")
    if not isinstance(queries, list):
        raise SystemExit("Benchmark requires queries list")
    query_by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for query in queries:
        if not isinstance(query, dict) or not query.get("query_id"):
            continue
        scene_id = str(query.get("scene_id"))
        query_by_scene.setdefault(scene_id, {})[str(query["query_id"])] = query

    native_files = sorted(args.native_dir.glob("native_results_*.json"))
    if not native_files:
        raise SystemExit(f"No native_results_*.json files under {args.native_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    query_dir = args.out / "query_results"
    query_dir.mkdir(exist_ok=True)

    scenes: list[dict[str, str]] = []
    provenance_by_scene: dict[str, Any] = {}
    scene_stats: list[dict[str, Any]] = []
    canonical_count = 0
    warnings: list[str] = []
    seen_query_ids: set[str] = set()

    for native_path in native_files:
        native = load_object(native_path)
        provenance = native.get("provenance")
        if not isinstance(provenance, dict):
            raise SystemExit(f"{native_path} lacks provenance")
        scene_id = require_nonempty_string(provenance, "scene_id", "provenance")
        benchmark_scene_id = CAPTURE_TO_BENCHMARK.get(scene_id, scene_id)
        validated_provenance, predictions = validate_native_export(
            native_path, native, baseline="conceptgraphs", scene_id=scene_id
        )
        scene_queries = query_by_scene.get(benchmark_scene_id, {})
        scenes.append({
            "scene_id": scene_id,
            "benchmark_scene_id": benchmark_scene_id,
        })
        provenance_by_scene[scene_id] = validated_provenance
        first_metrics = predictions[0].get("metrics") if predictions else {}
        construction_cost = (
            first_metrics.get("construction_cost")
            if isinstance(first_metrics, dict)
            else {}
        )
        native_map_object_count = None
        if isinstance(construction_cost, dict):
            native_map_object_count = construction_cost.get("native_map_object_count")
        found_count = sum(1 for prediction in predictions if prediction.get("found") is True)
        scene_status = "available"
        if native_map_object_count == 0:
            scene_status = "native_map_empty"
            warnings.append(
                f"{scene_id}: native ConceptGraphs map contained zero objects; "
                "canonical outputs are explicit found=false misses."
            )
        scene_stats.append({
            "scene_id": scene_id,
            "benchmark_scene_id": benchmark_scene_id,
            "query_count": len(predictions),
            "found_count": found_count,
            "native_map_object_count": native_map_object_count,
            "status": scene_status,
        })

        for prediction in predictions:
            query_id = require_nonempty_string(prediction, "query_id", "prediction")
            if query_id in seen_query_ids:
                raise SystemExit(f"Duplicate prediction query_id: {query_id}")
            query = scene_queries.get(query_id)
            if query is None:
                raise SystemExit(
                    f"Prediction {query_id} is absent from benchmark scene {benchmark_scene_id}"
                )
            result = adapt_prediction(prediction, query)
            result.update({
                "schema_version": QUERY_RESULT_SCHEMA,
                "run_id": args.run_id,
                "method": "conceptgraphs",
                "mode": args.mode,
                "availability_status": "available",
                "availability_reason": None,
                "scene_id": scene_id,
                "query_id": query_id,
                "query": query.get("query"),
                "query_type": query.get("query_type"),
            })
            write_json(query_dir / f"{query_id}.json", result)
            seen_query_ids.add(query_id)
            canonical_count += 1

    now = datetime.now(timezone.utc).isoformat()
    run_config = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": args.run_id,
        "method": "conceptgraphs",
        "mode": args.mode,
        "status": "complete",
        "scene_ids": [scene["scene_id"] for scene in scenes],
        "scenes": scenes,
        "scene_stats": scene_stats,
        "benchmark_path": str(args.benchmark),
        "native_input": str(args.native_dir),
        "provenance_by_scene": provenance_by_scene,
        "depth_validation": (
            {
                "status": "official_rgbd_metric_depth",
                "reason": f"Native ConceptGraphs consumed official {args.dataset_scope} RGB-D and absolute camera-to-world poses.",
            }
            if args.dataset_scope in {"scannet", "replica"}
            else {
                "status": "unverified_for_native_conceptgraphs_metric_alignment",
                "reason": "RGB-D was valid for native mapping, but metric alignment and independent 3D GT are not established.",
            }
        ),
        "bbox_3d_evaluation": (
            {
                "status": "official_dataset_gt_enabled",
                "gt_source": str(args.benchmark),
                "prediction_coordinate_frame": f"official_{args.dataset_scope}_world",
                "gt_coordinate_frame": f"official_{args.dataset_scope}_world",
                "reason": "Native mapping used absolute dataset camera-to-world poses and metric depth.",
            }
            if args.dataset_scope in {"scannet", "replica"}
            else {
                "status": "coarse_manual_gt_enabled",
                "gt_source": "docs/benchmarks/manual_bbox_gt_v2.json",
                "prediction_coordinate_frame": "conceptgraphs_map_from_source_camera_to_world",
                "gt_coordinate_frame": "camera_to_world",
                "reason": (
                    "ConceptGraphs consumed the captured source camera-to-world poses; "
                    "3D IoU is reported only as a coarse internal regression signal "
                    "against Person 2 manual depth-projected boxes."
                ),
                "not_official_dataset_gt": True,
            }
        ),
        "result_count": canonical_count,
        "finished_at": now,
    }
    write_json(args.out / "run_config.json", run_config)
    write_json(args.out / "logs.json", {
        "run_id": args.run_id,
        "mode": args.mode,
        "finished_at": now,
        "events": [{
            "stage": "multi_scene_native_output_adaptation",
            "status": "complete",
            "count": canonical_count,
        }],
        "warnings": warnings + [
            "ConceptGraphs full run uses native object-map CLIP retrieval, not SemanticSplat graph traversal.",
            "Native absence handling is uncalibrated; negative queries are expected to be difficult for this baseline.",
            "A local empty-detection guard is used for frames where YOLO returns zero boxes before MobileSAM.",
            "Token usage is OpenCLIP text-tokenizer usage, not provider/API billing tokens.",
        ],
    })
    scene_rows = "\n".join(
        "| {scene_id} | {query_count} | {found_count} | {native_map_object_count} | {status} |".format(
            **scene
        )
        for scene in scene_stats
    )
    gt_summary = (
        f"- GT: official `{args.dataset_scope}` object boxes from the benchmark.\n"
        if args.dataset_scope in {"scannet", "replica"}
        else "- Important limitation: captured-scene boxes are coarse manual GT, not official dataset GT.\n"
    )
    iou_summary = (
        "3D IoU uses official public-dataset GT boxes in the dataset world frame.\n\n"
        if args.dataset_scope in {"scannet", "replica"}
        else "3D IoU uses Person 2 manual coarse depth-projected GT boxes as an internal regression signal.\n\n"
    )
    summary = "".join([
        "# ConceptGraphs Full Baseline\n\n",
        f"- Mode: `{args.mode}`\n",
        f"- Scenes: {len(scenes)}\n",
        f"- Canonical results: {canonical_count}\n",
        f"- Native input directory: `{args.native_dir}`\n",
        f"- Scope: `{args.dataset_scope}` scenes with native ConceptGraphs maps and batch CLIP object retrieval.\n",
        gt_summary,
        "- Empty-scene handling: if a native map has zero objects, outputs are explicit `found=false` misses.\n\n",
        "Token usage is measured as OpenCLIP `ViT-H-14` non-padding text-tokenizer tokens, ",
        "not provider/API billing tokens.\n\n",
        iou_summary,
        "## Per-Scene Native Map Status\n\n",
        "| Scene | Queries | Found outputs | Native map objects | Status |\n",
        "|---|---:|---:|---:|---|\n",
        f"{scene_rows}\n",
    ])
    (args.out / "baseline_summary.md").write_text(
        summary,
        encoding="utf-8",
    )
    print(f"Adapted {canonical_count} ConceptGraphs result(s): {args.out}")


if __name__ == "__main__":
    try:
        main()
    except BaselineAdapterError as exc:
        raise SystemExit(f"ConceptGraphs full adapter blocked: {exc}") from exc
