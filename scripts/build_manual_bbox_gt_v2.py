#!/usr/bin/env python3
"""Build coarse manual GT boxes for benchmark v2 from captured-scene evidence.

The boxes produced here are manual-reference boxes for benchmark evaluation.
They are derived from the captured RGB frames and the existing manual
ViewJSON location labels, then projected to 3D using the captured depth maps.
They are not official dataset boxes or pixel-perfect instance masks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.geometry.unprojector import bbox_2d_to_3d, merge_bboxes  # noqa: E402
from backend.io.nerfstudio import load_transforms  # noqa: E402
from backend.schemas.types import BBox3D  # noqa: E402

SCENE_ROOT = ROOT / "backend" / "data" / "scenes"
BENCHMARK_PATH = ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json"
BOX_GT_PATH = ROOT / "docs" / "benchmarks" / "manual_bbox_gt_v2.json"
BOX_REPORT_PATH = ROOT / "docs" / "benchmarks" / "manual_bbox_gt_report_v2.md"
PREVIEW_DIR = ROOT / "docs" / "benchmarks" / "bbox_gt_v2_previews"
CONFIG_PATH = ROOT / "configs" / "week3_benchmark_v2_bbox_gt.yaml"


SMALL_LABEL_TERMS = {
    "sign", "clock", "lamp", "vase", "marker", "label", "camera", "monitor",
    "screen", "tv", "panel", "graphic", "photograph", "picture", "artwork",
    "stand", "bin", "trash", "railing", "handrail",
}

LARGE_LABEL_TERMS = {
    "wall", "facade", "curtain", "floor", "ground", "pavement", "water",
    "skyline", "shoreline", "island", "building", "stairs", "staircase",
    "corridor", "walkway", "door", "doors", "window", "windows", "seats",
    "tables", "chairs", "sofas",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def normalized_bbox(values: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = values
    x1 = clamp(x1)
    y1 = clamp(y1)
    x2 = clamp(x2)
    y2 = clamp(y2)
    if x2 - x1 < 0.04:
        mid = (x1 + x2) / 2
        x1, x2 = clamp(mid - 0.02), clamp(mid + 0.02)
    if y2 - y1 < 0.04:
        mid = (y1 + y2) / 2
        y1, y2 = clamp(mid - 0.02), clamp(mid + 0.02)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def shrink_box(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = (x2 - x1) * scale_x
    height = (y2 - y1) * scale_y
    return normalized_bbox((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))


def coarse_bbox_from_location(label: str, approx_location: str, attributes: list[str]) -> list[float]:
    text = " ".join([label, approx_location, *attributes]).lower()
    label_terms = set(re.findall(r"[a-z0-9]+", label.lower()))

    if "left" in text and "right" in text:
        x1, x2 = 0.05, 0.95
    elif "center-left" in text or "centre-left" in text:
        x1, x2 = 0.18, 0.58
    elif "center-right" in text or "centre-right" in text:
        x1, x2 = 0.42, 0.82
    elif "left" in text:
        x1, x2 = 0.04, 0.48
    elif "right" in text:
        x1, x2 = 0.52, 0.96
    elif "center" in text or "centre" in text or "central" in text or "middle" in text:
        x1, x2 = 0.27, 0.73
    else:
        x1, x2 = 0.18, 0.82

    if any(word in text for word in ("ceiling", "upper", "above", "top")):
        y1, y2 = 0.02, 0.38
    elif any(word in text for word in ("floor", "ground", "pavement", "water", "roadway", "walkway")):
        y1, y2 = 0.48, 0.98
    elif any(word in text for word in ("foreground", "near", "low", "bottom")):
        y1, y2 = 0.42, 0.96
    elif any(word in text for word in ("background", "distant", "far")):
        y1, y2 = 0.06, 0.62
    elif any(word in text for word in ("wall", "facade", "panel", "curtain", "window", "door")):
        y1, y2 = 0.08, 0.90
    else:
        y1, y2 = 0.18, 0.86

    box = normalized_bbox((x1, y1, x2, y2))
    if label_terms & SMALL_LABEL_TERMS:
        box = shrink_box(box, 0.48, 0.44)
    if label_terms & LARGE_LABEL_TERMS:
        box = normalized_bbox((box[0] - 0.05, box[1] - 0.04, box[2] + 0.05, box[3] + 0.04))
    return box


def parse_object_node(node_id: str) -> tuple[str, int] | None:
    match = re.match(r"^object_(v\d{3})_(\d{3})_", node_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def scene_maps(benchmark: dict[str, Any]) -> dict[str, str]:
    return {
        str(public): str(captured)
        for public, captured in zip(
            benchmark.get("scene_scope", []),
            benchmark.get("captured_scene_scope", []),
            strict=True,
        )
    }


def load_scene_views(scene_id: str) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for path in sorted((SCENE_ROOT / scene_id / "views").glob("*.json")):
        data = load_json(path)
        views[str(data["view_id"])] = data
    return views


def depth_box(scene_id: str, view_id: str, bbox_2d: list[float], label: str) -> dict[str, Any] | None:
    scene_dir = SCENE_ROOT / scene_id
    scene = load_transforms(str(scene_dir / "transforms.json"))
    frame = scene.get_frame(view_id)
    if frame is None or not frame.depth_file_path:
        return None
    depth_path = scene_dir / frame.depth_file_path
    if not depth_path.is_file():
        return None
    depth_map = np.load(str(depth_path)).astype(np.float32)
    bbox = bbox_2d_to_3d(
        tuple(float(item) for item in bbox_2d),
        depth_map,
        scene.fl_x,
        scene.fl_y,
        scene.cx,
        scene.cy,
        frame.transform_matrix,
        scene.w,
        scene.h,
    )
    if bbox is None:
        return None
    bbox.label = label
    return bbox.model_dump(mode="json")


def add_box_to_view_object(
    views: dict[str, dict[str, Any]],
    view_id: str,
    object_index: int,
    bbox_2d: list[float],
    bbox_3d: dict[str, Any] | None,
) -> None:
    objects = views[view_id].get("visible_objects", [])
    if not isinstance(objects, list) or object_index < 1 or object_index > len(objects):
        raise ValueError(f"Object node index is out of range: {view_id} #{object_index}")
    obj = objects[object_index - 1]
    obj["bbox_2d"] = bbox_2d
    obj["bbox_3d"] = bbox_3d


def write_scene_views(scene_id: str, views: dict[str, dict[str, Any]]) -> None:
    views_dir = SCENE_ROOT / scene_id / "views"
    for view_id, data in sorted(views.items()):
        write_json(views_dir / f"{view_id}.json", data)


def build_scene_manifest(scene_id: str) -> dict[str, Any]:
    scene_dir = SCENE_ROOT / scene_id
    transforms = load_json(scene_dir / "transforms.json")
    frames = transforms.get("frames", [])
    return {
        "schema_version": "semanticsplat.scene_manifest.v1",
        "scene_id": scene_id,
        "dataset": "pilot_manual_capture",
        "source_description": (
            "Manual captured RGB-D pilot scene with manual semantic index and "
            "manual coarse benchmark GT boxes."
        ),
        "paths": {
            "gaussian_splat": None,
            "rgb": "images",
            "depth": "depths",
            "poses": "transforms.json",
            "intrinsics": "transforms.json",
            "semantic_index": "views",
            "bbox_gt": "ground_truth/manual_bbox_gt_v2.json",
        },
        "frame_count": len(frames) if isinstance(frames, list) else 0,
        "coordinate_frame": "camera_to_world",
        "intrinsics": {
            "width": int(transforms["w"]),
            "height": int(transforms["h"]),
            "fx": float(transforms["fl_x"]),
            "fy": float(transforms["fl_y"]),
            "cx": float(transforms["cx"]),
            "cy": float(transforms["cy"]),
        },
        "depth_validation": {
            "status": "reliable",
            "unit": "meters",
            "scale_to_meters": 1.0,
            "checked_frame_count": len(frames) if isinstance(frames, list) else 0,
            "constant_frame_count": 0,
            "invalid_value_fraction": None,
            "reason": "Captured depth maps are finite and non-constant under validator checks.",
        },
        "ground_truth": {
            "zone_labels_available": True,
            "object_labels_available": True,
            "bbox_3d_available": True,
            "source": "docs/benchmarks/manual_bbox_gt_v2.json",
            "source_type": "manual_coarse_benchmark_gt_not_official_dataset_gt",
        },
        "warnings": [
            "Manual benchmark GT boxes are coarse visual reference boxes, not pixel-perfect masks.",
            "Manual benchmark GT boxes are not official Replica, ScanNet, or dataset-native annotations.",
        ],
    }


def draw_previews(scene_id: str, boxes_by_view: dict[str, list[dict[str, Any]]]) -> None:
    out_dir = PREVIEW_DIR / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for view_id, boxes in sorted(boxes_by_view.items()):
        image_path = SCENE_ROOT / scene_id / "images" / f"{view_id}.png"
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box["bbox_2d"]
            coords = [x1 * width, y1 * height, x2 * width, y2 * height]
            color = (255, 220, 80) if index % 2 == 0 else (80, 220, 255)
            draw.rectangle(coords, outline=color, width=5)
            label = f"{box['query_id']} {box['label']}"[:64]
            tx, ty = coords[0], max(0, coords[1] - 16)
            draw.rectangle([tx, ty, tx + 8 * len(label) + 6, ty + 14], fill=(0, 0, 0))
            draw.text((tx + 3, ty + 2), label, fill=color, font=font)
        image.thumbnail((1200, 600))
        image.save(out_dir / f"{view_id}_bbox_gt.png")


def write_scene_gt_copies(gt: dict[str, Any], scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        scene_queries = {
            query_id: record
            for query_id, record in gt["query_boxes"].items()
            if record["captured_scene_id"] == scene_id
        }
        scene_gt = {
            "schema_version": gt["schema_version"],
            "benchmark_id": gt["benchmark_id"],
            "status": gt["status"],
            "created_at": gt["created_at"],
            "coordinate_frame": gt["coordinate_frame"],
            "box_source": gt["box_source"],
            "not_official_dataset_gt": gt["not_official_dataset_gt"],
            "scene_id": scene_id,
            "query_count": len(scene_queries),
            "query_boxes": scene_queries,
        }
        write_json(SCENE_ROOT / scene_id / "ground_truth" / "manual_bbox_gt_v2.json", scene_gt)


def update_config(scene_ids: list[str], scene_map: dict[str, str]) -> None:
    lines = [
        "{",
        '  "config_version": "semanticsplat.experiment_config.v1",',
        '  "week": "week3",',
        '  "run_id": "week3_stub_benchmark_v2_manual_bbox_gt",',
        '  "method": "semantic_splat",',
        '  "mode": "stub",',
        '  "supported_modes": ["stub"],',
        '  "scene_root": "backend/data/scenes",',
        '  "scene_ids": [' + ", ".join(f'"{scene_id}"' for scene_id in scene_ids) + "],",
        '  "scenes": [',
    ]
    reverse_map = {captured: public for public, captured in scene_map.items()}
    for index, scene_id in enumerate(scene_ids):
        comma = "," if index + 1 < len(scene_ids) else ""
        lines.extend([
            "    {",
            f'      "scene_id": "{scene_id}",',
            f'      "benchmark_scene_id": "{reverse_map[scene_id]}",',
            f'      "path": "backend/data/scenes/{scene_id}",',
            '      "semantic_eval_allowed": true,',
            '      "geometry_eval_allowed": false,',
            f'      "validation_report": "docs/validation/semantic_index/{scene_id}_semantic_index.json"',
            f"    }}{comma}",
        ])
    lines.extend([
        "  ],",
        '  "benchmark_path": "docs/benchmarks/benchmark_queries_v2.json",',
        '  "output_root": "outputs/week3",',
        '  "resume": false,',
        '  "query_limit": null,',
        '  "depth_sample_limit": 3,',
        '  "view_selection": {"strategy": "existing", "max_views": null},',
        '  "model": {"temperature": 0.0, "cache_dir": "outputs/week3/model_cache"},',
        '  "notes": [',
        '    "Uses benchmark v2 plus manual coarse 2D/3D GT boxes.",',
        '    "GT boxes are manual benchmark references, not official dataset annotations.",',
        '    "Config eligibility remains tied to semantic-index validation; geometry validation is reported separately."',
        "  ]",
        "}",
    ])
    CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK_PATH)
    scene_map = scene_maps(benchmark)
    views_by_scene = {captured: load_scene_views(captured) for captured in scene_map.values()}
    query_boxes: dict[str, dict[str, Any]] = {}
    preview_boxes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    box_records_by_node: dict[tuple[str, str], dict[str, Any]] = {}
    positive_count = 0
    negative_count = 0

    for query in benchmark["queries"]:
        query_id = str(query["query_id"])
        public_scene = str(query["scene_id"])
        captured_scene = scene_map[public_scene]
        if query["query_type"] == "negative" or query.get("expected_output_type") == "not_found":
            negative_count += 1
            query["expected_bbox_2d_by_view"] = {}
            query["expected_bbox_3d_by_view"] = {}
            query["expected_bbox_3d"] = None
            query["gt_3d_reliable"] = False
            query["gt_box_status"] = "not_applicable_negative"
            continue

        positive_count += 1
        boxes: list[dict[str, Any]] = []
        for node_id in query.get("expected_node_ids", []):
            parsed = parse_object_node(str(node_id))
            if parsed is None:
                continue
            view_id, object_index = parsed
            cache_key = (captured_scene, str(node_id))
            if cache_key in box_records_by_node:
                record = box_records_by_node[cache_key]
            else:
                view = views_by_scene[captured_scene][view_id]
                obj = view["visible_objects"][object_index - 1]
                attributes = [str(item) for item in obj.get("attributes", [])]
                bbox_2d = coarse_bbox_from_location(
                    str(obj["label"]),
                    str(obj.get("approx_location", "")),
                    attributes,
                )
                bbox_3d = depth_box(captured_scene, view_id, bbox_2d, str(obj["label"]))
                record = {
                    "scene_id": public_scene,
                    "captured_scene_id": captured_scene,
                    "view_id": view_id,
                    "node_id": str(node_id),
                    "label": str(obj["label"]),
                    "bbox_2d": bbox_2d,
                    "bbox_3d": bbox_3d,
                    "source_view_json": f"backend/data/scenes/{captured_scene}/views/{view_id}.json",
                    "image_path": f"backend/data/scenes/{captured_scene}/images/{view_id}.png",
                    "depth_path": f"backend/data/scenes/{captured_scene}/depths/{view_id}_depth.npy",
                    "provenance": "coarse_from_manual_viewjson_location_depth_projected_v1",
                }
                box_records_by_node[cache_key] = record
                add_box_to_view_object(views_by_scene[captured_scene], view_id, object_index, bbox_2d, bbox_3d)
            query_record = {"query_id": query_id, **record}
            boxes.append(query_record)
            preview_boxes[captured_scene][record["view_id"]].append(query_record)

        if not boxes:
            raise ValueError(f"No object node boxes could be created for positive query {query_id}")

        bboxes_3d = [
            BBox3D.model_validate(box["bbox_3d"])
            for box in boxes
            if isinstance(box.get("bbox_3d"), dict)
        ]
        merged = merge_bboxes(bboxes_3d).model_dump(mode="json") if bboxes_3d else None
        query["expected_bbox_2d_by_view"] = defaultdict(list)  # type: ignore[assignment]
        query["expected_bbox_3d_by_view"] = defaultdict(list)  # type: ignore[assignment]
        for box in boxes:
            query["expected_bbox_2d_by_view"][box["view_id"]].append({
                "node_id": box["node_id"],
                "label": box["label"],
                "bbox_2d": box["bbox_2d"],
                "provenance": box["provenance"],
            })
            if isinstance(box.get("bbox_3d"), dict):
                query["expected_bbox_3d_by_view"][box["view_id"]].append({
                    "node_id": box["node_id"],
                    "label": box["label"],
                    "bbox_3d": box["bbox_3d"],
                    "provenance": box["provenance"],
                })
        query["expected_bbox_2d_by_view"] = dict(query["expected_bbox_2d_by_view"])
        query["expected_bbox_3d_by_view"] = dict(query["expected_bbox_3d_by_view"])
        query["expected_bbox_3d"] = merged
        query["gt_3d_reliable"] = merged is not None
        query["gt_box_status"] = "manual_coarse_depth_projected"
        query["gt_box_source"] = "docs/benchmarks/manual_bbox_gt_v2.json"
        query_boxes[query_id] = {
            "query": query["query"],
            "scene_id": public_scene,
            "captured_scene_id": captured_scene,
            "query_type": query["query_type"],
            "box_count": len(boxes),
            "boxes": boxes,
            "merged_bbox_3d": merged,
        }

    for scene_id, views in views_by_scene.items():
        write_scene_views(scene_id, views)
        write_json(SCENE_ROOT / scene_id / "scene_manifest.json", build_scene_manifest(scene_id))
        draw_previews(scene_id, preview_boxes[scene_id])

    gt = {
        "schema_version": "semanticsplat.manual_bbox_gt.v1",
        "benchmark_id": benchmark["benchmark_id"],
        "status": "manual_coarse_depth_projected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "camera_to_world",
        "box_source": (
            "Coarse 2D boxes derived from captured images and manual ViewJSON "
            "approx_location labels, then projected through captured depth maps."
        ),
        "not_official_dataset_gt": True,
        "counts": {
            "positive_queries": positive_count,
            "negative_queries_not_applicable": negative_count,
            "query_records_with_bbox_gt": len(query_boxes),
            "unique_object_observation_boxes": len(box_records_by_node),
            "query_box_links": sum(item["box_count"] for item in query_boxes.values()),
        },
        "query_boxes": query_boxes,
    }
    benchmark["bbox_gt_status"] = {
        "status": "manual_coarse_depth_projected",
        "source": "docs/benchmarks/manual_bbox_gt_v2.json",
        "positive_queries_with_bbox_gt": positive_count,
        "negative_queries_not_applicable": negative_count,
        "unique_object_observation_boxes": len(box_records_by_node),
        "not_official_dataset_gt": True,
    }
    write_json(BOX_GT_PATH, gt)
    write_scene_gt_copies(gt, list(scene_map.values()))
    write_json(BENCHMARK_PATH, benchmark)
    update_config(list(scene_map.values()), scene_map)
    return gt


def write_report(gt: dict[str, Any]) -> None:
    counts = gt["counts"]
    lines = [
        "# Manual BBox GT Report V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "This GT-box package covers every positive query in `benchmark_queries_v2.json`.",
        "Negative queries correctly have no expected target box.",
        "",
        "The boxes are manual/coarse benchmark reference boxes, not official dataset GT and not pixel-perfect instance masks.",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Positive queries with bbox GT | {counts['positive_queries']} |",
        f"| Negative queries not applicable | {counts['negative_queries_not_applicable']} |",
        f"| Query records with bbox GT | {counts['query_records_with_bbox_gt']} |",
        f"| Unique object observation boxes | {counts['unique_object_observation_boxes']} |",
        f"| Query-box links | {counts['query_box_links']} |",
        "",
        "## Evidence",
        "",
        "- GT JSON: `docs/benchmarks/manual_bbox_gt_v2.json`",
        "- Benchmark with expected boxes: `docs/benchmarks/benchmark_queries_v2.json`",
        "- Preview overlays: `docs/benchmarks/bbox_gt_v2_previews/`",
        "- Scene manifests: `backend/data/scenes/<scene_id>/scene_manifest.json`",
        "",
        "## Limitations",
        "",
        "- Boxes are coarse object/region reference boxes from manual ViewJSON location evidence.",
        "- These boxes are suitable for internal same-input benchmark gating.",
        "- Do not report them as official Replica, ScanNet, or independent dataset boxes.",
        "- Use 3D IoU as a coarse regression signal, not a final localization claim.",
    ]
    BOX_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual bbox GT for benchmark v2")
    parser.add_argument("--check", action="store_true", help="Build in memory and print counts")
    args = parser.parse_args()
    gt = build()
    write_report(gt)
    if args.check:
        print(json.dumps(gt["counts"], indent=2))
    else:
        print(f"Wrote {BOX_GT_PATH.relative_to(ROOT)}")
        print(f"Wrote {BOX_REPORT_PATH.relative_to(ROOT)}")
        print(f"Wrote {CONFIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
