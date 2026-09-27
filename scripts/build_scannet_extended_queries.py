#!/usr/bin/env python3
"""Extend the official ScanNet pilot with GT-backed functional and negative queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCENE_ROOT = ROOT / "backend" / "data" / "scenes"

FUNCTIONAL_SPECS = {
    "scannet_0011_00": (
        "Find a place where someone can sit for the meeting.",
        {"chair", "armchair"},
    ),
    "scannet_0030_00": (
        "Find a seat suitable for working at a desk.",
        {"chair", "office chair", "armchair"},
    ),
    "scannet_0046_00": ("Find a place to sleep.", {"bed"}),
    "scannet_0086_00": ("Find a place to wash your hands.", {"sink"}),
    "scannet_0222_00": ("Find a place to sleep.", {"bed"}),
    "scannet_0378_00": (
        "Find a seat at a workstation.",
        {"chair", "office chair"},
    ),
    "scannet_0389_00": ("Find a place to sit.", {"chair"}),
    "scannet_0435_00": ("Find a place to sleep.", {"bed"}),
}

NEGATIVE_SPECS = {
    "scannet_0011_00": "bathtub",
    "scannet_0030_00": "bed",
    "scannet_0046_00": "refrigerator",
    "scannet_0086_00": "bed",
    "scannet_0222_00": "bathtub",
    "scannet_0378_00": "bed",
    "scannet_0389_00": "bathtub",
    "scannet_0435_00": "refrigerator",
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def scene_objects(scene_id: str) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for path in sorted((SCENE_ROOT / scene_id / "views").glob("*.json")):
        view = read_json(path)
        view_id = str(view.get("view_id") or path.stem)
        for item in view.get("visible_objects", []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_3d")
            object_id = (
                item.get("object_id")
                or (bbox.get("object_id") if isinstance(bbox, dict) else None)
            )
            if object_id is None or not isinstance(bbox, dict):
                continue
            key = str(object_id)
            record = objects.setdefault(
                key,
                {
                    "object_id": key,
                    "label": str(item.get("label") or ""),
                    "bbox_3d": bbox,
                    "view_ids": [],
                },
            )
            if view_id not in record["view_ids"]:
                record["view_ids"].append(view_id)
    return objects


def functional_query(
    scene_id: str,
    query: str,
    labels: set[str],
    objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        record
        for record in objects.values()
        if record["label"].strip().casefold() in labels
    ]
    if not matches:
        raise ValueError(f"No functional GT objects for {scene_id}: {sorted(labels)}")
    matches.sort(key=lambda item: int(item["object_id"]))
    return {
        "query_id": f"{scene_id}_functional_gt_001",
        "scene_id": scene_id,
        "dataset_scene_id": scene_id.removeprefix("scannet_"),
        "source_dataset": "ScanNet-derived official GT",
        "source_annotation_id": "semanticsplat_functional_extension_v1",
        "query": query,
        "query_type": "functional",
        "expected_output_type": "object_3d_bbox",
        "expected_zone_ids": [],
        "expected_node_ids": [],
        "expected_view_ids": sorted(
            {
                view_id
                for match in matches
                for view_id in match["view_ids"]
            }
        ),
        "expected_object_labels": sorted(
            {str(match["label"]) for match in matches}
        ),
        "expected_object_id": matches[0]["object_id"],
        "expected_object_ids": [match["object_id"] for match in matches],
        "expected_anchor_object_ids": [],
        "expected_bbox_3d": matches[0]["bbox_3d"],
        "expected_bboxes_3d": [match["bbox_3d"] for match in matches],
        "verification_status": "verified_official_scannet_gt_functional_mapping",
        "verification_source": (
            "Official ScanNet instance labels and 3D boxes; functional mapping "
            "is declared by this benchmark protocol."
        ),
        "gt_3d_reliable": True,
        "acceptance_rule": (
            "Selected object ID must be one of the declared official ScanNet "
            "objects satisfying the functional category."
        ),
        "metrics_target": ["Recall@1", "Recall@3", "Recall@5", "MRR", "Acc@0.25"],
        "failure_labels": ["wrong object", "invalid/unavailable prediction", "bad bbox"],
    }


def negative_query(
    scene_id: str,
    absent_label: str,
    objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    present = {
        str(record["label"]).strip().casefold() for record in objects.values()
    }
    if absent_label.casefold() in present:
        raise ValueError(f"Negative label {absent_label!r} exists in {scene_id}")
    return {
        "query_id": f"{scene_id}_negative_gt_001",
        "scene_id": scene_id,
        "dataset_scene_id": scene_id.removeprefix("scannet_"),
        "source_dataset": "ScanNet-derived official GT",
        "source_annotation_id": "semanticsplat_negative_extension_v1",
        "query": f"Find the {absent_label}.",
        "query_type": "negative",
        "expected_output_type": "not_found",
        "expected_zone_ids": [],
        "expected_node_ids": [],
        "expected_view_ids": [],
        "expected_object_labels": [],
        "expected_object_id": None,
        "expected_object_ids": [],
        "expected_anchor_object_ids": [],
        "expected_bbox_3d": None,
        "expected_bboxes_3d": [],
        "verification_status": "verified_official_scannet_gt_absence",
        "negative_gt_verified": True,
        "verification_source": (
            "Absence checked against every official instance label imported "
            "for this ScanNet scene."
        ),
        "gt_3d_reliable": True,
        "acceptance_rule": "found=false",
        "metrics_target": ["verified_negative_accuracy"],
        "failure_labels": ["false positive", "invalid/unavailable prediction"],
    }


def build(base_path: Path) -> dict[str, Any]:
    base = read_json(base_path)
    base_queries = [
        query for query in base.get("queries", []) if isinstance(query, dict)
    ]
    extension: list[dict[str, Any]] = []
    for scene_id in FUNCTIONAL_SPECS:
        objects = scene_objects(scene_id)
        query, labels = FUNCTIONAL_SPECS[scene_id]
        extension.append(functional_query(scene_id, query, labels, objects))
        extension.append(
            negative_query(scene_id, NEGATIVE_SPECS[scene_id], objects)
        )
    return {
        **{key: value for key, value in base.items() if key != "queries"},
        "benchmark_id": "scannet_bbq_grounding_extended_v2",
        "benchmark_version": "official_referit3d_plus_functional_negative_v2",
        "query_count": len(base_queries) + len(extension),
        "base_query_count": len(base_queries),
        "extension_query_count": len(extension),
        "extension_scope": {
            "functional": 8,
            "verified_negative": 8,
            "oracle_map": True,
            "perception_accuracy": False,
        },
        "queries": [*base_queries, *extension],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "scannet_bbq_grounding_pilot_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "scannet_extended_grounding_v2.json",
    )
    args = parser.parse_args()
    benchmark = build(args.base.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {benchmark['query_count']} queries to {args.out.resolve()}")


if __name__ == "__main__":
    main()
