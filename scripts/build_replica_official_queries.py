#!/usr/bin/env python3
"""Build GT-backed official Replica object-grounding queries."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

SCENES = [
    ("replica_room0", "room0"),
    ("replica_room1", "room1"),
    ("replica_room2", "room2"),
    ("replica_office0", "office0"),
    ("replica_office1", "office1"),
    ("replica_office2", "office2"),
    ("replica_office3", "office3"),
    ("replica_office4", "office4"),
]

PREFERRED_LABELS = [
    "cabinet",
    "door",
    "vase",
    "rug",
    "bed",
    "picture",
    "table",
    "bottle",
    "shelf",
    "bowl",
    "tv-screen",
    "tablet",
    "clock",
    "camera",
    "desk",
    "monitor",
    "chair",
    "sofa",
    "bench",
    "stool",
    "lamp",
    "window",
    "book",
]

SEATING_LABELS = ["chair", "sofa", "bench", "stool", "bed"]
NEGATIVE_LABELS = ["fire hydrant", "traffic light", "bicycle", "microwave", "toilet", "bathtub"]


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def phrase(label: str) -> str:
    return label.replace("-", " ")


def bbox_for_query(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": obj["center"],
        "size": obj["size"],
        "label": obj["label"],
        "object_id": str(obj["object_id"]),
        "coordinate_frame": obj.get("coordinate_frame", "replica_habitat_mesh"),
    }


def grouped_objects(scene_id: str) -> dict[str, list[dict[str, Any]]]:
    gt_path = ROOT / "backend" / "data" / "scenes" / scene_id / "ground_truth" / "object_boxes_3d.json"
    gt = read_json(gt_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in gt.get("objects", []):
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label", "")).strip()
        if not label or label in {"undefined", "other-leaf", "floor", "wall", "ceiling"}:
            continue
        if not isinstance(obj.get("center"), list) or not isinstance(obj.get("size"), list):
            continue
        grouped[label].append(obj)
    return dict(grouped)


def choose_labels(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    labels = [label for label in PREFERRED_LABELS if label in grouped]
    labels.extend(label for label in sorted(grouped) if label not in labels)
    if len(labels) < 5:
        raise ValueError(f"Need at least five usable object labels, got {labels}")
    return labels


def query_record(
    *,
    scene_id: str,
    dataset_scene_id: str,
    query_id: str,
    query_text: str,
    query_type: str,
    label: str | None,
    objects: list[dict[str, Any]] | None,
    anchor_objects: list[dict[str, Any]] | None = None,
    negative: bool = False,
) -> dict[str, Any]:
    source = f"backend/data/scenes/{scene_id}/ground_truth/object_boxes_3d.json"
    if negative:
        return {
            "query_id": query_id,
            "scene_id": scene_id,
            "source_dataset": "Replica",
            "dataset_scene_id": dataset_scene_id,
            "query": query_text,
            "query_type": query_type,
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
            "acceptance_rule": "found=false because the queried object label is absent from official Replica GT for this scene",
            "verification_status": "verified_official_replica_gt_absence",
            "verification_source": [source],
            "metrics_target": ["not_found_correctness"],
            "failure_labels": ["wrong_object", "unavailable_prediction"],
            "gt_3d_reliable": False,
        }
    if not label or not objects:
        raise ValueError(f"Positive query {query_id} needs target objects")
    expected_boxes = [bbox_for_query(obj) for obj in objects]
    expected_ids = [str(obj["object_id"]) for obj in objects]
    anchor_ids = [str(obj["object_id"]) for obj in (anchor_objects or [])]
    return {
        "query_id": query_id,
        "scene_id": scene_id,
        "source_dataset": "Replica",
        "dataset_scene_id": dataset_scene_id,
        "query": query_text,
        "query_type": query_type,
        "expected_output_type": "object_3d_bbox",
        "expected_zone_ids": [],
        "expected_node_ids": [],
        "expected_view_ids": ["v001"],
        "expected_object_labels": [label],
        "expected_object_id": expected_ids[0],
        "expected_object_ids": expected_ids,
        "expected_anchor_object_ids": anchor_ids,
        "expected_bbox_3d": expected_boxes[0],
        "expected_bboxes_3d": expected_boxes,
        "acceptance_rule": "found=true and selected object ID is one acceptable official Replica GT object; Acc@k uses max IoU against expected_bboxes_3d",
        "verification_status": "verified_official_replica_gt",
        "verification_source": [source],
        "metrics_target": ["Recall@1", "ObjectIDHit", "Acc@0.1", "Acc@0.25", "Acc@0.5"],
        "failure_labels": ["wrong_object", "bad_bbox", "unavailable_prediction"],
        "gt_3d_reliable": True,
    }


def build_scene_queries(scene_id: str, dataset_scene_id: str) -> list[dict[str, Any]]:
    grouped = grouped_objects(scene_id)
    labels = choose_labels(grouped)
    seating = next((label for label in SEATING_LABELS if label in grouped), labels[0])
    absent = next(
        label for label in NEGATIVE_LABELS
        if all(part not in grouped for part in label.split())
    )
    rows = [
        ("simple_001", f"Find the {phrase(labels[0])}", "simple_object", labels[0]),
        ("simple_002", f"Find the {phrase(labels[1])}", "simple_object", labels[1]),
        ("attribute_001", f"Find the official Replica {phrase(labels[2])} object", "attribute", labels[2]),
        (
            "relational_001",
            f"Find the {phrase(labels[3])} near Replica object {grouped[labels[4]][0]['object_id']}",
            "relational",
            labels[3],
        ),
        (
            "multi_hop_001",
            f"First inspect the scene objects, then find the {phrase(labels[4])}",
            "multi_hop",
            labels[4],
        ),
        ("functional_001", f"Find a place to sit on the {phrase(seating)}", "functional", seating),
    ]
    queries = []
    for index, (suffix, query_text, query_type, label) in enumerate(rows, 1):
        anchor_objects = grouped[labels[4]] if query_type == "relational" else None
        queries.append(
            query_record(
                scene_id=scene_id,
                dataset_scene_id=dataset_scene_id,
                query_id=f"{scene_id}_{suffix}",
                query_text=query_text,
                query_type=query_type,
                label=label,
                objects=grouped[label],
                anchor_objects=anchor_objects,
            )
        )
    queries.append(
        query_record(
            scene_id=scene_id,
            dataset_scene_id=dataset_scene_id,
            query_id=f"{scene_id}_negative_001",
            query_text=f"Find the {absent}",
            query_type="negative",
            label=None,
            objects=None,
            negative=True,
        )
    )
    return queries


def build_benchmark() -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    for scene_id, dataset_scene_id in SCENES:
        queries.extend(build_scene_queries(scene_id, dataset_scene_id))
    counts = Counter(query["query_type"] for query in queries)
    return {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "replica_bbq_aligned_official_gt_pilot_v2",
        "status": "replica_official_gt_ready_scannet_separate_track_ready",
        "status_date": datetime.now(timezone.utc).date().isoformat(),
        "scene_scope": [scene_id for scene_id, _ in SCENES],
        "source_alignment": {
            "bbq_replica_scene_subset": [dataset_scene_id for _, dataset_scene_id in SCENES],
            "scanNet_status": "separate official eight-scene Nr3D/Sr3D+ grounding track ready",
        },
        "query_count": len(queries),
        "query_type_counts": dict(sorted(counts.items())),
        "ground_truth_scope": (
            "Replica queries are mapped to official Replica v1 object IDs and 3D boxes "
            "imported from habitat/info_semantic.json oriented_bbox fields. Repeated labels "
            "use multiple acceptable expected_object_ids and expected_bboxes_3d."
        ),
        "required_gt_fields_before_scoring": [
            "expected_object_ids",
            "expected_bboxes_3d",
            "expected_view_ids",
            "verification_status",
        ],
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "replica_scannet_pilot_queries.json",
    )
    args = parser.parse_args()
    benchmark = build_benchmark()
    write_json(args.out, benchmark)
    print(f"Wrote {len(benchmark['queries'])} queries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
