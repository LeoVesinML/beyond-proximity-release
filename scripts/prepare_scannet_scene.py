#!/usr/bin/env python3
"""Convert official ScanNet scenes into the SemanticSplat backend contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scene_tree_from_viewjson import build_scene_tree  # noqa: E402
from scripts.import_replica_semantic_mesh import read_ply_header  # noqa: E402
from scripts.scannet_sensor_data import SensorDataIndex  # noqa: E402
from scripts.validate_scene_geometry import validate_scene  # noqa: E402
from scripts.validate_semantic_index import validate_semantic_index  # noqa: E402


SCHEMA_VERSION = "semanticsplat.scannet_official_import.v1"
BBQ_SCENES = (
    "scene0011_00",
    "scene0030_00",
    "scene0046_00",
    "scene0086_00",
    "scene0222_00",
    "scene0378_00",
    "scene0389_00",
    "scene0435_00",
)
PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def prepare_empty_directory(path: Path, *, overwrite: bool, expected_prefix: str) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty; use --overwrite: {path}")
        if not path.name.startswith(expected_prefix):
            raise ValueError(f"Refusing to replace unexpected output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_axis_alignment(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip().startswith("axisAlignment"):
            continue
        values = [float(value) for value in line.split("=", 1)[1].split()]
        if len(values) != 16:
            raise ValueError(f"axisAlignment must contain 16 values: {path}")
        matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(matrix).all():
            raise ValueError(f"axisAlignment contains non-finite values: {path}")
        return matrix
    raise ValueError(f"Missing axisAlignment in {path}")


def read_vertex_properties(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    header = read_ply_header(path)
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError(f"ScanNet PLY must store vertices first: {path}")
    vertex = header.elements[0]
    if any(prop.is_list for prop in vertex.properties):
        raise ValueError(f"ScanNet vertex properties must be scalar: {path}")
    dtype_fields = []
    for prop in vertex.properties:
        numpy_type = PLY_NUMPY_TYPES.get(prop.data_type)
        if numpy_type is None:
            raise ValueError(f"Unsupported PLY type {prop.data_type!r}: {path}")
        dtype_fields.append((prop.name, numpy_type))
    available = {prop.name for prop in vertex.properties}
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"Missing PLY vertex properties {missing}: {path}")
    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        records = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=vertex.count)
    if records.size != vertex.count:
        raise EOFError(f"Expected {vertex.count} vertices, read {records.size}: {path}")
    return {name: np.asarray(records[name]) for name in names}


def load_label_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_raw: dict[str, dict[str, Any]] = {}
    by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw = str(row.get("raw_category") or "").strip()
            raw_id = str(row.get("id") or "").strip()
            if not raw or not raw_id:
                continue
            item = {
                "id": int(raw_id),
                "raw_category": raw,
                "category": str(row.get("category") or raw).strip(),
                "nyu40id": int(row["nyu40id"]) if str(row.get("nyu40id") or "").strip() else None,
                "nyu40class": str(row.get("nyu40class") or "").strip() or None,
            }
            by_raw[raw.lower()] = item
            by_id[item["id"]] = item
    if not by_raw:
        raise ValueError(f"ScanNet label map is empty: {path}")
    return by_raw, by_id


def aligned_points(points: np.ndarray, axis_alignment: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points.astype(np.float64), np.ones(points.shape[0])))
    return (homogeneous @ axis_alignment.T)[:, :3]


def object_box(
    group: dict[str, Any],
    points: np.ndarray,
    label_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not points.size:
        return None
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    size = upper - lower
    if not np.isfinite(size).all() or np.any(size <= 1e-6):
        return None
    center = (lower + upper) / 2.0
    raw_label = str(group.get("label") or f"object_{group.get('objectId')}").strip()
    canonical = str((label_info or {}).get("nyu40class") or raw_label)
    return {
        "object_id": str(group.get("objectId")),
        "instance_group_id": int(group.get("id", group.get("objectId", -1))),
        "label": raw_label,
        "canonical_label": canonical,
        "raw_category_id": (label_info or {}).get("id"),
        "class_id": (label_info or {}).get("nyu40id"),
        "center": [round(float(value), 6) for value in center],
        "size": [round(float(value), 6) for value in size],
        "min": [round(float(value), 6) for value in lower],
        "max": [round(float(value), 6) for value in upper],
        "point_count": int(points.shape[0]),
        "coordinate_frame": "scannet_axis_aligned_mesh",
        "bbox_source": "official aggregation segments over axis-aligned _vh_clean_2.ply vertices",
    }


def convert_ground_truth(
    raw_scene: Path,
    label_map_path: Path,
    *,
    visibility_sample_count: int = 96,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    scene_name = raw_scene.name
    mesh_path = raw_scene / f"{scene_name}_vh_clean_2.ply"
    labels_path = raw_scene / f"{scene_name}_vh_clean_2.labels.ply"
    segments_path = raw_scene / f"{scene_name}_vh_clean_2.0.010000.segs.json"
    aggregation_path = raw_scene / f"{scene_name}.aggregation.json"
    metadata_path = raw_scene / f"{scene_name}.txt"
    required = [mesh_path, labels_path, segments_path, aggregation_path, metadata_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ScanNet GT inputs: " + ", ".join(missing))

    axis_alignment = parse_axis_alignment(metadata_path)
    xyz = read_vertex_properties(mesh_path, ("x", "y", "z"))
    vertices = np.column_stack((xyz["x"], xyz["y"], xyz["z"]))
    vertices = aligned_points(vertices, axis_alignment)
    segment_data = read_json(segments_path)
    segment_indices = np.asarray(segment_data.get("segIndices"), dtype=np.int64)
    if segment_indices.size != vertices.shape[0]:
        raise ValueError(
            f"Segment index count {segment_indices.size} does not match vertex count {vertices.shape[0]}"
        )
    aggregation = read_json(aggregation_path)
    groups = aggregation.get("segGroups")
    if not isinstance(groups, list):
        raise ValueError(f"Aggregation has no segGroups list: {aggregation_path}")
    label_by_raw, label_by_id = load_label_map(label_map_path)

    boxes: list[dict[str, Any]] = []
    samples: dict[str, np.ndarray] = {}
    skipped: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: int(item.get("objectId", item.get("id", -1)))):
        if not isinstance(group, dict):
            continue
        object_id = str(group.get("objectId"))
        segment_ids = group.get("segments")
        if not isinstance(segment_ids, list) or not segment_ids:
            skipped.append({"object_id": object_id, "reason": "no segments"})
            continue
        mask = np.isin(segment_indices, np.asarray(segment_ids, dtype=np.int64))
        points = vertices[mask]
        raw_label = str(group.get("label") or "").strip().lower()
        box = object_box(group, points, label_by_raw.get(raw_label))
        if box is None:
            skipped.append({"object_id": object_id, "reason": "empty or degenerate geometry"})
            continue
        boxes.append(box)
        if points.shape[0] <= visibility_sample_count:
            samples[object_id] = points.copy()
        else:
            indices = np.linspace(0, points.shape[0] - 1, visibility_sample_count, dtype=np.int64)
            samples[object_id] = points[indices]

    semantic = read_vertex_properties(labels_path, ("label",))["label"].astype(np.int64)
    histogram = Counter(int(value) for value in semantic.tolist())
    classes = []
    for raw_id, count in sorted(histogram.items()):
        info = label_by_id.get(raw_id, {})
        classes.append(
            {
                "raw_category_id": raw_id,
                "raw_category": info.get("raw_category"),
                "nyu40id": info.get("nyu40id"),
                "nyu40class": info.get("nyu40class"),
                "vertex_count": count,
            }
        )
    segmentation_summary = {
        "schema_version": "semanticsplat.scannet_semantic_gt_summary.v1",
        "source": display_path(labels_path),
        "vertex_count": int(semantic.size),
        "class_count": len(histogram),
        "classes": classes,
        "metric_status": "GT ready; mAcc/mIoU/fmIoU require separate predicted per-vertex labels",
    }
    metadata = {
        "mesh_path": display_path(mesh_path),
        "labels_path": display_path(labels_path),
        "segments_path": display_path(segments_path),
        "aggregation_path": display_path(aggregation_path),
        "vertex_count": int(vertices.shape[0]),
        "instance_group_count": len(groups),
        "valid_box_count": len(boxes),
        "skipped_instances": skipped,
        "semantic_segmentation": segmentation_summary,
    }
    if not boxes:
        raise ValueError(f"No valid ScanNet boxes were converted from {raw_scene}")
    return boxes, samples, metadata, axis_alignment


def valid_pose(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (4, 4)
        and np.isfinite(matrix).all()
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4)
        and abs(float(np.linalg.det(matrix[:3, :3]))) > 1e-6
    )


def visibility_score(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> float | None:
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return None
    camera = np.column_stack((points, np.ones(points.shape[0]))) @ world_to_camera.T
    camera = camera[:, :3]
    in_front = camera[:, 2] > 0.1
    if not np.any(in_front):
        return None
    camera = camera[in_front]
    z = camera[:, 2]
    u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    margin_x = width * 0.03
    margin_y = height * 0.03
    inside = (u >= -margin_x) & (u < width + margin_x) & (v >= -margin_y) & (v < height + margin_y)
    if not np.any(inside):
        return None
    visible_u = u[inside]
    visible_v = v[inside]
    centrality = 1.0 - min(
        1.0,
        float(np.mean(np.abs(visible_u - width / 2.0)) / max(width / 2.0, 1.0)),
    )
    return float(np.count_nonzero(inside)) + centrality


def frame_visibility(
    camera_to_world: np.ndarray,
    samples: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    visible: dict[str, float] = {}
    for object_id, points in samples.items():
        score = visibility_score(points, camera_to_world, intrinsic, width, height)
        if score is not None:
            visible[object_id] = score
    return visible


def select_coverage_frames(
    sensor: SensorDataIndex,
    axis_alignment: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    max_views: int,
    candidate_stride: int,
) -> tuple[list[int], dict[str, int], dict[int, dict[str, float]], list[str]]:
    if max_views < 1 or candidate_stride < 1:
        raise ValueError("max_views and candidate_stride must be positive")
    intrinsic = sensor.header.intrinsic_depth.astype(np.float64)
    width = sensor.header.depth_width
    height = sensor.header.depth_height

    valid_indices = []
    for frame in sensor.frames:
        raw_pose = frame.camera_to_world.astype(np.float64)
        if not valid_pose(raw_pose):
            continue
        aligned_pose = axis_alignment @ raw_pose
        if valid_pose(aligned_pose):
            valid_indices.append(frame.index)
    if not valid_indices:
        raise ValueError(f"No valid camera poses in {sensor.path}")
    coarse = valid_indices[::candidate_stride]
    if valid_indices[-1] not in coarse:
        coarse.append(valid_indices[-1])

    visibility: dict[int, dict[str, float]] = {}
    for frame_index in coarse:
        pose = axis_alignment @ sensor.frames[frame_index].camera_to_world.astype(np.float64)
        visibility[frame_index] = frame_visibility(pose, samples, intrinsic, width, height)

    uncovered = set(samples)
    selected: list[int] = []

    def add_best(candidates: list[int]) -> bool:
        options = []
        for frame_index in candidates:
            if frame_index in selected:
                continue
            visible = visibility[frame_index]
            newly_covered = uncovered & set(visible)
            if not newly_covered:
                continue
            options.append(
                (
                    len(newly_covered),
                    sum(visible[object_id] for object_id in newly_covered),
                    len(visible),
                    -frame_index,
                    frame_index,
                )
            )
        if not options:
            return False
        frame_index = max(options)[-1]
        selected.append(frame_index)
        uncovered.difference_update(visibility[frame_index])
        return True

    while uncovered and len(selected) < max_views and add_best(coarse):
        pass

    if uncovered and len(selected) < max_views and candidate_stride > 1:
        fine = [index for index in valid_indices if index not in visibility]
        for frame_index in fine:
            pose = axis_alignment @ sensor.frames[frame_index].camera_to_world.astype(np.float64)
            visibility[frame_index] = frame_visibility(pose, samples, intrinsic, width, height)
        all_candidates = sorted(visibility)
        while uncovered and len(selected) < max_views and add_best(all_candidates):
            pass

    assignment: dict[str, int] = {}
    for object_id in sorted(samples, key=lambda value: int(value)):
        candidates = [
            (visibility[frame_index].get(object_id, -math.inf), -frame_index, frame_index)
            for frame_index in selected
            if object_id in visibility[frame_index]
        ]
        if candidates:
            assignment[object_id] = max(candidates)[-1]
    selected = sorted(set(assignment.values()))
    uncovered = sorted(set(samples) - set(assignment), key=lambda value: int(value))
    return selected, assignment, visibility, uncovered


def extract_selected_frames(
    sensor: SensorDataIndex,
    axis_alignment: np.ndarray,
    frame_indices: list[int],
    extracted_scene: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    prepare_empty_directory(extracted_scene, overwrite=overwrite, expected_prefix="scene")
    for folder in ("color", "depth", "pose", "intrinsic"):
        (extracted_scene / folder).mkdir(parents=True, exist_ok=True)
    np.savetxt(extracted_scene / "intrinsic" / "intrinsic_depth.txt", sensor.header.intrinsic_depth)
    np.savetxt(extracted_scene / "intrinsic" / "intrinsic_color.txt", sensor.header.intrinsic_color)
    np.savetxt(extracted_scene / "intrinsic" / "extrinsic_depth.txt", sensor.header.extrinsic_depth)
    np.savetxt(extracted_scene / "intrinsic" / "extrinsic_color.txt", sensor.header.extrinsic_color)

    records: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        decoded = sensor.decode(frame_index, resize_color_to_depth=True)
        pose = axis_alignment @ decoded.camera_to_world.astype(np.float64)
        stem = f"{frame_index:06d}"
        color_path = extracted_scene / "color" / f"{stem}.jpg"
        depth_path = extracted_scene / "depth" / f"{stem}.npy"
        pose_path = extracted_scene / "pose" / f"{stem}.txt"
        decoded.color.save(color_path, format="JPEG", quality=92)
        np.save(depth_path, decoded.depth_meters)
        np.savetxt(pose_path, pose)
        finite = decoded.depth_meters[np.isfinite(decoded.depth_meters)]
        records.append(
            {
                "original_frame_index": frame_index,
                "color": color_path.relative_to(extracted_scene).as_posix(),
                "depth": depth_path.relative_to(extracted_scene).as_posix(),
                "pose": pose_path.relative_to(extracted_scene).as_posix(),
                "timestamp_color": decoded.timestamp_color,
                "timestamp_depth": decoded.timestamp_depth,
                "depth_min": round(float(finite.min()), 6) if finite.size else None,
                "depth_max": round(float(finite.max()), 6) if finite.size else None,
                "depth_zero_fraction": round(float(np.count_nonzero(finite <= 0) / finite.size), 6)
                if finite.size
                else None,
            }
        )
    return records


def view_object(box: dict[str, Any], original_frame_index: int) -> dict[str, Any]:
    aliases = [box["label"]]
    if box.get("canonical_label") and box["canonical_label"] not in aliases:
        aliases.append(str(box["canonical_label"]))
    attributes = [
        "official_scannet_instance",
        f"object_id:{box['object_id']}",
        f"representative_frame:{original_frame_index}",
        *[f"label_alias:{alias}" for alias in aliases],
    ]
    if box.get("class_id") is not None:
        attributes.append(f"nyu40id:{box['class_id']}")
    return {
        "label": box["label"],
        "attributes": attributes,
        "approx_location": f"official ScanNet instance assigned to representative frame {original_frame_index}",
        "confidence": 1.0,
        "bbox_2d": None,
        "bbox_3d": {
            "center": box["center"],
            "size": box["size"],
            "label": box["label"],
            "object_id": box["object_id"],
            "coordinate_frame": box["coordinate_frame"],
        },
        "source": "official_gt",
    }


def import_scene(
    raw_scene: Path,
    extracted_scene: Path,
    output_scene: Path,
    label_map_path: Path,
    *,
    backend_scene_id: str,
    max_views: int,
    candidate_stride: int,
    overwrite: bool,
) -> dict[str, Any]:
    conversion_started = time.perf_counter()
    raw_scene = raw_scene.resolve()
    scene_name = raw_scene.name
    sens_path = raw_scene / f"{scene_name}.sens"
    if not sens_path.is_file():
        raise FileNotFoundError(f"Missing ScanNet sensor stream: {sens_path}")
    boxes, samples, gt_metadata, axis_alignment = convert_ground_truth(raw_scene, label_map_path)
    sensor = SensorDataIndex.scan(sens_path)
    selected, assignment, visibility, uncovered = select_coverage_frames(
        sensor,
        axis_alignment,
        samples,
        max_views=max_views,
        candidate_stride=candidate_stride,
    )
    if not selected:
        raise ValueError(f"Coverage selection produced no usable frames for {scene_name}")
    extracted_records = extract_selected_frames(
        sensor,
        axis_alignment,
        selected,
        extracted_scene,
        overwrite=overwrite,
    )
    write_json(
        extracted_scene / "selection_manifest.json",
        {
            "schema_version": "semanticsplat.scannet_extraction.v1",
            "source_sens": display_path(sens_path),
            "sensor": sensor.metadata(),
            "axis_alignment": axis_alignment.tolist(),
            "selection_strategy": "official-GT frustum coverage greedy",
            "candidate_stride": candidate_stride,
            "max_views": max_views,
            "selected_frame_count": len(selected),
            "selected_original_frame_indices": selected,
            "official_object_count": len(boxes),
            "indexed_object_count": len(assignment),
            "coverage_ratio": round(len(assignment) / len(boxes), 6),
            "uncovered_object_ids": uncovered,
            "frames": extracted_records,
        },
    )

    prepare_empty_directory(output_scene, overwrite=overwrite, expected_prefix="scannet_")
    boxes_by_id = {box["object_id"]: box for box in boxes}
    assigned_by_frame: dict[int, list[dict[str, Any]]] = {frame_index: [] for frame_index in selected}
    for object_id, frame_index in assignment.items():
        assigned_by_frame[frame_index].append(boxes_by_id[object_id])
    selected = [frame_index for frame_index in selected if assigned_by_frame[frame_index]]
    extracted_by_index = {item["original_frame_index"]: item for item in extracted_records}

    frames = []
    view_assignment: dict[str, str] = {}
    for view_number, frame_index in enumerate(selected, 1):
        view_id = f"v{view_number:03d}"
        source = extracted_by_index[frame_index]
        image_path = output_scene / "images" / f"{view_id}.jpg"
        depth_path = output_scene / "depths" / f"{view_id}_depth.npy"
        link_or_copy(extracted_scene / source["color"], image_path)
        link_or_copy(extracted_scene / source["depth"], depth_path)
        pose = np.loadtxt(extracted_scene / source["pose"]).reshape(4, 4)
        frames.append(
            {
                "view_id": view_id,
                "file_path": image_path.relative_to(output_scene).as_posix(),
                "depth_file_path": depth_path.relative_to(output_scene).as_posix(),
                "transform_matrix": pose.tolist(),
                "source_frame_index": frame_index,
                "source_timestamp_color": source["timestamp_color"],
                "source_timestamp_depth": source["timestamp_depth"],
            }
        )
        view_boxes = sorted(assigned_by_frame[frame_index], key=lambda item: int(item["object_id"]))
        for box in view_boxes:
            view_assignment[box["object_id"]] = view_id
        label_counts = Counter(box["label"] for box in view_boxes)
        summary = ", ".join(f"{label} x{count}" for label, count in sorted(label_counts.items()))
        write_json(
            output_scene / "views" / f"{view_id}.json",
            {
                "schema_version": "semanticsplat.captured_view_json.v1",
                "view_id": view_id,
                "scene_id": backend_scene_id,
                "image_path": image_path.relative_to(output_scene).as_posix(),
                "depth_path": depth_path.relative_to(output_scene).as_posix(),
                "pose_ref": f"transforms.json#/frames/{view_number - 1}/transform_matrix",
                "intrinsics_ref": "transforms.json",
                "semantic_index_mode": "official_gt",
                "summary": f"Official ScanNet representative-view object index: {summary}",
                "visible_regions": [
                    {
                        "label": "official ScanNet scene",
                        "approx_location": scene_name,
                        "source": "official_gt",
                    }
                ],
                "visible_objects": [view_object(box, frame_index) for box in view_boxes],
                "landmarks": [],
                "free_text_notes": (
                    "Objects are official ScanNet GT instances assigned once to a coverage-selected "
                    "representative RGB-D frame using frustum geometry."
                ),
                "warnings": [
                    "Official ScanNet GT is used as the semantic-map input for retrieval evaluation.",
                    "Representative-view assignment is derived from camera frusta and is not an official 2D visibility annotation.",
                    "Do not report this track as semantic perception or segmentation prediction accuracy.",
                ],
            },
        )

    intrinsic = sensor.header.intrinsic_depth.astype(float)
    write_json(
        output_scene / "transforms.json",
        {
            "camera_model": "OPENCV",
            "w": sensor.header.depth_width,
            "h": sensor.header.depth_height,
            "fl_x": float(intrinsic[0, 0]),
            "fl_y": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "frames": frames,
            "source_note": "Color images are resized to depth resolution; depth-camera intrinsics are used.",
        },
    )
    write_json(
        output_scene / "ground_truth" / "object_boxes_3d.json",
        {
            "schema_version": "semanticsplat.object_boxes_3d_gt.v1",
            "scene_id": backend_scene_id,
            "source_dataset": "ScanNet",
            "source_scene_id": scene_name,
            "coordinate_frame": "scannet_axis_aligned_mesh",
            "object_count": len(boxes),
            "indexed_object_count": len(assignment),
            "objects": boxes,
            "skipped_objects": gt_metadata["skipped_instances"],
        },
    )
    write_json(
        output_scene / "ground_truth" / "semantic_segmentation_summary.json",
        gt_metadata["semantic_segmentation"],
    )
    write_json(
        output_scene / "ground_truth" / "object_view_assignments.json",
        {
            "schema_version": "semanticsplat.scannet_object_view_assignment.v1",
            "method": "frustum-coverage representative view",
            "object_count": len(view_assignment),
            "object_to_view": view_assignment,
            "uncovered_object_ids": uncovered,
        },
    )
    write_json(
        output_scene / "source_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "scene_id": backend_scene_id,
            "source_dataset": "ScanNet",
            "source_scene_id": scene_name,
            "raw_scene_path": display_path(raw_scene),
            "extracted_scene_path": display_path(extracted_scene),
            "sensor": sensor.metadata(),
            "axis_alignment": axis_alignment.tolist(),
            "gt": {key: value for key, value in gt_metadata.items() if key != "semantic_segmentation"},
            "selected_frame_count": len(frames),
            "coverage_ratio": round(len(assignment) / len(boxes), 6),
        },
    )
    write_json(
        output_scene / "scene_manifest.json",
        {
            "schema_version": "semanticsplat.scene_manifest.v1",
            "scene_id": backend_scene_id,
            "dataset": "ScanNet",
            "source_dataset": "ScanNet",
            "source_scene_id": scene_name,
            "source_description": f"Official ScanNet v2 scene imported from {display_path(raw_scene)}",
            "paths": {
                "rgb": "images",
                "depth": "depths",
                "poses": "transforms.json",
                "intrinsics": "transforms.json",
                "semantic_index": "views",
                "gt_object_boxes_3d": "ground_truth/object_boxes_3d.json",
                "gt_semantic_summary": "ground_truth/semantic_segmentation_summary.json",
            },
            "frame_count": len(frames),
            "coordinate_frame": "scannet_axis_aligned_mesh",
            "ground_truth": {
                "zone_labels_available": False,
                "object_labels_available": True,
                "bbox_3d_available": True,
                "semantic_vertex_labels_available": True,
                "source": "official ScanNet aggregation, segments, labeled mesh, and label map",
            },
            "evaluation_eligibility": {
                "semantic_evaluation_allowed": True,
                "three_d_localization_allowed": True,
                "rgbd_visual_evaluation_allowed": True,
                "segmentation_metrics_allowed": False,
                "reason": (
                    "Official GT object IDs/boxes support retrieval and box-IoU scoring; semantic "
                    "segmentation metrics need independent predicted per-vertex labels."
                ),
            },
            "warnings": [
                "Official GT is the semantic-map input; this is an oracle-map retrieval track, not semantic perception accuracy.",
                "RGB is resized to the depth resolution for the backend frame contract.",
            ],
        },
    )

    tree_manifest = build_scene_tree(output_scene, output_scene / "views", output_scene / "tree")
    semantic_validation = validate_semantic_index(output_scene)
    geometry_validation = validate_scene(output_scene, scene_id=backend_scene_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": backend_scene_id,
        "source_scene_id": scene_name,
        "raw_frame_count": sensor.header.frame_count,
        "selected_frame_count": len(frames),
        "official_object_count": len(boxes),
        "indexed_object_count": len(assignment),
        "object_coverage_ratio": round(len(assignment) / len(boxes), 6),
        "uncovered_object_count": len(uncovered),
        "semantic_vertex_count": gt_metadata["semantic_segmentation"]["vertex_count"],
        "semantic_class_count": gt_metadata["semantic_segmentation"]["class_count"],
        "conversion_runtime_seconds": round(time.perf_counter() - conversion_started, 6),
        "tree": {
            "node_count": tree_manifest["node_count"],
            "semantic_item_count": tree_manifest["semantic_item_count"],
            "complete": tree_manifest["complete"],
        },
        "semantic_validation": semantic_validation,
        "geometry_validation": geometry_validation,
    }


def backend_scene_id(source_scene_id: str) -> str:
    if not source_scene_id.startswith("scene"):
        raise ValueError(f"Invalid ScanNet scene ID: {source_scene_id}")
    return "scannet_" + source_scene_id.removeprefix("scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-bbq-scenes", action="store_true")
    selection.add_argument("--scene", help="Official scene folder name, e.g. scene0011_00")
    parser.add_argument("--raw-root", type=Path, default=Path("data/scannet/scans"))
    parser.add_argument("--extracted-root", type=Path, default=Path("data/scannet/extracted"))
    parser.add_argument("--scene-root", type=Path, default=Path("backend/data/scenes"))
    parser.add_argument("--label-map", type=Path, default=Path("data/scannet/tasks/scannetv2-labels.combined.tsv"))
    parser.add_argument("--max-views", type=int, default=24)
    parser.add_argument("--candidate-stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/datasets/scannet_official_import_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_names = BBQ_SCENES if args.all_bbq_scenes else (str(args.scene),)
    reports = []
    for scene_name in scene_names:
        result = import_scene(
            args.raw_root / scene_name,
            args.extracted_root / scene_name,
            args.scene_root / backend_scene_id(scene_name),
            args.label_map,
            backend_scene_id=backend_scene_id(scene_name),
            max_views=args.max_views,
            candidate_stride=args.candidate_stride,
            overwrite=args.overwrite,
        )
        reports.append(result)
        validation_path = ROOT / "docs" / "validation" / "public_datasets" / f"{result['scene_id']}.json"
        compact_validation = {
            key: value
            for key, value in result.items()
            if key not in {"geometry_validation"}
        }
        geometry = result["geometry_validation"]
        compact_validation["geometry_validation"] = {
            key: value for key, value in geometry.items() if key != "per_frame"
        }
        write_json(validation_path, compact_validation)
        print(
            f"Imported {scene_name}: views={result['selected_frame_count']} "
            f"objects={result['indexed_object_count']}/{result['official_object_count']} "
            f"semantic={result['semantic_validation']['semantic_eval_allowed']} "
            f"geometry={result['geometry_validation']['three_d_localization_allowed']}"
        )
    summary = {
        "schema_version": "semanticsplat.scannet_official_import_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(reports),
        "official_object_count": sum(report["official_object_count"] for report in reports),
        "indexed_object_count": sum(report["indexed_object_count"] for report in reports),
        "selected_frame_count": sum(report["selected_frame_count"] for report in reports),
        "semantic_vertex_count": sum(report["semantic_vertex_count"] for report in reports),
        "conversion_runtime_seconds": round(
            sum(report["conversion_runtime_seconds"] for report in reports), 6
        ),
        "all_semantic_valid": all(report["semantic_validation"]["semantic_eval_allowed"] for report in reports),
        "all_geometry_valid": all(report["geometry_validation"]["three_d_localization_allowed"] for report in reports),
        "scenes": [
            {
                "scene_id": report["scene_id"],
                "source_scene_id": report["source_scene_id"],
                "raw_frame_count": report["raw_frame_count"],
                "selected_frame_count": report["selected_frame_count"],
                "official_object_count": report["official_object_count"],
                "indexed_object_count": report["indexed_object_count"],
                "object_coverage_ratio": report["object_coverage_ratio"],
                "semantic_vertex_count": report["semantic_vertex_count"],
                "semantic_class_count": report["semantic_class_count"],
                "conversion_runtime_seconds": report["conversion_runtime_seconds"],
            }
            for report in reports
        ],
    }
    write_json(args.report, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
