#!/usr/bin/env python3
"""Import official Replica semantic meshes as queryable GT object indexes."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scene_tree_from_viewjson import build_scene_tree  # noqa: E402
from scripts.validate_semantic_index import validate_semantic_index  # noqa: E402


SCHEMA_VERSION = "semanticsplat.replica_semantic_mesh_import.v1"
IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
PLY_TYPES = {
    "char": ("b", int),
    "int8": ("b", int),
    "uchar": ("B", int),
    "uint8": ("B", int),
    "short": ("h", int),
    "int16": ("h", int),
    "ushort": ("H", int),
    "uint16": ("H", int),
    "int": ("i", int),
    "int32": ("i", int),
    "uint": ("I", int),
    "uint32": ("I", int),
    "float": ("f", float),
    "float32": ("f", float),
    "double": ("d", float),
    "float64": ("d", float),
}
INSTANCE_FIELD_CANDIDATES = (
    "object_id",
    "instance_id",
    "semantic_id",
    "segment_id",
    "label",
    "category_id",
    "class_id",
)


@dataclass(frozen=True)
class PlyProperty:
    name: str
    data_type: str
    count_type: str | None = None
    item_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None and self.item_type is not None


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty]


@dataclass(frozen=True)
class PlyHeader:
    format_name: str
    elements: list[PlyElement]
    header_bytes: int


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized[:48] or "object"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_ply_header(path: Path) -> PlyHeader:
    data = bytearray()
    marker = -1
    with path.open("rb") as handle:
        while marker < 0:
            chunk = handle.read(16384)
            if not chunk:
                break
            data.extend(chunk)
            marker = data.find(b"end_header")
            if len(data) > 4 * 1024 * 1024:
                raise ValueError(f"PLY header is too large: {path}")
    if marker < 0:
        raise ValueError(f"PLY header has no end_header marker: {path}")
    line_end = data.find(b"\n", marker)
    header_bytes = line_end + 1 if line_end >= 0 else marker + len(b"end_header")
    text = bytes(data[:header_bytes]).decode("ascii", errors="strict")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    format_name = ""
    elements: list[PlyElement] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[PlyProperty] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            format_name = " ".join(parts[1:3])
        elif parts[0] == "element" and len(parts) == 3:
            if current_name is not None:
                elements.append(PlyElement(current_name, current_count, current_properties))
            current_name = parts[1]
            current_count = int(parts[2])
            current_properties = []
        elif parts[0] == "property" and current_name is not None:
            if len(parts) == 3:
                current_properties.append(PlyProperty(name=parts[2], data_type=parts[1]))
            elif len(parts) == 5 and parts[1] == "list":
                current_properties.append(
                    PlyProperty(
                        name=parts[4],
                        data_type="list",
                        count_type=parts[2],
                        item_type=parts[3],
                    )
                )
    if current_name is not None:
        elements.append(PlyElement(current_name, current_count, current_properties))
    if format_name not in {"ascii 1.0", "binary_little_endian 1.0"}:
        raise ValueError(f"Unsupported PLY format {format_name!r}; expected ascii or binary_little_endian")
    return PlyHeader(format_name=format_name, elements=elements, header_bytes=header_bytes)


def read_binary_scalar(handle: BinaryIO, data_type: str) -> int | float:
    spec = PLY_TYPES.get(data_type)
    if spec is None:
        raise ValueError(f"Unsupported PLY scalar type: {data_type}")
    fmt, caster = spec
    size = struct.calcsize("<" + fmt)
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected EOF while reading PLY binary scalar")
    return caster(struct.unpack("<" + fmt, data)[0])


def read_binary_record(handle: BinaryIO, properties: list[PlyProperty]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for prop in properties:
        if prop.is_list:
            assert prop.count_type is not None and prop.item_type is not None
            count = int(read_binary_scalar(handle, prop.count_type))
            row[prop.name] = [read_binary_scalar(handle, prop.item_type) for _ in range(count)]
        else:
            row[prop.name] = read_binary_scalar(handle, prop.data_type)
    return row


def read_ascii_record(tokens: list[str], properties: list[PlyProperty]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    cursor = 0
    for prop in properties:
        if prop.is_list:
            if cursor >= len(tokens):
                raise ValueError("Unexpected end of PLY ASCII record")
            count = int(tokens[cursor])
            cursor += 1
            row[prop.name] = [int(tokens[cursor + offset]) for offset in range(count)]
            cursor += count
        else:
            caster = PLY_TYPES.get(prop.data_type, ("", float))[1]
            row[prop.name] = caster(tokens[cursor])
            cursor += 1
    return row


def candidate_instance_field(properties: list[PlyProperty]) -> str | None:
    names = {prop.name for prop in properties}
    for candidate in INSTANCE_FIELD_CANDIDATES:
        if candidate in names:
            return candidate
    lowered = {name.lower(): name for name in names}
    for token in ("object", "instance", "semantic", "segment", "label", "class"):
        for lower, original in lowered.items():
            if token in lower:
                return original
    return None


def update_bounds(bounds: dict[int, list[list[float]]], object_id: int, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0).astype(float).tolist()
    maxs = points.max(axis=0).astype(float).tolist()
    if object_id not in bounds:
        bounds[object_id] = [mins, maxs]
        return
    current = bounds[object_id]
    current[0] = [min(current[0][axis], mins[axis]) for axis in range(3)]
    current[1] = [max(current[1][axis], maxs[axis]) for axis in range(3)]


def parse_ply_object_bounds(mesh_path: Path) -> tuple[dict[int, list[list[float]]], dict[str, Any]]:
    header = read_ply_header(mesh_path)
    vertex_element = next((element for element in header.elements if element.name == "vertex"), None)
    if vertex_element is None:
        raise ValueError("PLY has no vertex element")
    x_name = next((name for name in ("x", "X") if any(prop.name == name for prop in vertex_element.properties)), None)
    y_name = next((name for name in ("y", "Y") if any(prop.name == name for prop in vertex_element.properties)), None)
    z_name = next((name for name in ("z", "Z") if any(prop.name == name for prop in vertex_element.properties)), None)
    if not x_name or not y_name or not z_name:
        raise ValueError("PLY vertex element must expose x/y/z coordinates")

    vertex_instance_field = candidate_instance_field(vertex_element.properties)
    face_element = next((element for element in header.elements if element.name == "face"), None)
    face_instance_field = candidate_instance_field(face_element.properties) if face_element else None
    bounds: dict[int, list[list[float]]] = {}

    with mesh_path.open("rb") as handle:
        handle.seek(header.header_bytes)
        vertices = np.zeros((vertex_element.count, 3), dtype=np.float32)
        for element in header.elements:
            if header.format_name == "ascii 1.0":
                for row_index in range(element.count):
                    line = handle.readline().decode("ascii")
                    row = read_ascii_record(line.split(), element.properties)
                    if element.name == "vertex":
                        vertices[row_index] = [float(row[x_name]), float(row[y_name]), float(row[z_name])]
                        if vertex_instance_field and row.get(vertex_instance_field) is not None:
                            update_bounds(bounds, int(row[vertex_instance_field]), vertices[row_index : row_index + 1])
                    elif element.name == "face" and face_instance_field:
                        indices = row.get("vertex_indices") or row.get("vertex_index") or row.get("vertices")
                        if isinstance(indices, list) and row.get(face_instance_field) is not None:
                            update_bounds(bounds, int(row[face_instance_field]), vertices[np.asarray(indices, dtype=int)])
            else:
                for row_index in range(element.count):
                    row = read_binary_record(handle, element.properties)
                    if element.name == "vertex":
                        vertices[row_index] = [float(row[x_name]), float(row[y_name]), float(row[z_name])]
                        if vertex_instance_field and row.get(vertex_instance_field) is not None:
                            update_bounds(bounds, int(row[vertex_instance_field]), vertices[row_index : row_index + 1])
                    elif element.name == "face" and face_instance_field:
                        indices = row.get("vertex_indices") or row.get("vertex_index") or row.get("vertices")
                        if isinstance(indices, list) and row.get(face_instance_field) is not None:
                            update_bounds(bounds, int(row[face_instance_field]), vertices[np.asarray(indices, dtype=int)])

    metadata = {
        "format": header.format_name,
        "vertex_count": vertex_element.count,
        "face_count": face_element.count if face_element else 0,
        "vertex_instance_field": vertex_instance_field,
        "face_instance_field": face_instance_field,
        "object_count_with_bounds": len(bounds),
    }
    if not bounds:
        raise ValueError("No object/semantic/instance ID field was found in the Replica semantic mesh")
    return bounds, metadata


def ply_header_metadata(mesh_path: Path) -> dict[str, Any]:
    header = read_ply_header(mesh_path)
    vertex_element = next((element for element in header.elements if element.name == "vertex"), None)
    face_element = next((element for element in header.elements if element.name == "face"), None)
    return {
        "format": header.format_name,
        "vertex_count": vertex_element.count if vertex_element else 0,
        "face_count": face_element.count if face_element else 0,
        "object_count_with_bounds": None,
        "bbox_source": "info_semantic.json/oriented_bbox",
    }


def load_info_semantic(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return {}

    class_names: dict[int, str] = {}
    classes = data.get("classes")
    if isinstance(classes, list):
        for item in classes:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id", item.get("class_id"))
            name = item.get("name", item.get("label"))
            if raw_id is not None and isinstance(name, str) and name.strip():
                class_names[int(raw_id)] = name.strip()

    label_by_id: dict[int, dict[str, Any]] = {}
    id_to_label = data.get("id_to_label")
    if isinstance(id_to_label, dict):
        for raw_id, raw_label in id_to_label.items():
            if isinstance(raw_label, str) and raw_label.strip():
                label_by_id[int(raw_id)] = {"label": raw_label.strip()}

    objects = data.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id", item.get("object_id", item.get("instance_id")))
            if raw_id is None:
                continue
            object_id = int(raw_id)
            raw_class_id = item.get("class_id", item.get("category_id"))
            label = item.get("name", item.get("label", item.get("class_name")))
            if not isinstance(label, str) or not label.strip():
                label = class_names.get(int(raw_class_id)) if raw_class_id is not None else None
            label_by_id[object_id] = {
                "label": label.strip() if isinstance(label, str) and label.strip() else f"object_{object_id}",
                "class_id": int(raw_class_id) if raw_class_id is not None else None,
                "raw": item,
            }
    return label_by_id


def boxes_from_info_semantic(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return [], [{"reason": "info_semantic root is not an object"}]
    objects = data.get("objects")
    if not isinstance(objects, list):
        return [], [{"reason": "info_semantic has no objects array"}]

    boxes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            skipped.append({"reason": "object entry is not a JSON object"})
            continue
        raw_id = item.get("id", item.get("object_id", item.get("instance_id")))
        if raw_id is None:
            skipped.append({"reason": "object entry has no id"})
            continue
        object_id = int(raw_id)
        bbox = item.get("oriented_bbox")
        abb = bbox.get("abb") if isinstance(bbox, dict) else None
        if not isinstance(abb, dict):
            skipped.append({"object_id": object_id, "reason": "missing oriented_bbox.abb"})
            continue
        center = abb.get("center")
        size = abb.get("sizes", abb.get("size"))
        if not (
            isinstance(center, list)
            and isinstance(size, list)
            and len(center) == 3
            and len(size) == 3
        ):
            skipped.append({"object_id": object_id, "reason": "invalid bbox center/size"})
            continue
        center_values = [float(value) for value in center]
        size_values = [float(value) for value in size]
        if any(not math.isfinite(value) for value in [*center_values, *size_values]) or any(
            value <= 0.0 for value in size_values
        ):
            skipped.append({"object_id": object_id, "reason": "non-finite or non-positive bbox"})
            continue
        label = str(item.get("class_name") or item.get("name") or item.get("label") or f"object_{object_id}")
        boxes.append(
            {
                "object_id": str(object_id),
                "label": label,
                "center": [round(value, 6) for value in center_values],
                "size": [round(value, 9) for value in size_values],
                "min": [round(center_values[axis] - size_values[axis] / 2.0, 6) for axis in range(3)],
                "max": [round(center_values[axis] + size_values[axis] / 2.0, 6) for axis in range(3)],
                "coordinate_frame": "replica_habitat_mesh",
                "class_id": int(item["class_id"]) if item.get("class_id") is not None else None,
                "node_id": item.get("node_id"),
                "orientation": bbox.get("orientation") if isinstance(bbox, dict) else None,
                "bbox_source": "info_semantic.json/oriented_bbox",
            }
        )
    return boxes, skipped


def bbox_from_bounds(object_id: int, label: str, bounds: list[list[float]]) -> dict[str, Any] | None:
    mins, maxs = bounds
    size = [float(maxs[axis] - mins[axis]) for axis in range(3)]
    if any(not math.isfinite(value) or value <= 0.0 for value in size):
        return None
    center = [float((mins[axis] + maxs[axis]) / 2.0) for axis in range(3)]
    return {
        "object_id": str(object_id),
        "label": label,
        "center": [round(value, 6) for value in center],
        "size": [round(value, 9) for value in size],
        "min": [round(float(value), 6) for value in mins],
        "max": [round(float(value), 6) for value in maxs],
        "coordinate_frame": "replica_habitat_mesh",
    }


def placeholder_capture(scene_dir: Path) -> None:
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "depths").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(scene_dir / "images" / "v001.png")
    np.save(scene_dir / "depths" / "v001_depth.npy", np.ones((2, 2), dtype=np.float32))
    write_json(
        scene_dir / "transforms.json",
        {
            "camera_model": "OPENCV",
            "w": 2,
            "h": 2,
            "fl_x": 1.0,
            "fl_y": 1.0,
            "cx": 1.0,
            "cy": 1.0,
            "frames": [
                {
                    "view_id": "v001",
                    "file_path": "images/v001.png",
                    "depth_file_path": "depths/v001_depth.npy",
                    "transform_matrix": IDENTITY,
                    "synthetic_placeholder": True,
                }
            ],
        },
    )


def view_object_for(box: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    attributes = ["official_replica_instance", f"object_id:{box['object_id']}"]
    if info.get("class_id") is not None:
        attributes.append(f"class_id:{info['class_id']}")
    return {
        "label": box["label"],
        "attributes": attributes,
        "approx_location": "official Replica semantic mesh instance",
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


def import_replica_scene(
    source_scene: Path,
    output_scene: Path,
    *,
    scene_id: str,
    source_scene_id: str,
    copy_mesh: bool,
    overwrite: bool,
    include_unlabeled: bool,
) -> dict[str, Any]:
    source_scene = source_scene.resolve()
    output_scene = output_scene.resolve()
    habitat_dir = source_scene / "habitat"
    mesh_path = habitat_dir / "mesh_semantic.ply"
    info_path = habitat_dir / "info_semantic.json"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing official Replica semantic mesh: {mesh_path}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing official Replica info_semantic.json: {info_path}")
    if output_scene.exists() and any(output_scene.iterdir()) and not overwrite:
        raise FileExistsError(f"Output scene is not empty; use --overwrite: {output_scene}")
    if output_scene.exists() and overwrite:
        shutil.rmtree(output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)

    info_by_object_id = load_info_semantic(info_path)
    boxes, skipped = boxes_from_info_semantic(info_path)
    if boxes:
        ply_metadata = ply_header_metadata(mesh_path)
    else:
        bounds, ply_metadata = parse_ply_object_bounds(mesh_path)
        boxes = []
        skipped = []
        for object_id, object_bounds in sorted(bounds.items()):
            info = info_by_object_id.get(object_id, {})
            label = str(info.get("label") or "").strip()
            if not label and not include_unlabeled:
                skipped.append({"object_id": object_id, "reason": "missing label in info_semantic.json"})
                continue
            box = bbox_from_bounds(object_id, label or f"object_{object_id}", object_bounds)
            if box is None:
                skipped.append({"object_id": object_id, "reason": "invalid or degenerate bbox"})
                continue
            box["class_id"] = info.get("class_id")
            box["bbox_source"] = "mesh_semantic.ply/object_id_bounds"
            boxes.append(box)

    if not boxes:
        raise ValueError("No valid labeled object boxes were extracted from the Replica semantic mesh")

    placeholder_capture(output_scene)
    if copy_mesh:
        geometry_dir = output_scene / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, geometry_dir / "mesh_semantic.ply")

    visible_objects = [
        view_object_for(box, info_by_object_id.get(int(box["object_id"]), {}))
        for box in boxes
    ]
    label_counts: dict[str, int] = {}
    for box in boxes:
        label_counts[box["label"]] = label_counts.get(box["label"], 0) + 1
    write_json(
        output_scene / "views" / "v001.json",
        {
            "schema_version": "semanticsplat.captured_view_json.v1",
            "view_id": "v001",
            "scene_id": scene_id,
            "image_path": "images/v001.png",
            "depth_path": "depths/v001_depth.npy",
            "pose_ref": "transforms.json#/frames/0/transform_matrix",
            "intrinsics_ref": "transforms.json",
            "semantic_index_mode": "official_gt",
            "summary": (
                f"Official Replica semantic mesh index for {source_scene_id}: "
                + ", ".join(f"{label} x{count}" for label, count in sorted(label_counts.items())[:30])
            ),
            "visible_regions": [
                {
                    "label": "official Replica semantic mesh",
                    "approx_location": source_scene_id,
                    "source": "official_gt",
                }
            ],
            "visible_objects": visible_objects,
            "landmarks": [],
            "free_text_notes": "Mesh-level semantic GT index; RGB/depth files are synthetic placeholders for query-runner compatibility.",
            "warnings": [
                "Official Replica semantic mesh import.",
                "The image/depth frame is a synthetic placeholder and must not be used as visual or depth evidence.",
                "Use object_boxes_3d.json for GT boxes and mesh_semantic.ply provenance for geometry.",
            ],
        },
    )

    ground_truth_dir = output_scene / "ground_truth"
    write_json(
        ground_truth_dir / "object_boxes_3d.json",
        {
            "schema_version": "semanticsplat.object_boxes_3d_gt.v1",
            "scene_id": scene_id,
            "source_dataset": "Replica",
            "source_scene_id": source_scene_id,
            "source_mesh": display_path(mesh_path),
            "source_info": display_path(info_path),
            "coordinate_frame": "replica_habitat_mesh",
            "object_count": len(boxes),
            "objects": boxes,
            "skipped_objects": skipped,
        },
    )
    write_json(
        output_scene / "source_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id,
            "source_dataset": "Replica",
            "source_scene_id": source_scene_id,
            "source_scene_path": display_path(source_scene),
            "mesh_semantic_ply": display_path(mesh_path),
            "info_semantic_json": display_path(info_path),
            "ply": ply_metadata,
            "copy_mesh": copy_mesh,
            "warning": "Imported as mesh-level object GT; no rendered RGB-D camera trajectory is provided by this importer.",
        },
    )
    write_json(
        output_scene / "scene_manifest.json",
        {
            "schema_version": "semanticsplat.scene_manifest.v1",
            "scene_id": scene_id,
            "dataset": "Replica",
            "source_dataset": "Replica",
            "source_scene_id": source_scene_id,
            "source_description": f"Official Replica v1 semantic mesh imported from {display_path(source_scene)}",
            "paths": {
                "rgb": "images",
                "depth": "depths",
                "poses": "transforms.json",
                "intrinsics": "transforms.json",
                "semantic_index": "views",
                "gt_object_boxes_3d": "ground_truth/object_boxes_3d.json",
                "source_mesh": display_path(mesh_path),
            },
            "frame_count": 1,
            "coordinate_frame": "replica_habitat_mesh",
            "ground_truth": {
                "zone_labels_available": False,
                "object_labels_available": True,
                "bbox_3d_available": True,
                "source": "official Replica habitat/mesh_semantic.ply + info_semantic.json",
            },
            "evaluation_eligibility": {
                "semantic_evaluation_allowed": True,
                "three_d_localization_allowed": True,
                "rgbd_visual_evaluation_allowed": False,
                "reason": "Object labels and 3D boxes come from official Replica mesh GT; rendered RGB-D views are not available from this import.",
            },
            "warnings": [
                "Synthetic placeholder image/depth exists only to satisfy the current ViewJSON runner contract.",
                "Do not report depth, visual, or rendered-view metrics from this placeholder frame.",
            ],
        },
    )
    tree_manifest = build_scene_tree(output_scene, output_scene / "views", output_scene / "tree")
    validation = validate_semantic_index(output_scene)
    return {
        "scene_id": scene_id,
        "source_scene_id": source_scene_id,
        "object_count": len(boxes),
        "skipped_count": len(skipped),
        "tree_manifest": tree_manifest,
        "semantic_validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-scene-id", help="Original Replica scene id; defaults to source directory name")
    parser.add_argument("--output", type=Path, help="Output backend scene directory")
    parser.add_argument("--copy-mesh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional import report JSON path")
    args = parser.parse_args()

    source_scene_id = args.source_scene_id or args.source_scene.name
    output = args.output or ROOT / "backend" / "data" / "scenes" / args.scene_id
    result = import_replica_scene(
        args.source_scene,
        output,
        scene_id=args.scene_id,
        source_scene_id=source_scene_id,
        copy_mesh=args.copy_mesh,
        overwrite=args.overwrite,
        include_unlabeled=args.include_unlabeled,
    )
    report = args.report or ROOT / "docs" / "validation" / "public_datasets" / f"{args.scene_id}.json"
    write_json(report, result)
    print(json.dumps({"output": display_path(output), "report": display_path(report), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
