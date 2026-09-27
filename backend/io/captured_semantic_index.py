"""Validation helpers for the captured-scene semantic-index contract."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


VIEW_SCHEMA_VERSION = "semanticsplat.captured_view_json.v1"
TREE_MANIFEST_SCHEMA_VERSION = "semanticsplat.semantic_tree_manifest.v1"
TREE_NODE_SCHEMA_VERSION = "semanticsplat.semantic_tree_node.v1"
VALID_MODES = {"manual", "live", "cached_live", "stub", "official_gt"}
VALID_LANDMARK_KINDS = {"landmark", "sign", "facility"}
VIEW_ID_PATTERN = re.compile(r"^v[0-9]+$")

# Phase H adapters in model_client and live_session read this contract directly.
QUERY_RUNNER_SCHEMA_COMPATIBLE = True


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def safe_scene_path(scene_dir: Path, raw_value: Any, field_name: str) -> tuple[Path, str]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{field_name} must be a non-empty scene-relative path")
    relative = Path(raw_value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {field_name}: {raw_value}")
    resolved = (scene_dir / relative).resolve()
    try:
        normalized = resolved.relative_to(scene_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes scene directory: {raw_value}") from exc
    return resolved, normalized


def load_frame_map(scene_dir: Path) -> dict[str, dict[str, Any]]:
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")
    transforms = load_json(transforms_path)
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json must contain a non-empty frames list")

    result: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {index} is not an object")
        view_id = frame.get("view_id")
        if not isinstance(view_id, str) or not VIEW_ID_PATTERN.fullmatch(view_id):
            raise ValueError(f"Frame {index} has invalid view_id: {view_id!r}")
        if view_id in result:
            raise ValueError(f"Duplicate transform frame view_id: {view_id}")
        result[view_id] = {"index": index, "frame": frame}
    return result


def _require_string(data: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = data.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _validate_source(entry: dict[str, Any], mode: str, context: str) -> str:
    source = _require_string(entry, "source")
    if source not in VALID_MODES:
        raise ValueError(f"{context}.source must be one of {sorted(VALID_MODES)}")
    if source != mode:
        raise ValueError(f"{context}.source={source!r} disagrees with semantic_index_mode={mode!r}")
    return source


def _validate_bbox_2d(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context}.bbox_2d must be null or [x1, y1, x2, y2]")
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        raise ValueError(f"{context}.bbox_2d must contain finite numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{context}.bbox_2d must use ordered normalized coordinates")


def _validate_bbox_3d(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context}.bbox_3d must be null or an object")
    for field in ("center", "size"):
        vector = value.get(field)
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"{context}.bbox_3d.{field} must have three values")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in vector):
            raise ValueError(f"{context}.bbox_3d.{field} must contain finite numbers")
    if any(float(item) <= 0 for item in value["size"]):
        raise ValueError(f"{context}.bbox_3d.size values must be positive")


def validate_view_record(
    data: dict[str, Any],
    *,
    path: Path,
    scene_dir: Path,
    frame_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if data.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {VIEW_SCHEMA_VERSION!r}")
    view_id = _require_string(data, "view_id")
    if path.stem != view_id:
        raise ValueError(f"filename {path.name!r} disagrees with view_id={view_id!r}")
    if view_id not in frame_map:
        raise ValueError(f"view_id {view_id!r} is absent from transforms.json")
    scene_id = _require_string(data, "scene_id")
    if scene_id != scene_dir.name:
        raise ValueError(f"scene_id={scene_id!r} does not match directory {scene_dir.name!r}")

    mode = _require_string(data, "semantic_index_mode")
    if mode not in VALID_MODES:
        raise ValueError(f"semantic_index_mode must be one of {sorted(VALID_MODES)}")
    _require_string(data, "summary", allow_empty=True)
    _require_string(data, "free_text_notes", allow_empty=True)

    expected = frame_map[view_id]
    frame = expected["frame"]
    image_file, image_path = safe_scene_path(scene_dir, data.get("image_path"), "image_path")
    depth_file, depth_path = safe_scene_path(scene_dir, data.get("depth_path"), "depth_path")
    if not image_file.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")
    if not depth_file.is_file():
        raise ValueError(f"depth_path does not exist: {depth_path}")
    expected_image = str(frame.get("file_path", "")).replace("\\", "/")
    expected_depth = str(frame.get("depth_file_path", "")).replace("\\", "/")
    if image_path != expected_image:
        raise ValueError(f"image_path={image_path!r} does not match transforms reference {expected_image!r}")
    if depth_path != expected_depth:
        raise ValueError(f"depth_path={depth_path!r} does not match transforms reference {expected_depth!r}")

    expected_pose_ref = f"transforms.json#/frames/{expected['index']}/transform_matrix"
    if data.get("pose_ref") != expected_pose_ref:
        raise ValueError(f"pose_ref must be {expected_pose_ref!r}")
    if data.get("intrinsics_ref") != "transforms.json":
        raise ValueError("intrinsics_ref must be 'transforms.json'")

    regions = data.get("visible_regions")
    objects = data.get("visible_objects")
    landmarks = data.get("landmarks")
    warnings = data.get("warnings")
    if not isinstance(regions, list):
        raise ValueError("visible_regions must be an array")
    if not isinstance(objects, list):
        raise ValueError("visible_objects must be an array")
    if not isinstance(landmarks, list):
        raise ValueError("landmarks must be an array")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError("warnings must be an array of strings")

    for index, region in enumerate(regions):
        context = f"visible_regions[{index}]"
        if not isinstance(region, dict):
            raise ValueError(f"{context} must be an object")
        _require_string(region, "label")
        _require_string(region, "approx_location", allow_empty=True)
        _validate_source(region, mode, context)

    for index, obj in enumerate(objects):
        context = f"visible_objects[{index}]"
        if not isinstance(obj, dict):
            raise ValueError(f"{context} must be an object")
        _require_string(obj, "label")
        _require_string(obj, "approx_location", allow_empty=True)
        attributes = obj.get("attributes")
        if not isinstance(attributes, list) or not all(isinstance(item, str) for item in attributes):
            raise ValueError(f"{context}.attributes must be an array of strings")
        confidence = obj.get("confidence")
        if confidence is not None and not (
            isinstance(confidence, (int, float))
            and math.isfinite(confidence)
            and 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"{context}.confidence must be null or a number in [0, 1]")
        if mode == "manual" and confidence is not None:
            raise ValueError(f"{context}.confidence must be null for manual annotations")
        _validate_bbox_2d(obj.get("bbox_2d"), context)
        _validate_bbox_3d(obj.get("bbox_3d"), context)
        _validate_source(obj, mode, context)

    for index, landmark in enumerate(landmarks):
        context = f"landmarks[{index}]"
        if not isinstance(landmark, dict):
            raise ValueError(f"{context} must be an object")
        _require_string(landmark, "label")
        _require_string(landmark, "approx_location", allow_empty=True)
        kind = _require_string(landmark, "kind")
        if kind not in VALID_LANDMARK_KINDS:
            raise ValueError(f"{context}.kind must be one of {sorted(VALID_LANDMARK_KINDS)}")
        _validate_source(landmark, mode, context)
    return data


def inspect_view_index(scene_dir: Path, views_dir: Path) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    frame_map = load_frame_map(scene_dir)
    files = sorted(views_dir.glob("*.json")) if views_dir.exists() else []
    valid_views: list[dict[str, Any]] = []
    invalid_files: list[dict[str, str]] = []
    file_view_ids: set[str] = set()
    for path in files:
        file_view_ids.add(path.stem)
        try:
            data = load_json(path)
            valid_views.append(
                validate_view_record(data, path=path, scene_dir=scene_dir, frame_map=frame_map)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid_files.append({"file": path.name, "error": str(exc)})

    expected_ids = set(frame_map)
    return {
        "frame_map": frame_map,
        "expected_view_ids": sorted(expected_ids),
        "template_file_count": len(files),
        "valid_views": sorted(valid_views, key=lambda item: item["view_id"]),
        "invalid_files": invalid_files,
        "missing_view_ids": sorted(expected_ids - file_view_ids),
        "unexpected_view_ids": sorted(file_view_ids - expected_ids),
    }


def semantic_counts(views: list[dict[str, Any]]) -> dict[str, int]:
    region_count = sum(len(view["visible_regions"]) for view in views)
    object_count = sum(len(view["visible_objects"]) for view in views)
    landmark_count = 0
    sign_count = 0
    facility_count = 0
    for view in views:
        for item in view["landmarks"]:
            if item["kind"] == "sign":
                sign_count += 1
            elif item["kind"] == "facility":
                facility_count += 1
            else:
                landmark_count += 1
    return {
        "visible_region_count": region_count,
        "visible_object_count": object_count,
        "landmark_count": landmark_count,
        "sign_count": sign_count,
        "facility_count": facility_count,
        "semantic_item_count": region_count + object_count + landmark_count + sign_count + facility_count,
        "annotated_view_count": sum(bool(view["summary"].strip()) for view in views),
        "bbox_3d_count": sum(
            obj.get("bbox_3d") is not None
            for view in views
            for obj in view["visible_objects"]
        ),
    }


def load_captured_scene_index(
    scene_dir: Path,
    *,
    require_complete: bool = True,
    allow_unindexed_frames: bool = False,
) -> dict[str, Any]:
    """Load a captured ViewJSON/tree index without routing through legacy schemas."""
    scene_dir = scene_dir.resolve()
    inspection = inspect_view_index(scene_dir, scene_dir / "views")
    problems = [f"{item['file']}: {item['error']}" for item in inspection["invalid_files"]]
    missing_view_ids = inspection["missing_view_ids"]
    if missing_view_ids and not allow_unindexed_frames:
        problems.append("missing ViewJSON: " + ", ".join(inspection["missing_view_ids"]))
    if inspection["unexpected_view_ids"]:
        problems.append("unexpected ViewJSON: " + ", ".join(inspection["unexpected_view_ids"]))
    if problems:
        raise ValueError("Captured ViewJSON validation failed: " + " | ".join(problems))

    manifest_path = scene_dir / "tree" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Captured semantic tree manifest is missing")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != TREE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Captured semantic tree manifest has an unsupported schema")
    if manifest.get("scene_id") != scene_dir.name:
        raise ValueError("Captured semantic tree manifest scene_id mismatch")
    node_ids = manifest.get("node_ids")
    if not isinstance(node_ids, list) or not node_ids:
        raise ValueError("Captured semantic tree manifest has no node_ids")

    nodes: dict[str, dict[str, Any]] = {}
    for raw_node_id in node_ids:
        node_id = str(raw_node_id)
        path = scene_dir / "tree" / f"node_{node_id}.json"
        if not path.is_file():
            raise ValueError(f"Captured semantic tree node is missing: {path.name}")
        node = load_json(path)
        if node.get("schema_version") != TREE_NODE_SCHEMA_VERSION:
            raise ValueError(f"Captured semantic tree node has unsupported schema: {path.name}")
        if node.get("node_id") != node_id or node.get("scene_id") != scene_dir.name:
            raise ValueError(f"Captured semantic tree node identity mismatch: {path.name}")
        nodes[node_id] = node

    root_id = manifest.get("root_node_id")
    if root_id not in nodes or nodes[str(root_id)].get("node_type") != "root":
        raise ValueError("Captured semantic tree root is missing or invalid")
    counts = semantic_counts(inspection["valid_views"])
    indexed_view_ids = {view["view_id"] for view in inspection["valid_views"]}
    if require_complete:
        if not manifest.get("complete"):
            raise ValueError("Captured semantic index is incomplete; manual annotation is required")
        if counts["semantic_item_count"] == 0:
            raise ValueError("Captured semantic index has no semantic items")
        if counts["annotated_view_count"] != len(indexed_view_ids):
            raise ValueError("Captured semantic index contains empty view summaries")
        manifest_view_count = manifest.get("view_count")
        if isinstance(manifest_view_count, int) and manifest_view_count != len(indexed_view_ids):
            raise ValueError("Captured semantic tree manifest view_count does not match indexed ViewJSON files")
    warnings: list[str] = []
    if missing_view_ids and allow_unindexed_frames:
        warnings.append(
            "Unindexed captured frames are excluded from semantic query search: "
            + ", ".join(missing_view_ids)
        )
    return {
        "manifest": manifest,
        "views": {view["view_id"]: view for view in inspection["valid_views"]},
        "tree": nodes,
        "counts": counts,
        "unindexed_frame_ids": missing_view_ids,
        "warnings": warnings,
    }
