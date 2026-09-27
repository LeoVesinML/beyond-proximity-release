#!/usr/bin/env python3
"""Check whether a manually captured pilot is safe to scale beyond one scene."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_scene_geometry import pose_details  # noqa: E402
from backend.schemas.types import ViewJSON  # noqa: E402
from backend.tree.nodes import TreeNode  # noqa: E402


REQUIRED_METADATA_KEYS = {
    "scene_id",
    "frame_id",
    "image_path",
    "depth_path",
    "pose_reference",
    "intrinsics_reference",
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def resolution_value(values: list[tuple[int, int]]) -> list[int] | dict[str, Any] | None:
    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) == 1:
        return list(unique[0])
    return {"consistent": False, "values": [list(value) for value in unique]}


def ground_truth_status(scene_dir: Path) -> tuple[bool, bool]:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.exists():
        return False, False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, False
    gt = manifest.get("ground_truth") if isinstance(manifest.get("ground_truth"), dict) else {}
    semantic_gt = bool(gt.get("zone_labels_available") or gt.get("object_labels_available"))
    geometry_gt = bool(gt.get("bbox_3d_available"))
    return semantic_gt, geometry_gt


def semantic_index_status(
    scene_dir: Path,
    frame_ids: list[str],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    valid_view_ids: set[str] = set()
    invalid_view_ids: list[str] = []
    views_dir = scene_dir / "views"
    view_files = sorted(views_dir.glob("*.json")) if views_dir.exists() else []
    for path in view_files:
        try:
            view = ViewJSON.model_validate(load_json(path))
            if view.view_id != path.stem:
                raise ValueError(f"view_id is {view.view_id!r}, expected {path.stem!r}")
            valid_view_ids.add(path.stem)
        except Exception as exc:
            invalid_view_ids.append(path.stem)
            warnings.append(f"{path.name}: invalid ViewJSON: {exc}")

    missing_view_ids = sorted(set(frame_ids) - valid_view_ids)
    views_complete = bool(frame_ids and not missing_view_ids and not invalid_view_ids)
    tree_manifest = scene_dir / "tree" / "manifest.json"
    tree_valid = False
    tree_node_count = 0
    tree_view_ids: set[str] = set()
    if tree_manifest.exists():
        try:
            data = load_json(tree_manifest)
            root_id = data.get("root_id")
            node_ids = data.get("node_ids")
            if not isinstance(root_id, str) or not root_id:
                raise ValueError("manifest root_id is missing")
            if not isinstance(node_ids, list) or not node_ids or not all(isinstance(item, str) for item in node_ids):
                raise ValueError("manifest node_ids must be a non-empty string list")

            nodes: dict[str, TreeNode] = {}
            for node_id in node_ids:
                node_path = scene_dir / "tree" / f"node_{node_id}.json"
                if not node_path.exists():
                    raise ValueError(f"missing tree node file {node_path.name}")
                node = TreeNode.from_dict(load_json(node_path))
                if node.node_id != node_id:
                    raise ValueError(f"{node_path.name} contains node_id {node.node_id!r}")
                nodes[node_id] = node
                tree_view_ids.update(node.view_ids)

            if root_id not in nodes:
                raise ValueError(f"root node {root_id!r} is not listed in manifest")
            dangling_children = sorted({child for node in nodes.values() for child in node.children_ids} - set(nodes))
            if dangling_children:
                raise ValueError(f"missing child nodes: {', '.join(dangling_children)}")
            tree_node_count = len(nodes)
            tree_valid = True
        except Exception as exc:
            warnings.append(f"Invalid semantic tree: {exc}")
            tree_valid = False

    tree_covers_all_views = bool(frame_ids and set(frame_ids).issubset(tree_view_ids))
    complete = bool(views_complete and tree_valid and tree_covers_all_views)
    return {
        "view_json_count": len(valid_view_ids),
        "invalid_view_json_count": len(invalid_view_ids),
        "missing_view_ids": missing_view_ids,
        "tree_node_count": tree_node_count,
        "tree_valid": tree_valid,
        "tree_covers_all_views": tree_covers_all_views,
        "complete": complete,
    }, warnings


def check_capture(scene_dir: Path) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")
    transforms = load_json(transforms_path)
    frames = transforms.get("frames")
    if not isinstance(frames, list):
        raise ValueError("transforms.json must contain a frames list")

    warnings: list[str] = []
    rgb_sizes: list[tuple[int, int]] = []
    depth_sizes: list[tuple[int, int]] = []
    positive_depth_parts: list[np.ndarray] = []
    depth_pixel_count = 0
    zero_or_invalid_count = 0
    constant_frames: list[str] = []
    valid_pose_count = 0
    metadata_complete_count = 0
    rgb_count = 0
    depth_count = 0
    pose_count = 0
    frame_ids: list[str] = []

    for index, raw_frame in enumerate(frames):
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_id = str(frame.get("view_id") or f"frame_{index + 1:03d}")
        frame_ids.append(frame_id)

        rgb_value = frame.get("file_path")
        if isinstance(rgb_value, str):
            rgb_path = scene_dir / Path(rgb_value.replace("\\", "/"))
            if rgb_path.exists():
                try:
                    with Image.open(rgb_path) as image:
                        image.load()
                        rgb_sizes.append((int(image.width), int(image.height)))
                    rgb_count += 1
                except Exception as exc:
                    warnings.append(f"{frame_id}: unreadable RGB: {exc}")
            else:
                warnings.append(f"{frame_id}: missing RGB {rgb_value}")
        else:
            warnings.append(f"{frame_id}: missing RGB reference")

        depth_value = frame.get("depth_file_path")
        if isinstance(depth_value, str):
            depth_path = scene_dir / Path(depth_value.replace("\\", "/"))
            if depth_path.exists():
                try:
                    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
                    if depth.ndim != 2:
                        raise ValueError(f"expected 2D depth, got shape {depth.shape}")
                    depth_sizes.append((int(depth.shape[1]), int(depth.shape[0])))
                    depth_count += 1
                    depth_pixel_count += int(depth.size)
                    valid = np.isfinite(depth) & (depth > 0)
                    zero_or_invalid_count += int(depth.size - np.count_nonzero(valid))
                    positive = depth[valid]
                    if positive.size == 0:
                        constant_frames.append(frame_id)
                        warnings.append(f"{frame_id}: no positive finite depth")
                    else:
                        positive_depth_parts.append(positive)
                        minimum = float(np.min(positive))
                        maximum = float(np.max(positive))
                        if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-6):
                            constant_frames.append(frame_id)
                            warnings.append(f"{frame_id}: constant depth {minimum:g}")
                except Exception as exc:
                    warnings.append(f"{frame_id}: unreadable depth: {exc}")
            else:
                warnings.append(f"{frame_id}: missing depth {depth_value}")
        else:
            warnings.append(f"{frame_id}: missing depth reference")

        if frame.get("transform_matrix") is not None:
            pose_count += 1
            details = pose_details(frame.get("transform_matrix"))
            if details["valid"]:
                valid_pose_count += 1
            else:
                warnings.append(f"{frame_id}: invalid pose matrix")

        metadata_path = scene_dir / "captures" / f"{frame_id}.json"
        if not metadata_path.exists():
            warnings.append(f"{frame_id}: missing capture metadata")
        else:
            try:
                metadata = load_json(metadata_path)
                missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
                references_match = bool(
                    metadata.get("scene_id") == scene_dir.name
                    and metadata.get("frame_id") == frame_id
                    and metadata.get("image_path") == rgb_value
                    and metadata.get("depth_path") == depth_value
                )
                if missing:
                    warnings.append(f"{frame_id}: metadata missing {', '.join(missing)}")
                elif not references_match:
                    warnings.append(f"{frame_id}: metadata references do not match transforms")
                else:
                    metadata_complete_count += 1
            except Exception as exc:
                warnings.append(f"{frame_id}: invalid capture metadata: {exc}")

    width = transforms.get("w")
    height = transforms.get("h")
    intrinsics_values = [
        transforms.get("fl_x"), transforms.get("fl_y"),
        transforms.get("cx"), transforms.get("cy"),
    ]
    intrinsics_valid = bool(
        isinstance(width, int) and width > 0
        and isinstance(height, int) and height > 0
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in intrinsics_values)
        and float(transforms.get("fl_x", 0)) > 0
        and float(transforms.get("fl_y", 0)) > 0
    )
    rgb_resolution = resolution_value(rgb_sizes)
    depth_resolution = resolution_value(depth_sizes)
    intrinsics_match = bool(
        intrinsics_valid
        and rgb_sizes
        and all(size == (int(width), int(height)) for size in rgb_sizes)
    )
    resolutions_match = bool(
        len(rgb_sizes) == len(depth_sizes) == len(frames)
        and all(rgb == depth for rgb, depth in zip(rgb_sizes, depth_sizes))
    )

    if positive_depth_parts:
        positive_depth = np.concatenate(positive_depth_parts)
        depth_min = float(np.min(positive_depth))
        depth_max = float(np.max(positive_depth))
        depth_mean = float(np.mean(positive_depth))
    else:
        depth_min = depth_max = depth_mean = None
    depth_is_constant = bool(constant_frames) if depth_count else None
    zero_depth_ratio = (
        round(zero_or_invalid_count / depth_pixel_count, 6) if depth_pixel_count else None
    )

    metadata_complete = bool(frames and metadata_complete_count == len(frames))
    pose_all_valid = bool(frames and pose_count == valid_pose_count == len(frames))
    counts_match = bool(frames and rgb_count == depth_count == pose_count == len(frames))
    capture_scale_ready = bool(
        counts_match
        and resolutions_match
        and intrinsics_match
        and pose_all_valid
        and metadata_complete
        and depth_is_constant is False
        and zero_depth_ratio is not None
        and zero_depth_ratio <= 0.5
    )
    semantic_index, semantic_index_warnings = semantic_index_status(scene_dir, frame_ids)
    warnings.extend(semantic_index_warnings)
    semantic_index_complete = bool(semantic_index["complete"])
    semantic_gt, geometry_gt = ground_truth_status(scene_dir)
    semantic_eval_allowed = bool(capture_scale_ready and semantic_index_complete and semantic_gt)
    geometry_eval_allowed = bool(capture_scale_ready and geometry_gt)

    if not intrinsics_match:
        warnings.append("Intrinsics do not match every RGB render resolution.")
    if not resolutions_match:
        warnings.append("RGB/depth counts or resolutions do not match.")
    if not metadata_complete:
        warnings.append("Capture metadata is incomplete.")
    if zero_depth_ratio is not None and zero_depth_ratio > 0.5:
        warnings.append(f"Zero/invalid depth ratio {zero_depth_ratio:.3f} exceeds 0.500.")
    if not semantic_index_complete:
        warnings.append("ViewJSON/tree semantic index is incomplete or missing.")
    if not semantic_gt:
        warnings.append("Independent semantic ground truth is missing.")
    if not geometry_gt:
        warnings.append("Independent GT 3D boxes/masks are missing.")

    return {
        "scene": str(scene_dir),
        "rgb_count": rgb_count,
        "depth_count": depth_count,
        "pose_count": pose_count,
        "intrinsics_status": "valid" if intrinsics_valid else "invalid_or_missing",
        "metadata_status": "complete" if metadata_complete else "incomplete_or_missing",
        "rgb_resolution": rgb_resolution,
        "depth_resolution": depth_resolution,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_mean": depth_mean,
        "depth_is_constant": depth_is_constant,
        "zero_depth_ratio": zero_depth_ratio,
        "pose_validity": {
            "valid_count": valid_pose_count,
            "pose_count": pose_count,
            "all_valid": pose_all_valid,
        },
        "intrinsics_match_canvas": intrinsics_match,
        "semantic_index": semantic_index,
        "semantic_eval_allowed": semantic_eval_allowed,
        "geometry_eval_allowed": geometry_eval_allowed,
        "scaling_allowed": capture_scale_ready,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one manual PLY capture pilot")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check_capture(args.scene)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Capture pilot check failed: {exc}") from exc
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    print(json.dumps(report, indent=2))
    print(
        f"scaling_allowed={str(report['scaling_allowed']).lower()} "
        f"semantic_eval_allowed={str(report['semantic_eval_allowed']).lower()} "
        f"geometry_eval_allowed={str(report['geometry_eval_allowed']).lower()}"
    )


if __name__ == "__main__":
    main()
