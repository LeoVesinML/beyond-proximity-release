#!/usr/bin/env python3
"""Validate RGB-D scene geometry and metric eligibility."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_replica_scene import (  # noqa: E402
    detect_layout,
    display_path,
    intrinsics_from_replica,
    load_json,
    normalized_relative,
    read_ply_header,
    sorted_media,
    valid_pose,
    write_json,
)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def depth_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Depth array must be 2D, got {array.shape}")
    return array


def metadata_details(scene_path: Path, *, scene_id: str, frame_count: int) -> dict[str, Any]:
    if scene_path.is_file():
        candidates = [
            scene_path.with_suffix(".json"),
            scene_path.with_name(f"{scene_path.stem}.metadata.json"),
            scene_path.with_name(f"{scene_path.stem}_metadata.json"),
            scene_path.parent / scene_path.stem / "metadata.json",
        ]
        metadata_root = scene_path.parent
    else:
        candidates = [
            scene_path / "metadata.json",
            scene_path / "source_metadata.json",
            scene_path / "scene_manifest.json",
        ]
        metadata_root = scene_path

    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.is_file():
            existing.append(candidate)
            seen.add(resolved)
    if not existing:
        return {
            "status": "missing",
            "present": False,
            "consistent": None,
            "files_checked": [],
            "checks_performed": 0,
            "issues": ["No metadata sidecar was found."],
        }

    entries: list[dict[str, Any]] = []
    issues: list[str] = []
    checks_performed = 0
    for path in existing:
        entry_issues: list[str] = []
        try:
            data = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            entry_issues.append(f"unreadable JSON: {exc}")
            data = {}

        declared_scene_id = data.get("scene_id")
        if declared_scene_id is not None:
            checks_performed += 1
            if str(declared_scene_id) != scene_id:
                entry_issues.append(
                    f"scene_id mismatch: metadata={declared_scene_id!r}, expected={scene_id!r}"
                )
        declared_frame_count = data.get("frame_count")
        if declared_frame_count is not None:
            checks_performed += 1
            try:
                normalized_count = int(declared_frame_count)
            except (TypeError, ValueError):
                entry_issues.append(f"frame_count is not an integer: {declared_frame_count!r}")
            else:
                if normalized_count != frame_count:
                    entry_issues.append(
                        f"frame_count mismatch: metadata={normalized_count}, observed={frame_count}"
                    )

        declared_paths = data.get("paths")
        if isinstance(declared_paths, dict):
            for key, value in declared_paths.items():
                if value is None or not isinstance(value, str):
                    continue
                checks_performed += 1
                try:
                    relative = normalized_relative(value)
                except ValueError as exc:
                    entry_issues.append(f"paths.{key} is invalid: {exc}")
                    continue
                if not (metadata_root / relative).exists():
                    entry_issues.append(f"paths.{key} does not exist: {value}")

        entries.append({
            "path": display_path(path),
            "readable": not any(issue.startswith("unreadable JSON") for issue in entry_issues),
            "issues": entry_issues,
        })
        issues.extend(f"{display_path(path)}: {issue}" for issue in entry_issues)

    if issues:
        status = "inconsistent"
        consistent: bool | None = False
    elif checks_performed:
        status = "consistent"
        consistent = True
    else:
        status = "present_unverified"
        consistent = None
    return {
        "status": status,
        "present": True,
        "consistent": consistent,
        "files_checked": entries,
        "checks_performed": checks_performed,
        "issues": issues,
    }


def validate_ply_scene(
    scene_path: Path,
    *,
    scene_id: str | None = None,
    scene_root: Path | None = None,
) -> dict[str, Any]:
    resolved = scene_path.resolve()
    resolved_root = scene_root.resolve() if scene_root is not None else None
    resolved_scene_id = scene_id or (resolved_root.name if resolved_root is not None else resolved.stem)
    warnings = [
        "Standalone PLY asset has no RGB frame records.",
        "Depth is unavailable; constant-depth and zero-depth checks are N/A, not passed.",
        "Camera poses and intrinsics are unavailable.",
        "Semantic labels, instance labels, and GT 3D boxes are unavailable.",
    ]
    try:
        ply = read_ply_header(resolved)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        ply = {
            "valid": False,
            "format": None,
            "comments": [],
            "elements": {},
            "properties": {},
            "header_bytes": None,
            "semantic_or_instance_properties": [],
            "error": str(exc),
        }
        warnings.append(f"PLY header is invalid or unreadable: {exc}")

    metadata = metadata_details(resolved_root or resolved, scene_id=resolved_scene_id, frame_count=0)
    warnings.extend(metadata["issues"])
    checks = {
        "missing_frames": 1,
        "missing_rgb": 0,
        "missing_depth": 0,
        "constant_depth": 0,
        "excessive_zero_depth": 0,
        "rgb_depth_resolution_mismatch": 0,
        "intrinsics_resolution_mismatch": 0,
        "invalid_pose": 0,
        "metadata_missing": 0 if metadata["present"] else 1,
        "metadata_inconsistency": 1 if metadata["consistent"] is False else 0,
        "invalid_ply_header": 0 if ply["valid"] else 1,
    }
    return {
        "schema_version": "semanticsplat.geometry_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": resolved_scene_id,
        "scene_path": display_path(resolved_root or resolved),
        "layout": "ply_backend" if resolved_root is not None else "ply_only",
        "asset_type": "gaussian_splat_ply",
        "asset_valid": bool(ply["valid"]),
        "frame_count": 0,
        "rgb_count": 0,
        "depth_count": 0,
        "pose_count": 0,
        "valid_pose_count": 0,
        "intrinsics": {
            "w": 0,
            "h": 0,
            "fl_x": 0.0,
            "fl_y": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "present": False,
            "status": "missing",
        },
        "metadata": metadata,
        "ply": {**ply, "path": display_path(resolved)},
        "checks": checks,
        "check_availability": {
            "rgb_count": "measured_absent",
            "depth_count": "measured_absent",
            "pose_count": "measured_absent",
            "rgb_depth_resolution_match": "N/A",
            "intrinsics_resolution_match": "N/A",
            "constant_depth": "N/A",
            "zero_depth_ratio": "N/A",
            "invalid_pose_matrices": "N/A",
            "missing_frames": "measured",
            "metadata_consistency": metadata["status"],
        },
        "semantic_evaluation_allowed": False,
        "three_d_localization_allowed": False,
        "metrics_eligibility": {
            "semantic_retrieval": "not_allowed",
            "expected_view_node_zone": "not_allowed",
            "bbox_3d": "N/A",
            "iou_3d": "N/A",
        },
        "warnings": warnings,
        "per_frame": [],
    }


def pose_details(matrix: Any) -> dict[str, Any]:
    base_valid = valid_pose(matrix)
    result = {
        "valid": base_valid,
        "shape_valid": False,
        "finite": False,
        "homogeneous_row_valid": False,
        "rotation_orthonormal": False,
        "rotation_determinant": None,
    }
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return result
    result["shape_valid"] = array.shape == (4, 4)
    result["finite"] = bool(np.isfinite(array).all()) if array.shape == (4, 4) else False
    if array.shape != (4, 4) or not result["finite"]:
        return result
    result["homogeneous_row_valid"] = bool(np.allclose(array[3], [0, 0, 0, 1], atol=1e-5))
    rotation = array[:3, :3]
    determinant = float(np.linalg.det(rotation))
    result["rotation_determinant"] = round(determinant, 6)
    result["rotation_orthonormal"] = bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) and abs(determinant - 1.0) <= 1e-3
    )
    result["valid"] = bool(base_valid and result["rotation_orthonormal"])
    return result


def scene_inputs(scene_dir: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    # A canonical PLY directory becomes a backend RGB-D scene after manual
    # capture. Prefer real frame records over the geometry-only marker.
    if (scene_dir / "transforms.json").exists():
        layout = "backend"
    elif (scene_dir / "poses.json").exists() and (scene_dir / "intrinsics.json").exists():
        layout = "replica_style"
    else:
        layout = detect_layout(scene_dir)
    if layout in {"ply_only", "ply_backend"}:
        return layout, [], {}
    if layout == "backend":
        transforms = load_json(scene_dir / "transforms.json")
        frames = []
        for index, raw in enumerate(transforms.get("frames", [])):
            if not isinstance(raw, dict):
                frames.append({
                    "view_id": f"invalid_{index + 1:03d}",
                    "rgb_path": Path(f"missing_{index + 1:03d}.png"),
                    "depth_path": None,
                    "transform_matrix": raw,
                })
                continue
            rgb_value = raw.get("file_path", f"missing_{index + 1:03d}.png")
            depth_value = raw.get("depth_file_path")
            frames.append({
                "view_id": str(raw.get("view_id") or f"v{index + 1:03d}"),
                "rgb_path": normalized_relative(rgb_value),
                "depth_path": normalized_relative(depth_value) if depth_value else None,
                "transform_matrix": raw.get("transform_matrix"),
            })
        intrinsics = {
            "w": int(transforms.get("w", 0)),
            "h": int(transforms.get("h", 0)),
            "fl_x": float(transforms.get("fl_x", 0.0)),
            "fl_y": float(transforms.get("fl_y", transforms.get("fl_x", 0.0))),
            "cx": float(transforms.get("cx", 0.0)),
            "cy": float(transforms.get("cy", 0.0)),
        }
    else:
        poses = load_json(scene_dir / "poses.json")
        frames = []
        raw_frames = poses.get("frames")
        rgb_candidates = sorted_media(scene_dir / "rgb", {".png", ".jpg", ".jpeg"})
        depth_candidates = sorted_media(scene_dir / "depth", {".npy", ".png", ".tif", ".tiff"})
        if isinstance(raw_frames, list):
            for index, raw in enumerate(raw_frames):
                raw = raw if isinstance(raw, dict) else {}
                rgb_value = raw.get("rgb_file", raw.get("file_path", raw.get("rgb")))
                depth_value = raw.get("depth_file", raw.get("depth_file_path", raw.get("depth")))
                if rgb_value is None and index < len(rgb_candidates):
                    rgb_value = rgb_candidates[index].relative_to(scene_dir)
                if depth_value is None and index < len(depth_candidates):
                    depth_value = depth_candidates[index].relative_to(scene_dir)
                frames.append({
                    "view_id": str(raw.get("view_id") or f"v{index + 1:03d}"),
                    "rgb_path": normalized_relative(rgb_value or f"missing_{index + 1:03d}.png"),
                    "depth_path": normalized_relative(depth_value) if depth_value else None,
                    "transform_matrix": raw.get("transform_matrix", raw.get("pose")),
                })
        else:
            matrices = poses.get("poses") if isinstance(poses.get("poses"), list) else []
            for index, matrix in enumerate(matrices):
                frames.append({
                    "view_id": f"v{index + 1:03d}",
                    "rgb_path": rgb_candidates[index].relative_to(scene_dir) if index < len(rgb_candidates) else Path(f"missing_{index + 1:03d}.png"),
                    "depth_path": depth_candidates[index].relative_to(scene_dir) if index < len(depth_candidates) else None,
                    "transform_matrix": matrix,
                })
        intrinsics = intrinsics_from_replica(scene_dir / "intrinsics.json")
    return layout, frames, intrinsics


def validate_scene(
    scene_dir: Path,
    *,
    scene_id: str | None = None,
    max_zero_fraction: float = 0.5,
) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    if scene_dir.is_file():
        if scene_dir.suffix.lower() != ".ply":
            raise ValueError(f"Unsupported scene file: {scene_dir}; expected a .ply asset")
        return validate_ply_scene(scene_dir, scene_id=scene_id)
    if not scene_dir.is_dir():
        raise ValueError(f"Scene path does not exist or is not a directory/file: {scene_dir}")
    canonical_ply = scene_dir / "geometry" / "scene.ply"
    has_frame_records = bool(
        (scene_dir / "transforms.json").exists()
        or ((scene_dir / "poses.json").exists() and (scene_dir / "intrinsics.json").exists())
    )
    if canonical_ply.is_file() and not has_frame_records:
        return validate_ply_scene(
            canonical_ply,
            scene_id=scene_id or scene_dir.name,
            scene_root=scene_dir,
        )

    layout, frames, intrinsics = scene_inputs(scene_dir)
    resolved_scene_id = scene_id or scene_dir.name
    metadata = metadata_details(scene_dir, scene_id=resolved_scene_id, frame_count=len(frames))
    warnings: list[str] = []
    checks = {
        "missing_frames": 0,
        "missing_rgb": 0,
        "missing_depth": 0,
        "constant_depth": 0,
        "excessive_zero_depth": 0,
        "rgb_depth_resolution_mismatch": 0,
        "intrinsics_resolution_mismatch": 0,
        "invalid_pose": 0,
        "metadata_missing": 0 if metadata["present"] else 1,
        "metadata_inconsistency": 1 if metadata["consistent"] is False else 0,
    }
    per_frame = []
    rgb_count = 0
    depth_count = 0
    valid_pose_count = 0
    if not frames:
        checks["missing_frames"] = 1
        warnings.append("Scene has no frame records.")

    declared_size = (int(intrinsics.get("w", 0)), int(intrinsics.get("h", 0)))
    intrinsics_present = bool(
        declared_size[0] > 0
        and declared_size[1] > 0
        and float(intrinsics.get("fl_x", 0.0)) > 0
        and float(intrinsics.get("fl_y", 0.0)) > 0
    )
    if not intrinsics_present:
        warnings.append("Camera intrinsics are incomplete or invalid.")

    for frame in frames:
        view_id = str(frame["view_id"])
        rgb_path = scene_dir / frame["rgb_path"]
        depth_path = scene_dir / frame["depth_path"] if frame.get("depth_path") is not None else None
        record: dict[str, Any] = {
            "view_id": view_id,
            "rgb_path": str(frame["rgb_path"]).replace("\\", "/"),
            "depth_path": str(frame["depth_path"]).replace("\\", "/") if frame.get("depth_path") is not None else None,
            "rgb_size": None,
            "depth_size": None,
            "depth_min": None,
            "depth_max": None,
            "zero_fraction": None,
            "depth_constant": None,
            "pose": pose_details(frame["transform_matrix"]),
            "warnings": [],
        }
        if record["pose"]["valid"]:
            valid_pose_count += 1
        else:
            checks["invalid_pose"] += 1
            record["warnings"].append("invalid pose matrix")

        if not rgb_path.exists():
            checks["missing_rgb"] += 1
            record["warnings"].append("missing RGB")
        else:
            rgb_count += 1
            try:
                record["rgb_size"] = list(image_size(rgb_path))
            except Exception as exc:
                checks["missing_rgb"] += 1
                rgb_count -= 1
                record["warnings"].append(f"unreadable RGB: {exc}")

        if depth_path is None or not depth_path.exists():
            checks["missing_depth"] += 1
            record["warnings"].append("missing depth")
        else:
            depth_count += 1
            try:
                array = depth_array(depth_path)
                record["depth_size"] = [int(array.shape[1]), int(array.shape[0])]
                finite = array[np.isfinite(array)]
                if finite.size == 0:
                    record["warnings"].append("depth has no finite values")
                    checks["excessive_zero_depth"] += 1
                else:
                    minimum = float(np.min(finite))
                    maximum = float(np.max(finite))
                    zero_fraction = float(np.count_nonzero(finite <= 0) / finite.size)
                    constant = bool(math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-7))
                    record["depth_min"] = minimum
                    record["depth_max"] = maximum
                    record["zero_fraction"] = round(zero_fraction, 6)
                    record["depth_constant"] = constant
                    if constant:
                        checks["constant_depth"] += 1
                        record["warnings"].append(f"constant depth value {minimum:g}")
                    if zero_fraction > max_zero_fraction:
                        checks["excessive_zero_depth"] += 1
                        record["warnings"].append(
                            f"zero-depth fraction {zero_fraction:.3f} exceeds {max_zero_fraction:.3f}"
                        )
            except Exception as exc:
                checks["missing_depth"] += 1
                depth_count -= 1
                record["warnings"].append(f"unreadable depth: {exc}")

        if record["rgb_size"] and record["depth_size"] and record["rgb_size"] != record["depth_size"]:
            checks["rgb_depth_resolution_mismatch"] += 1
            record["warnings"].append("RGB/depth resolution mismatch")
        if record["rgb_size"] and declared_size != tuple(record["rgb_size"]):
            checks["intrinsics_resolution_mismatch"] += 1
            record["warnings"].append(
                f"intrinsics resolution {declared_size[0]}x{declared_size[1]} does not match RGB"
            )
        per_frame.append(record)

    if checks["constant_depth"]:
        warnings.append(f"Constant depth detected in {checks['constant_depth']} frame(s).")
    if checks["missing_depth"]:
        warnings.append(f"Depth is missing or unreadable for {checks['missing_depth']} frame(s).")
    if checks["excessive_zero_depth"]:
        warnings.append(f"Excessive zero/invalid depth detected in {checks['excessive_zero_depth']} frame(s).")
    if checks["rgb_depth_resolution_mismatch"]:
        warnings.append(
            f"RGB/depth resolution mismatch in {checks['rgb_depth_resolution_mismatch']} frame(s)."
        )
    if checks["intrinsics_resolution_mismatch"]:
        warnings.append(
            f"Intrinsics/RGB resolution mismatch in {checks['intrinsics_resolution_mismatch']} frame(s)."
        )
    if checks["invalid_pose"]:
        warnings.append(f"Invalid pose matrices detected in {checks['invalid_pose']} frame(s).")
    if metadata["status"] == "missing":
        warnings.append("Metadata consistency is unavailable because no metadata file exists.")
    elif metadata["status"] == "present_unverified":
        warnings.append("Metadata exists but has no scene_id, frame_count, or path fields to verify.")
    elif metadata["status"] == "inconsistent":
        warnings.extend(metadata["issues"])

    frame_ids = [str(frame["view_id"]) for frame in frames]
    views_dir = scene_dir / "views"
    view_ids = {path.stem for path in views_dir.glob("*.json")} if views_dir.exists() else set()
    tree_manifest_path = scene_dir / "tree" / "manifest.json"
    tree_valid = False
    if tree_manifest_path.exists():
        try:
            tree_manifest = load_json(tree_manifest_path)
            root_id = tree_manifest.get("root_id")
            tree_valid = bool(root_id and (scene_dir / "tree" / f"node_{root_id}.json").exists())
        except (OSError, ValueError, json.JSONDecodeError):
            tree_valid = False
    semantic_index_complete = bool(frame_ids and set(frame_ids).issubset(view_ids) and tree_valid)
    if not semantic_index_complete:
        warnings.append("ViewJSON/tree semantic index is incomplete or missing.")

    manifest_path = scene_dir / "scene_manifest.json"
    bbox_3d_gt_available = False
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
            ground_truth = manifest.get("ground_truth")
            if isinstance(ground_truth, dict):
                bbox_3d_gt_available = bool(ground_truth.get("bbox_3d_available"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    semantic_allowed = bool(
        frames
        and rgb_count == len(frames)
        and valid_pose_count == len(frames)
        and checks["metadata_inconsistency"] == 0
        and semantic_index_complete
    )
    geometry_failures = sum(
        checks[key]
        for key in (
            "missing_frames",
            "missing_rgb",
            "missing_depth",
            "constant_depth",
            "excessive_zero_depth",
            "rgb_depth_resolution_mismatch",
            "intrinsics_resolution_mismatch",
            "invalid_pose",
            "metadata_inconsistency",
        )
    )
    geometry_inputs_valid = bool(
        frames
        and rgb_count == len(frames)
        and valid_pose_count == len(frames)
        and intrinsics_present
        and depth_count == len(frames)
        and geometry_failures == 0
    )
    three_d_allowed = bool(geometry_inputs_valid and bbox_3d_gt_available)
    if geometry_inputs_valid and not bbox_3d_gt_available:
        warnings.append("Geometry inputs are valid, but independent GT 3D boxes/masks are unavailable.")
    if semantic_allowed and not three_d_allowed:
        warnings.append("Semantic evaluation is allowed, but 3D localization metrics are N/A.")

    return {
        "schema_version": "semanticsplat.geometry_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": resolved_scene_id,
        "scene_path": display_path(scene_dir),
        "layout": layout,
        "frame_count": len(frames),
        "rgb_count": rgb_count,
        "depth_count": depth_count,
        "pose_count": len(frames),
        "valid_pose_count": valid_pose_count,
        "intrinsics": {**intrinsics, "present": intrinsics_present},
        "metadata": metadata,
        "semantic_index": {
            "view_json_count": len(view_ids),
            "tree_valid": tree_valid,
            "complete": semantic_index_complete,
        },
        "ground_truth": {
            "bbox_3d_available": bbox_3d_gt_available,
        },
        "checks": checks,
        "check_availability": {
            "rgb_count": "measured",
            "depth_count": "measured",
            "pose_count": "measured",
            "rgb_depth_resolution_match": "measured" if rgb_count and depth_count else "N/A",
            "intrinsics_resolution_match": "measured" if rgb_count and intrinsics_present else "N/A",
            "constant_depth": "measured" if depth_count else "N/A",
            "zero_depth_ratio": "measured" if depth_count else "N/A",
            "invalid_pose_matrices": "measured" if frames else "N/A",
            "missing_frames": "measured",
            "metadata_consistency": metadata["status"],
        },
        "semantic_evaluation_allowed": semantic_allowed,
        "geometry_inputs_valid": geometry_inputs_valid,
        "three_d_localization_allowed": three_d_allowed,
        "metrics_eligibility": {
            "semantic_retrieval": "allowed" if semantic_allowed else "not_allowed",
            "expected_view_node_zone": "allowed" if semantic_allowed else "not_allowed",
            "bbox_3d": "allowed" if three_d_allowed else "N/A",
            "iou_3d": "allowed" if three_d_allowed else "N/A",
        },
        "warnings": warnings,
        "per_frame": per_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RGB-D scene geometry and metric eligibility")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scene", type=Path, help="One scene directory or standalone PLY asset")
    inputs.add_argument("--scenes", type=Path, nargs="+", help="Batch of scene directories or PLY assets")
    parser.add_argument("--scene-id", help="Override scene ID in single-scene mode")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out", type=Path, help="Output directory; required for --scenes")
    parser.add_argument("--max-zero-fraction", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if semantic evaluation is blocked")
    parser.add_argument("--require-3d", action="store_true", help="Exit non-zero if 3D evaluation is blocked")
    args = parser.parse_args()
    if not 0 <= args.max_zero_fraction <= 1:
        raise SystemExit("--max-zero-fraction must be between 0 and 1")
    if args.output and args.out:
        parser.error("use --output or --out, not both")
    if args.scenes and args.scene_id:
        parser.error("--scene-id is only valid with --scene")
    if args.scenes and args.output:
        parser.error("--output is only valid with --scene; use --out for batch mode")
    if args.scenes and not args.out:
        parser.error("--out is required with --scenes")

    scene_paths = args.scenes or [args.scene]
    reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for scene_path in scene_paths:
        report = validate_scene(
            scene_path,
            scene_id=args.scene_id if args.scene else None,
            max_zero_fraction=args.max_zero_fraction,
        )
        report_scene_id = str(report["scene_id"])
        if report_scene_id in seen_ids:
            parser.error(f"duplicate scene_id would overwrite output: {report_scene_id}")
        seen_ids.add(report_scene_id)
        reports.append(report)

        destination: Path | None = None
        if args.output:
            destination = args.output
        elif args.out:
            destination = args.out / f"{report_scene_id}_geometry.json"
        if destination:
            write_json(destination, report)
            print(f"Wrote {destination}")
        elif len(scene_paths) == 1:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        print(
            f"scene={report_scene_id} layout={report['layout']} frames={report['frame_count']} "
            f"semantic={report['semantic_evaluation_allowed']} "
            f"3d={report['three_d_localization_allowed']}"
        )

    semantic_allowed_count = sum(bool(report["semantic_evaluation_allowed"]) for report in reports)
    three_d_allowed_count = sum(bool(report["three_d_localization_allowed"]) for report in reports)
    if len(reports) > 1:
        print(
            f"summary scenes={len(reports)} semantic_allowed={semantic_allowed_count} "
            f"3d_allowed={three_d_allowed_count}"
        )
    if args.strict and semantic_allowed_count != len(reports):
        raise SystemExit(2)
    if args.require_3d and three_d_allowed_count != len(reports):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
