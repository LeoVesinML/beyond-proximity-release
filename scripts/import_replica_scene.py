#!/usr/bin/env python3
"""Normalize RGB-D directories and standalone Gaussian-splat PLY assets."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_ply_header(path: Path, *, max_header_bytes: int = 1024 * 1024) -> dict[str, Any]:
    data = bytearray()
    header_end = -1
    with path.open("rb") as handle:
        while len(data) < max_header_bytes:
            chunk = handle.read(min(16384, max_header_bytes - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            marker = data.find(b"end_header")
            if marker >= 0:
                header_end = marker + len(b"end_header")
                break
    if not data.startswith(b"ply\n") and not data.startswith(b"ply\r\n"):
        raise ValueError("File does not start with a PLY signature")
    if header_end < 0:
        raise ValueError(f"PLY header exceeds {max_header_bytes} bytes or has no end_header marker")

    header_text = bytes(data[:header_end]).decode("ascii")
    lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    format_name: str | None = None
    comments: list[str] = []
    elements: dict[str, int] = {}
    properties: dict[str, list[str]] = {}
    current_element: str | None = None
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 3:
            format_name = f"{parts[1]} {parts[2]}"
        elif parts[0] == "comment":
            comments.append(line.removeprefix("comment").strip())
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            elements[current_element] = int(parts[2])
            properties.setdefault(current_element, [])
        elif parts[0] == "property" and current_element is not None:
            properties[current_element].append(parts[-1])

    vertex_properties = properties.get("vertex", [])
    semantic_tokens = ("semantic", "class", "label", "instance", "object_id", "bbox")
    semantic_properties = [
        name for name in vertex_properties if any(token in name.lower() for token in semantic_tokens)
    ]
    return {
        "valid": True,
        "format": format_name,
        "comments": comments,
        "elements": elements,
        "properties": properties,
        "header_bytes": header_end,
        "semantic_or_instance_properties": semantic_properties,
    }


def normalized_relative(value: Any) -> Path:
    text = str(value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe scene-relative path: {value}")
    return path


def valid_pose(matrix: Any) -> bool:
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        array.shape == (4, 4)
        and np.isfinite(array).all()
        and np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
    )


def detect_layout(input_dir: Path) -> str:
    if input_dir.is_file():
        if input_dir.suffix.lower() == ".ply":
            return "ply_only"
        raise ValueError(f"Unsupported scene file: {input_dir}")
    if (input_dir / "geometry" / "scene.ply").is_file():
        return "ply_backend"
    if (input_dir / "poses.json").exists() and (input_dir / "intrinsics.json").exists():
        return "replica_style"
    if (input_dir / "transforms.json").exists():
        return "backend"
    raise ValueError(
        f"Could not detect scene layout in {input_dir}; expected poses.json + intrinsics.json or transforms.json"
    )


def intrinsics_from_replica(path: Path) -> dict[str, Any]:
    data = load_json(path)
    width = data.get("width", data.get("w"))
    height = data.get("height", data.get("h"))
    return {
        "fl_x": float(data.get("fx", data.get("fl_x", 0.0))),
        "fl_y": float(data.get("fy", data.get("fl_y", data.get("fx", data.get("fl_x", 0.0))))),
        "cx": float(data.get("cx", (float(width) / 2 if width else 0.0))),
        "cy": float(data.get("cy", (float(height) / 2 if height else 0.0))),
        "w": int(width) if width is not None else 0,
        "h": int(height) if height is not None else 0,
        "camera_model": str(data.get("camera_model", "OPENCV")),
    }


def sorted_media(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def replica_frames(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    poses = load_json(input_dir / "poses.json")
    raw_frames = poses.get("frames")
    rgb_candidates = sorted_media(input_dir / "rgb", {".png", ".jpg", ".jpeg"})
    depth_candidates = sorted_media(input_dir / "depth", {".npy", ".png", ".tif", ".tiff"})

    if isinstance(raw_frames, list) and raw_frames:
        frames = []
        for index, raw in enumerate(raw_frames):
            if not isinstance(raw, dict):
                raise ValueError(f"poses.json frame {index} must be an object")
            matrix = raw.get("transform_matrix", raw.get("pose"))
            if not valid_pose(matrix):
                raise ValueError(f"Invalid 4x4 pose matrix at frame {index}")
            rgb_value = raw.get("rgb_file", raw.get("file_path", raw.get("rgb")))
            depth_value = raw.get("depth_file", raw.get("depth_file_path", raw.get("depth")))
            if rgb_value is None and index < len(rgb_candidates):
                rgb_value = rgb_candidates[index].relative_to(input_dir)
            if depth_value is None and index < len(depth_candidates):
                depth_value = depth_candidates[index].relative_to(input_dir)
            if rgb_value is None:
                raise ValueError(f"Missing RGB path for pose frame {index}")
            frames.append({
                "view_id": str(raw.get("view_id") or f"v{index + 1:03d}"),
                "rgb_path": normalized_relative(rgb_value),
                "depth_path": normalized_relative(depth_value) if depth_value is not None else None,
                "transform_matrix": matrix,
            })
        return frames, poses

    matrices: list[Any] = []
    if isinstance(poses.get("poses"), list):
        matrices = poses["poses"]
    elif all(isinstance(value, list) for value in poses.values()):
        matrices = [poses[key] for key in sorted(poses)]
    if not matrices:
        raise ValueError("poses.json must contain frames, poses, or a mapping of pose matrices")
    if len(rgb_candidates) != len(matrices):
        raise ValueError(f"Pose/RGB count mismatch: {len(matrices)} poses vs {len(rgb_candidates)} RGB files")
    if depth_candidates and len(depth_candidates) != len(matrices):
        raise ValueError(f"Pose/depth count mismatch: {len(matrices)} poses vs {len(depth_candidates)} depth files")
    frames = []
    for index, matrix in enumerate(matrices):
        if not valid_pose(matrix):
            raise ValueError(f"Invalid 4x4 pose matrix at frame {index}")
        frames.append({
            "view_id": f"v{index + 1:03d}",
            "rgb_path": rgb_candidates[index].relative_to(input_dir),
            "depth_path": depth_candidates[index].relative_to(input_dir) if depth_candidates else None,
            "transform_matrix": matrix,
        })
    return frames, poses


def backend_frames(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transforms = load_json(input_dir / "transforms.json")
    frames = []
    for index, raw in enumerate(transforms.get("frames", [])):
        if not isinstance(raw, dict):
            raise ValueError(f"transforms.json frame {index} must be an object")
        matrix = raw.get("transform_matrix")
        if not valid_pose(matrix):
            raise ValueError(f"Invalid 4x4 pose matrix at frame {index}")
        rgb_value = raw.get("file_path")
        if not rgb_value:
            raise ValueError(f"Missing file_path at backend frame {index}")
        depth_value = raw.get("depth_file_path")
        frames.append({
            "view_id": str(raw.get("view_id") or f"v{index + 1:03d}"),
            "rgb_path": normalized_relative(rgb_value),
            "depth_path": normalized_relative(depth_value) if depth_value else None,
            "transform_matrix": matrix,
        })
    return frames, transforms


def copy_rgb(source: Path, destination: Path) -> tuple[int, int]:
    if not source.exists():
        raise FileNotFoundError(f"Missing RGB file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        rgb.save(destination)
    return int(size[0]), int(size[1])


def copy_depth(source: Path, destination: Path, depth_scale: float) -> tuple[int, int]:
    if not source.exists():
        raise FileNotFoundError(f"Missing depth file: {source}")
    if source.suffix.lower() == ".npy":
        array = np.asarray(np.load(source), dtype=np.float32)
    else:
        with Image.open(source) as image:
            array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Depth must be a 2D array: {source} has shape {array.shape}")
    array = array * float(depth_scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array)
    return int(array.shape[1]), int(array.shape[0])


def manifest_for(
    *,
    scene_id: str,
    dataset: str,
    source: Path,
    output: Path,
    intrinsics: dict[str, Any],
    frame_count: int,
    depth_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "semanticsplat.scene_manifest.v1",
        "scene_id": scene_id,
        "dataset": dataset,
        "source_description": f"Imported from local path {display_path(source)}",
        "paths": {
            "rgb": "images",
            "depth": "depths",
            "poses": "transforms.json",
            "intrinsics": "transforms.json",
            "semantic_index": "views" if (output / "views").exists() else None,
        },
        "frame_count": frame_count,
        "coordinate_frame": "camera_to_world",
        "intrinsics": {
            "width": intrinsics["w"],
            "height": intrinsics["h"],
            "fx": intrinsics["fl_x"],
            "fy": intrinsics["fl_y"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
        },
        "depth_validation": {
            "status": "unchecked" if depth_count else "missing",
            "unit": "meters",
            "scale_to_meters": 1.0,
            "checked_frame_count": 0,
            "constant_frame_count": 0,
            "invalid_value_fraction": None,
            "reason": "Run scripts/validate_scene_geometry.py after import.",
        },
        "ground_truth": {
            "zone_labels_available": False,
            "object_labels_available": False,
            "bbox_3d_available": False,
            "source": None,
        },
        "warnings": warnings,
    }


def ply_manifest_for(
    *,
    scene_id: str,
    dataset: str,
    source: Path,
    ply: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "semanticsplat.scene_manifest.v1",
        "scene_id": scene_id,
        "dataset": dataset,
        "source_description": (
            f"Imported PLY-only SuperSplat asset from {display_path(source)}; "
            "original dataset provenance is unavailable."
        ),
        "paths": {
            "gaussian_splat": "geometry/scene.ply",
            "rgb": None,
            "depth": None,
            "poses": None,
            "intrinsics": None,
            "semantic_index": None,
        },
        "frame_count": 0,
        "coordinate_frame": "unknown",
        "intrinsics": {
            "present": False,
            "width": 0,
            "height": 0,
            "fx": 0.0,
            "fy": 0.0,
            "cx": 0.0,
            "cy": 0.0,
        },
        "depth_validation": {
            "status": "missing",
            "unit": "unknown",
            "scale_to_meters": None,
            "checked_frame_count": 0,
            "constant_frame_count": 0,
            "invalid_value_fraction": None,
            "reason": "No depth frames were supplied with the PLY asset.",
        },
        "ground_truth": {
            "zone_labels_available": False,
            "object_labels_available": False,
            "bbox_3d_available": False,
            "source": None,
        },
        "evaluation_eligibility": {
            "semantic_evaluation_allowed": False,
            "three_d_evaluation_allowed": False,
            "reason": "RGB-D frames, camera calibration, semantic index, and independent GT are unavailable.",
        },
        "ply": {
            "format": ply["format"],
            "comments": ply["comments"],
            "elements": ply["elements"],
            "properties": ply["properties"],
            "semantic_or_instance_properties": ply["semantic_or_instance_properties"],
        },
        "warnings": warnings,
    }


def import_ply_scene(
    input_file: Path,
    output_dir: Path,
    *,
    scene_id: str,
    dataset: str,
    overwrite: bool,
) -> dict[str, Any]:
    input_file = input_file.resolve()
    output_dir = output_dir.resolve()
    if input_file.suffix.lower() != ".ply" or not input_file.is_file():
        raise ValueError(f"Expected a standalone PLY file: {input_file}")
    ply = read_ply_header(input_file)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output scene is not empty; use --overwrite explicitly: {output_dir}")

    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    destination = geometry_dir / "scene.ply"
    if input_file == destination.resolve():
        raise ValueError("Input and output PLY paths must differ")
    shutil.copy2(input_file, destination)

    warnings = [
        "PLY-only import: RGB frames are unavailable.",
        "PLY-only import: depth, poses, and intrinsics are unavailable.",
        "PLY-only import: semantic labels, instance labels, and GT 3D boxes are unavailable.",
        "Original dataset provenance is unavailable; dataset identity was not inferred from the filename.",
    ]
    source_metadata = {
        "schema_version": "semanticsplat.source_metadata.v1",
        "scene_id": scene_id,
        "dataset": dataset,
        "source_layout": "ply_only",
        "source_path": display_path(input_file),
        "source_filename": input_file.name,
        "original_metadata_available": False,
        "generated_by_importer": True,
        "ply": {
            "format": ply["format"],
            "comments": ply["comments"],
            "elements": ply["elements"],
            "properties": ply["properties"],
        },
        "warnings": warnings,
    }
    write_json(output_dir / "source_metadata.json", source_metadata)
    manifest = ply_manifest_for(
        scene_id=scene_id,
        dataset=dataset,
        source=input_file,
        ply=ply,
        warnings=warnings,
    )
    write_json(output_dir / "scene_manifest.json", manifest)
    return manifest


def import_scene(
    input_dir: Path,
    output_dir: Path,
    *,
    scene_id: str,
    dataset: str,
    depth_scale: float,
    copy_semantic_index: bool,
    overwrite: bool,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    layout = detect_layout(input_dir)
    if layout == "ply_only":
        return import_ply_scene(
            input_dir,
            output_dir,
            scene_id=scene_id,
            dataset=dataset,
            overwrite=overwrite,
        )
    if layout == "ply_backend":
        return import_ply_scene(
            input_dir / "geometry" / "scene.ply",
            output_dir,
            scene_id=scene_id,
            dataset=dataset,
            overwrite=overwrite,
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output scene is not empty; use --overwrite explicitly: {output_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output must differ; use the existing backend scene directly")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "depths").mkdir(exist_ok=True)

    if layout == "replica_style":
        frames, source_metadata = replica_frames(input_dir)
        intrinsics = intrinsics_from_replica(input_dir / "intrinsics.json")
    else:
        frames, source_metadata = backend_frames(input_dir)
        intrinsics = {
            "fl_x": float(source_metadata.get("fl_x", 0.0)),
            "fl_y": float(source_metadata.get("fl_y", source_metadata.get("fl_x", 0.0))),
            "cx": float(source_metadata.get("cx", 0.0)),
            "cy": float(source_metadata.get("cy", 0.0)),
            "w": int(source_metadata.get("w", 0)),
            "h": int(source_metadata.get("h", 0)),
            "camera_model": str(source_metadata.get("camera_model", "OPENCV")),
        }

    backend_frames = []
    warnings: list[str] = []
    depth_count = 0
    observed_rgb_sizes: set[tuple[int, int]] = set()
    observed_depth_sizes: set[tuple[int, int]] = set()
    seen_ids: set[str] = set()
    for frame in frames:
        view_id = frame["view_id"]
        if view_id in seen_ids:
            raise ValueError(f"Duplicate view_id: {view_id}")
        seen_ids.add(view_id)
        rgb_destination = output_dir / "images" / f"{view_id}.png"
        observed_rgb_sizes.add(copy_rgb(input_dir / frame["rgb_path"], rgb_destination))
        backend_frame = {
            "file_path": f"images/{view_id}.png",
            "view_id": view_id,
            "transform_matrix": frame["transform_matrix"],
        }
        if frame["depth_path"] is not None:
            depth_destination = output_dir / "depths" / f"{view_id}_depth.npy"
            observed_depth_sizes.add(copy_depth(input_dir / frame["depth_path"], depth_destination, depth_scale))
            backend_frame["depth_file_path"] = f"depths/{view_id}_depth.npy"
            depth_count += 1
        backend_frames.append(backend_frame)

    if len(observed_rgb_sizes) == 1:
        observed_w, observed_h = next(iter(observed_rgb_sizes))
        if not intrinsics["w"] or not intrinsics["h"]:
            intrinsics["w"], intrinsics["h"] = observed_w, observed_h
    if observed_rgb_sizes and (intrinsics["w"], intrinsics["h"]) not in observed_rgb_sizes:
        warnings.append("Intrinsics resolution does not match imported RGB resolution; geometry validation required.")
    if observed_depth_sizes and observed_depth_sizes != observed_rgb_sizes:
        warnings.append("RGB/depth resolution sets differ; 3D evaluation is blocked until corrected.")
    if depth_count != len(backend_frames):
        warnings.append(f"Depth is missing for {len(backend_frames) - depth_count} frames.")

    transforms = {**intrinsics, "frames": backend_frames}
    write_json(output_dir / "transforms.json", transforms)
    source_meta_path = input_dir / "metadata.json"
    if source_meta_path.exists():
        shutil.copy2(source_meta_path, output_dir / "source_metadata.json")
    else:
        write_json(output_dir / "source_metadata.json", {
            "source_layout": layout,
            "source_path": display_path(input_dir),
            "warnings": warnings,
        })

    if copy_semantic_index and layout == "backend":
        for name in ("views", "tree"):
            source_dir = input_dir / name
            destination_dir = output_dir / name
            if source_dir.exists():
                if destination_dir.exists() and overwrite:
                    shutil.rmtree(destination_dir)
                shutil.copytree(source_dir, destination_dir, dirs_exist_ok=overwrite)

    manifest = manifest_for(
        scene_id=scene_id,
        dataset=dataset,
        source=input_dir,
        output=output_dir,
        intrinsics=intrinsics,
        frame_count=len(backend_frames),
        depth_count=depth_count,
        warnings=warnings,
    )
    write_json(output_dir / "scene_manifest.json", manifest)
    return manifest


def discover_batch_inputs(batch_root: Path) -> list[Path]:
    batch_root = batch_root.resolve()
    if not batch_root.is_dir():
        raise ValueError(f"Batch input must be a directory: {batch_root}")
    discovered: list[Path] = []
    for candidate in sorted(batch_root.iterdir(), key=lambda path: path.name.lower()):
        if candidate.is_file() and candidate.suffix.lower() == ".ply":
            discovered.append(candidate)
        elif candidate.is_dir():
            try:
                detect_layout(candidate)
            except ValueError:
                continue
            discovered.append(candidate)
    if not discovered:
        raise ValueError(f"No importable PLY or RGB-D scenes found in {batch_root}")
    return discovered


def default_dataset_for(source: Path) -> str:
    return "unknown_supersplat_export" if detect_layout(source) in {"ply_only", "ply_backend"} else "replica"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize RGB-D or PLY scenes into backend scene format")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One RGB-D scene directory or standalone PLY")
    inputs.add_argument("--batch", type=Path, help="Directory containing importable scenes")
    parser.add_argument("--scene-id", help="Scene ID override in single-input mode")
    parser.add_argument("--output", type=Path, help="Exact output directory, or output root in batch mode")
    parser.add_argument("--backend-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", help="Dataset/source identifier; defaults honestly by detected layout")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Multiply source depth values by this factor")
    parser.add_argument("--copy-semantic-index", action="store_true", help="Copy views/tree from backend-format input")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise SystemExit("--depth-scale must be a positive finite number")
    if args.batch and args.scene_id:
        parser.error("--scene-id is only valid with --input")

    canonical_root = args.backend_root.resolve() / "backend" / "data" / "scenes"
    if args.batch:
        output_root = args.output.resolve() if args.output else canonical_root
        sources = discover_batch_inputs(args.batch)
        seen_ids: set[str] = set()
        for source in sources:
            scene_id = source.stem if source.is_file() else source.name
            if scene_id in seen_ids:
                parser.error(f"duplicate scene_id in batch: {scene_id}")
            seen_ids.add(scene_id)
            output = output_root / scene_id
            dataset = args.dataset or default_dataset_for(source)
            manifest = import_scene(
                source,
                output,
                scene_id=scene_id,
                dataset=dataset,
                depth_scale=args.depth_scale,
                copy_semantic_index=args.copy_semantic_index,
                overwrite=args.overwrite,
            )
            print(
                f"Imported scene={scene_id} layout={detect_layout(source)} "
                f"frames={manifest['frame_count']} output={display_path(output)}"
            )
        print(f"summary imported={len(sources)} output_root={display_path(output_root)}")
        return

    source = args.input
    layout = detect_layout(source)
    scene_id = args.scene_id or (source.stem if source.is_file() else None)
    if not scene_id:
        parser.error("--scene-id is required for directory input")
    output = args.output.resolve() if args.output else canonical_root / scene_id
    dataset = args.dataset or default_dataset_for(source)
    manifest = import_scene(
        source,
        output,
        scene_id=scene_id,
        dataset=dataset,
        depth_scale=args.depth_scale,
        copy_semantic_index=args.copy_semantic_index,
        overwrite=args.overwrite,
    )
    print(f"Imported scene={scene_id} layout={layout} frames={manifest['frame_count']} output={display_path(output)}")
    print(f"Manifest: {display_path(output / 'scene_manifest.json')}")


if __name__ == "__main__":
    main()
