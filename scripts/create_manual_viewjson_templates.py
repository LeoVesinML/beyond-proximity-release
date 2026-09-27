#!/usr/bin/env python3
"""Create empty manual semantic-annotation templates for captured RGB-D views."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "semanticsplat.captured_view_json.v1"
VIEW_ID_PATTERN = re.compile(r"^v[0-9]+$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TEMPLATE_WARNING = (
    "Manual annotation template only; semantic content must be filled from the "
    "referenced image before semantic execution."
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def scene_relative_path(scene_dir: Path, raw_value: Any, field_name: str) -> tuple[Path, str]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Missing {field_name} path")
    relative = Path(raw_value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {field_name} path: {raw_value}")
    resolved = (scene_dir / relative).resolve()
    try:
        normalized = resolved.relative_to(scene_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field_name} path escapes scene directory: {raw_value}") from exc
    return resolved, normalized


def captured_frames(scene_dir: Path) -> list[dict[str, Any]]:
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")
    transforms = load_json(transforms_path)
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json must contain a non-empty frames list")

    records: list[dict[str, Any]] = []
    seen_view_ids: set[str] = set()
    referenced_images: set[str] = set()
    for index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"Frame {index} is not a JSON object")
        view_id = raw_frame.get("view_id")
        if not isinstance(view_id, str) or not VIEW_ID_PATTERN.fullmatch(view_id):
            raise ValueError(f"Frame {index} has invalid view_id: {view_id!r}")
        if view_id in seen_view_ids:
            raise ValueError(f"Duplicate view_id in transforms.json: {view_id}")
        seen_view_ids.add(view_id)

        image_file, image_path = scene_relative_path(scene_dir, raw_frame.get("file_path"), "image")
        depth_file, depth_path = scene_relative_path(
            scene_dir, raw_frame.get("depth_file_path"), "depth"
        )
        if not image_file.is_file():
            raise FileNotFoundError(f"Missing captured RGB for {view_id}: {image_file}")
        if not depth_file.is_file():
            raise FileNotFoundError(f"Missing captured depth for {view_id}: {depth_file}")
        if raw_frame.get("transform_matrix") is None:
            raise ValueError(f"Missing transform_matrix for {view_id}")

        referenced_images.add(image_path)
        records.append(
            {
                "view_id": view_id,
                "image_path": image_path,
                "depth_path": depth_path,
                "pose_ref": f"transforms.json#/frames/{index}/transform_matrix",
            }
        )

    images_dir = scene_dir / "images"
    disk_images = {
        path.resolve().relative_to(scene_dir.resolve()).as_posix()
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    } if images_dir.exists() else set()
    unreferenced = sorted(disk_images - referenced_images)
    if unreferenced:
        raise ValueError(
            "Captured RGB files are not represented in transforms.json: " + ", ".join(unreferenced)
        )
    if len(disk_images) != len(records):
        raise ValueError(
            f"RGB/frame count mismatch: images={len(disk_images)} frames={len(records)}"
        )
    return records


def template_for(scene_id: str, frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "view_id": frame["view_id"],
        "scene_id": scene_id,
        "image_path": frame["image_path"],
        "depth_path": frame["depth_path"],
        "pose_ref": frame["pose_ref"],
        "intrinsics_ref": "transforms.json",
        "semantic_index_mode": "manual",
        "summary": "",
        "visible_regions": [],
        "visible_objects": [],
        "landmarks": [],
        "free_text_notes": "",
        "warnings": [TEMPLATE_WARNING],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_templates(scene_dir: Path, out_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    frames = captured_frames(scene_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for frame in frames:
        output_path = out_dir / f"{frame['view_id']}.json"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        write_json(output_path, template_for(scene_dir.name, frame))
        generated += 1

    return {
        "scene_id": scene_dir.name,
        "scene": str(scene_dir),
        "output": str(out_dir.resolve()),
        "captured_view_count": len(frames),
        "generated_count": generated,
        "skipped_existing_count": skipped,
        "semantic_index_mode": "manual",
        "semantic_content_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create empty manual ViewJSON templates from a captured scene"
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files, including manual annotations; use only intentionally",
    )
    args = parser.parse_args()

    try:
        result = create_templates(args.scene, args.out, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Manual ViewJSON template generation failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
