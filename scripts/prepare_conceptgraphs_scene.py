#!/usr/bin/env python3
"""Convert a captured SemanticSplat RGB-D scene to ConceptGraphs Azure layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    transforms_path = args.scene / "transforms.json"
    if not transforms_path.is_file():
        raise SystemExit(f"Missing transforms.json: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    source_metadata_path = args.scene / "source_metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    source_warning = str(source_metadata.get("warning") or "").lower()
    if "no rendered rgb-d camera trajectory" in source_warning:
        raise SystemExit(
            f"{args.scene.name} is mesh-level GT only, not a native RGB-D trajectory"
        )
    if int(transforms.get("w", 0)) <= 4 or int(transforms.get("h", 0)) <= 4:
        raise SystemExit(
            f"{args.scene.name} has placeholder camera dimensions; refusing baseline conversion"
        )
    frames = transforms.get("frames", [])[: args.limit]
    if not frames:
        raise SystemExit("Scene contains no frames")

    scene_id = args.scene.name
    scene_out = args.out / scene_id
    color_out = scene_out / "color"
    depth_out = scene_out / "depth"
    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    pose_lines: list[str] = []
    manifest_frames: list[dict] = []
    for index, frame in enumerate(frames):
        view_id = str(frame.get("view_id") or f"v{index + 1:03d}")
        image_path = args.scene / frame["file_path"]
        depth_path = args.scene / frame["depth_file_path"]
        if not image_path.is_file() or not depth_path.is_file():
            raise SystemExit(f"Missing RGB-D source for {view_id}")

        image = Image.open(image_path).convert("RGB")
        image.save(color_out / f"{view_id}.jpg", quality=95)
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (image.height, image.width):
            raise SystemExit(f"RGB/depth resolution mismatch for {view_id}")
        valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.535)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(np.rint(depth[valid] * 1000.0), 1, 65535).astype(
            np.uint16
        )
        if np.unique(depth_mm[valid]).size < 2:
            raise SystemExit(f"Depth is constant or empty for {view_id}")
        Image.fromarray(depth_mm).save(depth_out / f"{view_id}.png")

        pose = np.asarray(frame["transform_matrix"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise SystemExit(f"Invalid pose for {view_id}")
        pose_lines.append(" ".join(f"{value:.12g}" for value in pose.reshape(-1)))
        manifest_frames.append({
            "view_id": view_id,
            "source_image": str(image_path),
            "source_depth": str(depth_path),
            "valid_depth_ratio": float(valid.mean()),
        })

    (scene_out / "poses_global_dvo.txt").write_text(
        "\n".join(pose_lines) + "\n", encoding="utf-8"
    )
    config = {
        "dataset_name": "azure",
        "camera_params": {
            "image_height": int(transforms["h"]),
            "image_width": int(transforms["w"]),
            "fx": float(transforms["fl_x"]),
            "fy": float(transforms["fl_y"]),
            "cx": float(transforms["cx"]),
            "cy": float(transforms["cy"]),
            "png_depth_scale": 1000.0,
            "crop_edge": 0,
        },
    }
    config_path = args.out / "captured_scene.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    source_dataset = source_metadata.get("source_dataset")
    pose_convention = (
        f"official {source_dataset} OpenCV camera-to-world"
        if source_dataset in {"ScanNet", "Replica"}
        else "source camera-to-world; camera-axis convention documented by source scene"
    )
    (scene_out / "conversion_manifest.json").write_text(
        json.dumps({
            "adapter": "semanticsplat_capture_to_conceptgraphs_azure_v1",
            "scene_id": scene_id,
            "frame_count": len(manifest_frames),
            "frames": manifest_frames,
            "pose_convention": pose_convention,
            "source_metadata": str(source_metadata_path) if source_metadata_path.is_file() else None,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest_frames)} ConceptGraphs frame(s) at {scene_out}")
    print(f"Dataset config: {config_path}")


if __name__ == "__main__":
    main()
