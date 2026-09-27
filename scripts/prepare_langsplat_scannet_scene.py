#!/usr/bin/env python3
"""Convert a SemanticSplat ScanNet RGB-D subset to LangSplat/COLMAP text layout."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def rotmat_to_qvec(matrix: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = matrix.flat
    k = np.array([
        [rxx - ryy - rzz, ryx + rxy, rzx + rxz, ryz - rzy],
        [ryx + rxy, ryy - rxx - rzz, rzy + ryz, rzx - rxz],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, rxy - ryx],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    values, vectors = np.linalg.eigh(k)
    qvec = vectors[[3, 0, 1, 2], np.argmax(values)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-points", type=int, default=100_000)
    args = parser.parse_args()
    if args.point_stride < 1 or args.voxel_size <= 0 or args.max_points < 1:
        raise SystemExit("Invalid point sampling settings")

    transforms_path = args.scene / "transforms.json"
    source_metadata_path = args.scene / "source_metadata.json"
    if not transforms_path.is_file() or not source_metadata_path.is_file():
        raise SystemExit("LangSplat public conversion requires transforms and source metadata")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("source_dataset") != "ScanNet":
        raise SystemExit("This converter accepts official ScanNet imports only")
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("Scene contains no frames")

    images_dir = args.out / "images"
    input_dir = args.out / "input"
    depth_dir = args.out / "depth"
    sparse_dir = args.out / "sparse" / "0"
    for directory in (images_dir, input_dir, depth_dir, sparse_dir):
        directory.mkdir(parents=True, exist_ok=True)

    width, height = int(transforms["w"]), int(transforms["h"])
    fx, fy = float(transforms["fl_x"]), float(transforms["fl_y"])
    cx, cy = float(transforms["cx"]), float(transforms["cy"])
    (sparse_dir / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n",
        encoding="utf-8",
    )

    image_rows: list[str] = []
    manifest_frames: list[dict[str, object]] = []
    voxels: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for image_id, frame in enumerate(frames, start=1):
        view_id = str(frame.get("view_id") or f"v{image_id:03d}")
        source_image = args.scene / str(frame["file_path"])
        source_depth = args.scene / str(frame["depth_file_path"])
        if not source_image.is_file() or not source_depth.is_file():
            raise SystemExit(f"Missing RGB-D source for {view_id}")
        image = Image.open(source_image).convert("RGB")
        if image.size != (width, height):
            raise SystemExit(f"Unexpected image size for {view_id}: {image.size}")
        name = f"{view_id}.jpg"
        image.save(images_dir / name, quality=95)
        shutil.copy2(images_dir / name, input_dir / name)
        depth = np.load(source_depth).astype(np.float32)
        if depth.shape != (height, width):
            raise SystemExit(f"RGB/depth resolution mismatch for {view_id}")
        np.save(depth_dir / f"{view_id}.npy", depth)

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise SystemExit(f"Invalid pose for {view_id}")
        w2c = np.linalg.inv(c2w)
        qvec = rotmat_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        image_rows.append(
            f"{image_id} {' '.join(f'{v:.12g}' for v in qvec)} "
            f"{' '.join(f'{v:.12g}' for v in tvec)} 1 {name}\n\n"
        )

        rgb = np.asarray(image)
        vv, uu = np.mgrid[0:height:args.point_stride, 0:width:args.point_stride]
        zz = depth[vv, uu]
        valid = np.isfinite(zz) & (zz > 0.1) & (zz < 10.0)
        uu, vv, zz = uu[valid], vv[valid], zz[valid]
        camera_points = np.column_stack(((uu - cx) * zz / fx, (vv - cy) * zz / fy, zz))
        world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        colors = rgb[vv, uu]
        for point, color in zip(world_points, colors):
            key = tuple(np.floor(point / args.voxel_size).astype(np.int64).tolist())
            voxels.setdefault(key, (point, color))
        manifest_frames.append({
            "view_id": view_id,
            "image_name": name,
            "source_image": str(source_image),
            "source_depth": str(source_depth),
            "camera_to_world": c2w.tolist(),
        })

    (sparse_dir / "images.txt").write_text("".join(image_rows), encoding="utf-8")
    points = list(voxels.values())[: args.max_points]
    point_rows = [
        f"{index} {point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
        f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
        for index, (point, color) in enumerate(points, start=1)
    ]
    (sparse_dir / "points3D.txt").write_text("".join(point_rows), encoding="utf-8")
    manifest = {
        "schema_version": "semanticsplat.langsplat_scannet_conversion.v1",
        "scene_id": args.scene.name,
        "source_dataset": "ScanNet",
        "source_metadata": str(source_metadata_path),
        "frame_count": len(manifest_frames),
        "point_count": len(points),
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "point_stride": args.point_stride,
        "voxel_size_m": args.voxel_size,
        "pose_convention": "official ScanNet OpenCV camera-to-world converted to COLMAP world-to-camera",
        "frames": manifest_frames,
    }
    (args.out / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared LangSplat ScanNet scene: {args.out}")
    print(f"Frames: {len(manifest_frames)}; initial points: {len(points)}")


if __name__ == "__main__":
    main()
