"""API routes: validated view capture from the browser R-key workflow."""
from __future__ import annotations

import base64
import io
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from backend.config import DATA_DIR
from backend.io.nerfstudio import FrameEntry, get_or_create_transforms, save_transforms
from backend.schemas.types import CapturePayload

router = APIRouter()


def _pose_is_valid(matrix: list[list[float]]) -> bool:
    try:
        pose = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return False
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        return False
    rotation = pose[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-3
    )


def _same_calibration(scene: Any, payload: CapturePayload) -> bool:
    return bool(
        scene.w == payload.width
        and scene.h == payload.height
        and np.allclose(
            [scene.fl_x, scene.fl_y, scene.cx, scene.cy],
            [payload.fl_x, payload.fl_y, payload.cx, payload.cy],
            rtol=1e-5,
            atol=1e-4,
        )
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _refresh_existing_manifest(scene_dir: Path, frame_count: int) -> None:
    path = scene_dir / "scene_manifest.json"
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    paths = manifest.setdefault("paths", {})
    paths.update({
        "rgb": "images",
        "depth": "depths",
        "poses": "transforms.json",
        "intrinsics": "transforms.json",
    })
    manifest["frame_count"] = frame_count
    eligibility = manifest.setdefault("evaluation_eligibility", {})
    eligibility["semantic_evaluation_allowed"] = False
    eligibility["three_d_evaluation_allowed"] = False
    eligibility["reason"] = (
        "Manual captures exist but require geometry validation, semantic index, and independent GT."
    )
    _write_json(path, manifest)


@router.post("/save")
def save_capture(payload: CapturePayload):
    scene_dir = DATA_DIR / payload.scene_id
    if not scene_dir.exists():
        raise HTTPException(status_code=404, detail=f"Scene '{payload.scene_id}' not found")

    if not _pose_is_valid(payload.transform_matrix):
        raise HTTPException(status_code=422, detail="Invalid camera-to-world pose matrix")
    calibration = [payload.fl_x, payload.fl_y, payload.cx, payload.cy]
    if not np.isfinite(calibration).all() or payload.fl_x <= 0 or payload.fl_y <= 0:
        raise HTTPException(status_code=422, detail="Invalid camera intrinsics")

    try:
        image_bytes = base64.b64decode(payload.rgb_b64.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            image_size = image.size
            image_format = image.format
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid RGB PNG: {exc}") from exc
    if image_format != "PNG":
        raise HTTPException(status_code=422, detail=f"RGB capture must be PNG, got {image_format}")
    if image_size != (payload.width, payload.height):
        raise HTTPException(
            status_code=422,
            detail=(
                f"RGB size mismatch: PNG is {image_size[0]}x{image_size[1]}, "
                f"payload is {payload.width}x{payload.height}"
            ),
        )

    try:
        depth_raw = base64.b64decode(payload.depth_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid depth payload: {exc}") from exc
    expected_bytes = payload.width * payload.height * 4
    if len(depth_raw) != expected_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"Depth byte-size mismatch: got {len(depth_raw)}, expected {expected_bytes}",
        )
    n_floats = len(depth_raw) // 4
    depth = np.array(struct.unpack(f"<{n_floats}f", depth_raw), dtype=np.float32)
    depth = depth.reshape(payload.height, payload.width)
    positive = depth[np.isfinite(depth) & (depth > 0)]
    if positive.size == 0:
        raise HTTPException(status_code=422, detail="Depth has no positive finite renderer values")
    depth_min = float(np.min(positive))
    depth_max = float(np.max(positive))
    depth_mean = float(np.mean(positive))
    if np.isclose(depth_min, depth_max, rtol=0.0, atol=1e-6):
        raise HTTPException(
            status_code=422,
            detail=f"Depth is constant ({depth_min:g}); capture rejected",
        )
    zero_depth_ratio = float(np.count_nonzero(~np.isfinite(depth) | (depth <= 0)) / depth.size)

    tf_path = scene_dir / "transforms.json"
    transforms = get_or_create_transforms(
        str(tf_path),
        fl_x=payload.fl_x,
        fl_y=payload.fl_y,
        cx=payload.cx,
        cy=payload.cy,
        w=payload.width,
        h=payload.height,
    )
    if transforms.frames and not _same_calibration(transforms, payload):
        raise HTTPException(
            status_code=409,
            detail="Capture calibration differs from existing frames; use a new scene ID or fixed render size",
        )
    if not transforms.frames:
        # Scene initialization may have written placeholder defaults. The first
        # real capture establishes the actual renderer calibration.
        transforms.fl_x = payload.fl_x
        transforms.fl_y = payload.fl_y
        transforms.cx = payload.cx
        transforms.cy = payload.cy
        transforms.w = payload.width
        transforms.h = payload.height

    images_dir = scene_dir / "images"
    depths_dir = scene_dir / "depths"
    images_dir.mkdir(exist_ok=True)
    depths_dir.mkdir(exist_ok=True)
    image_filename = f"{payload.view_id}.png"
    depth_filename = f"{payload.view_id}_depth.npy"
    image_path = images_dir / image_filename
    depth_path = depths_dir / depth_filename
    image_path.write_bytes(image_bytes)
    np.save(str(depth_path), depth)

    transforms.frames = [frame for frame in transforms.frames if frame.view_id != payload.view_id]
    transforms.add_frame(FrameEntry(
        file_path=f"images/{image_filename}",
        transform_matrix=payload.transform_matrix,
        view_id=payload.view_id,
        depth_file_path=f"depths/{depth_filename}",
    ))
    save_transforms(transforms, str(tf_path))

    captured_at = payload.captured_at_utc or datetime.now(timezone.utc).isoformat()
    metadata_path = scene_dir / "captures" / f"{payload.view_id}.json"
    _write_json(metadata_path, {
        "schema_version": "semanticsplat.capture.v1",
        "scene_id": payload.scene_id,
        "frame_id": payload.view_id,
        "captured_at_utc": captured_at,
        "source_ply_url": payload.source_ply_url,
        "image_path": f"images/{image_filename}",
        "depth_path": f"depths/{depth_filename}",
        "pose_reference": {
            "path": "transforms.json",
            "frame_id": payload.view_id,
            "field": "frames[].transform_matrix",
            "coordinate_convention": payload.pose_coordinate_convention,
            "source_ply_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "source_ply_alignment_status": "display_transform_documented_metric_alignment_unverified",
        },
        "intrinsics_reference": {
            "path": "transforms.json",
            "scope": "global",
            "fl_x": payload.fl_x,
            "fl_y": payload.fl_y,
            "cx": payload.cx,
            "cy": payload.cy,
            "width": payload.width,
            "height": payload.height,
        },
        "depth": {
            "source": payload.depth_source,
            "unit": "scene_units",
            "near": payload.depth_near,
            "far": payload.depth_far,
            "positive_finite_count": int(positive.size),
            "minimum": depth_min,
            "maximum": depth_max,
            "mean": depth_mean,
            "zero_or_invalid_ratio": zero_depth_ratio,
            "constant": False,
        },
    })
    _refresh_existing_manifest(scene_dir, len(transforms.frames))

    return {
        "status": "saved",
        "view_id": payload.view_id,
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "metadata_path": str(metadata_path),
        "total_frames": len(transforms.frames),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "zero_depth_ratio": zero_depth_ratio,
    }
