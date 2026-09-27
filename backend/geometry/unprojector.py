"""Unproject 2D bounding boxes to 3D using depth maps and camera intrinsics/extrinsics."""
from typing import List, Optional, Tuple

import numpy as np

from backend.schemas.types import BBox3D
from backend.config import DEPTH_NEAR, DEPTH_FAR


def unproject_point(
    u: float,
    v: float,
    depth: float,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
) -> np.ndarray:
    """Unproject a single pixel (u, v) with given depth to world coordinates.

    Args:
        u, v: pixel coordinates (float, origin top-left)
        depth: linear depth in metres
        fl_x, fl_y: focal lengths in pixels
        cx, cy: principal point in pixels
        c2w: 4×4 camera-to-world matrix (numpy)

    Returns:
        world point as (3,) array
    """
    # Camera-space point
    x_c = (u - cx) * depth / fl_x
    y_c = (v - cy) * depth / fl_y
    z_c = depth
    p_cam = np.array([x_c, y_c, z_c, 1.0])
    p_world = c2w @ p_cam
    return p_world[:3]


def depth_stats_in_bbox(
    depth_map: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Return (median, std) of depth values inside a normalised bounding box."""
    px1 = int(x1 * width)
    py1 = int(y1 * height)
    px2 = int(x2 * width)
    py2 = int(y2 * height)
    px1, px2 = max(0, px1), min(width - 1, px2)
    py1, py2 = max(0, py1), min(height - 1, py2)
    patch = depth_map[py1:py2 + 1, px1:px2 + 1]
    valid = patch[(patch > DEPTH_NEAR) & (patch < DEPTH_FAR)]
    if len(valid) == 0:
        return 0.0, 0.0
    return float(np.median(valid)), float(np.std(valid))


def bbox_2d_to_3d(
    bbox_norm: Tuple[float, float, float, float],
    depth_map: np.ndarray,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    transform_matrix: List[List[float]],
    width: int,
    height: int,
) -> Optional[BBox3D]:
    """Convert a normalised 2D bbox + depth map to a 3D BBox3D.

    Returns None if depth data is unavailable.
    """
    x1_n, y1_n, x2_n, y2_n = bbox_norm
    c2w = np.array(transform_matrix)

    # Sample depth at the four corners and centre
    pts_norm = [
        ((x1_n + x2_n) / 2, (y1_n + y2_n) / 2),  # centre
        (x1_n, y1_n), (x2_n, y1_n),
        (x1_n, y2_n), (x2_n, y2_n),
    ]

    depth_med, _ = depth_stats_in_bbox(depth_map, x1_n, y1_n, x2_n, y2_n, width, height)
    if depth_med <= 0:
        return None

    world_pts = []
    for u_n, v_n in pts_norm:
        u = u_n * width
        v = v_n * height
        # Approximate depth for corner points using the median
        d = float(depth_map[
            min(int(v_n * height), height - 1),
            min(int(u_n * width), width - 1),
        ])
        if d <= DEPTH_NEAR or d >= DEPTH_FAR:
            d = depth_med
        world_pts.append(unproject_point(u, v, d, fl_x, fl_y, cx, cy, c2w))

    world_pts_arr = np.array(world_pts)
    mn = world_pts_arr.min(axis=0)
    mx = world_pts_arr.max(axis=0)
    center = ((mn + mx) / 2).tolist()
    size = (mx - mn).tolist()

    return BBox3D(
        center=(center[0], center[1], center[2]),
        size=(size[0], size[1], size[2]),
    )


def merge_bboxes(bboxes: List[BBox3D]) -> Optional[BBox3D]:
    """Return the axis-aligned union bounding box of a list of BBox3Ds."""
    if not bboxes:
        return None
    mins = np.array([
        [b.center[0] - b.size[0] / 2,
         b.center[1] - b.size[1] / 2,
         b.center[2] - b.size[2] / 2]
        for b in bboxes
    ])
    maxs = np.array([
        [b.center[0] + b.size[0] / 2,
         b.center[1] + b.size[1] / 2,
         b.center[2] + b.size[2] / 2]
        for b in bboxes
    ])
    mn = mins.min(axis=0)
    mx = maxs.max(axis=0)
    center = ((mn + mx) / 2).tolist()
    size = (mx - mn).tolist()
    return BBox3D(
        center=(center[0], center[1], center[2]),
        size=(size[0], size[1], size[2]),
        label=bboxes[0].label,
    )
