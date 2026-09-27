"""MCP tools: geometry helpers for query resolution."""
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import DATA_DIR, DEPTH_NEAR, DEPTH_FAR
from backend.geometry.unprojector import bbox_2d_to_3d, merge_bboxes
from backend.io.nerfstudio import load_transforms
from backend.schemas.types import BBox3D


def _load_depth_map(scene_id: str, view_id: str) -> Optional[np.ndarray]:
    """Load depth map for a view. Returns H×W float32 array in metres."""
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return None
    scene = load_transforms(str(tf_path))
    frame = scene.get_frame(view_id)
    if frame is None or frame.depth_file_path is None:
        return None
    depth_path = scene_dir / frame.depth_file_path
    if not depth_path.exists():
        return None
    return np.load(str(depth_path)).astype(np.float32)


def unproject_bbox_tool(
    scene_id: str,
    view_id: str,
    bbox_norm: List[float],
    label: str = "",
) -> Dict[str, Any]:
    """Unproject a normalised 2D bbox to a 3D BBox3D.

    Args:
        bbox_norm: [x1, y1, x2, y2] all in range [0, 1]
    """
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return {"error": "Scene not found"}

    scene = load_transforms(str(tf_path))
    frame = scene.get_frame(view_id)
    if frame is None:
        return {"error": f"view_id '{view_id}' not found"}

    depth_map = _load_depth_map(scene_id, view_id)
    if depth_map is None:
        return {"error": "Depth map not available for this view"}

    bbox = bbox_2d_to_3d(
        bbox_norm=tuple(bbox_norm),
        depth_map=depth_map,
        fl_x=scene.fl_x,
        fl_y=scene.fl_y,
        cx=scene.cx,
        cy=scene.cy,
        transform_matrix=frame.transform_matrix,
        width=scene.w,
        height=scene.h,
    )
    if bbox is None:
        return {"error": "Could not compute 3D bbox — invalid depth values"}

    bbox.label = label
    return {
        "bbox_3d": bbox.model_dump(),
        "view_id": view_id,
    }


def merge_bboxes_tool(bboxes_dicts: List[dict]) -> Dict[str, Any]:
    """Merge multiple BBox3Ds into a single axis-aligned union bbox."""
    bboxes = [BBox3D.model_validate(b) for b in bboxes_dicts]
    merged = merge_bboxes(bboxes)
    if merged is None:
        return {"error": "No bounding boxes provided"}
    return {"merged_bbox_3d": merged.model_dump()}


def get_capture_intrinsics(scene_id: str) -> Dict[str, Any]:
    """Return camera intrinsics for a scene."""
    tf_path = DATA_DIR / scene_id / "transforms.json"
    if not tf_path.exists():
        return {"error": "Scene not found"}
    scene = load_transforms(str(tf_path))
    return {
        "fl_x": scene.fl_x, "fl_y": scene.fl_y,
        "cx": scene.cx,     "cy": scene.cy,
        "w": scene.w,       "h": scene.h,
    }


def compute_inter_node_distances(
    scene_id: str,
    node_ids: List[str],
) -> Dict[str, Any]:
    """Return pairwise Euclidean distances between node centroids (for query reasoning)."""
    from backend.tree.storage import load_node

    scene_dir = str(DATA_DIR / scene_id)
    centroids: Dict[str, List[float]] = {}
    for nid in node_ids:
        node = load_node(scene_dir, nid)
        if node:
            centroids[nid] = node.centroid

    distances: Dict[str, Dict[str, float]] = {}
    ids = list(centroids.keys())
    for i, a in enumerate(ids):
        distances[a] = {}
        for b in ids[i + 1:]:
            pa = np.array(centroids[a])
            pb = np.array(centroids[b])
            distances[a][b] = float(np.linalg.norm(pa - pb))

    return {"centroids": centroids, "distances": distances}
