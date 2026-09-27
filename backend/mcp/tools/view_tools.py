"""MCP tools: view management — images, analysis, summaries."""
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR
from backend.io.nerfstudio import load_transforms
from backend.io.view_store import (
    list_view_ids,
    list_view_summaries,
    load_view_analysis,
    save_view_analysis,
    view_analysis_exists,
)
from backend.schemas.types import VIEW_ANALYSIS_PROMPT, ViewJSON


def get_unanalyzed_views(scene_id: str) -> List[str]:
    """Return view_ids that have captured images but no VLM analysis yet."""
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return []
    scene = load_transforms(str(tf_path))
    return [
        f.view_id
        for f in scene.frames
        if f.view_id and not view_analysis_exists(str(scene_dir), f.view_id)
    ]


def get_view_image(scene_id: str, view_id: str) -> Dict[str, Any]:
    """Return base64-encoded PNG for a view and the prompt Cursor AI should use.

    Cursor AI should:
    1. Decode the image from 'image_b64'.
    2. Analyse it using 'prompt' with its own vision capability.
    3. Call save_view_analysis() with the resulting JSON.
    """
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return {"error": "Scene not found"}

    scene = load_transforms(str(tf_path))
    frame = scene.get_frame(view_id)
    if frame is None:
        return {"error": f"view_id '{view_id}' not found"}

    img_path = scene_dir / frame.file_path
    if not img_path.exists():
        return {"error": f"Image file not found: {frame.file_path}"}

    with open(img_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    return {
        "view_id": view_id,
        "image_b64": img_b64,
        "mime_type": "image/png",
        "prompt": VIEW_ANALYSIS_PROMPT,
        "instruction": (
            "Analyse the image above using the prompt, then call save_view_analysis "
            f"with scene_id='{scene_id}', view_id='{view_id}', and the JSON result."
        ),
    }


def save_view_analysis_tool(
    scene_id: str,
    view_id: str,
    view_json_dict: dict,
) -> Dict[str, Any]:
    """Persist Cursor AI's VLM analysis for a view."""
    scene_dir = str(DATA_DIR / scene_id)
    try:
        view_json = ViewJSON.model_validate({**view_json_dict, "view_id": view_id})
        save_view_analysis(scene_dir, view_id, view_json)
        return {"status": "saved", "view_id": view_id}
    except Exception as e:
        return {"error": str(e)}


def get_view_full_json(scene_id: str, view_id: str) -> Dict[str, Any]:
    """Return the full ViewJSON dict for a given view."""
    scene_dir = str(DATA_DIR / scene_id)
    v = load_view_analysis(scene_dir, view_id)
    if v is None:
        return {"error": f"No analysis for view '{view_id}'"}
    return v.model_dump()


def list_view_summaries_tool(scene_id: str) -> List[dict]:
    """Return [{view_id, summary, object_count, room_type}] for all analysed views."""
    return list_view_summaries(str(DATA_DIR / scene_id))


def get_view_camera_pose(scene_id: str, view_id: str) -> Dict[str, Any]:
    """Return camera intrinsics and extrinsics for a view."""
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return {"error": "Scene not found"}
    scene = load_transforms(str(tf_path))
    frame = scene.get_frame(view_id)
    if frame is None:
        return {"error": f"view_id '{view_id}' not found"}
    return {
        "view_id": view_id,
        "transform_matrix": frame.transform_matrix,
        "position": frame.camera_position(),
        "rotation_3x3": frame.rotation_matrix_3x3(),
        "fl_x": scene.fl_x,
        "fl_y": scene.fl_y,
        "cx": scene.cx,
        "cy": scene.cy,
        "w": scene.w,
        "h": scene.h,
    }
