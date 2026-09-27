"""MCP tools: scene management — list, init, get info."""
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.config import DATA_DIR
from backend.io.nerfstudio import get_or_create_transforms, load_transforms


def list_scenes() -> List[Dict[str, Any]]:
    """List all scenes in the data directory."""
    if not DATA_DIR.exists():
        return []
    result = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir():
            tf = p / "transforms.json"
            frame_count = 0
            if tf.exists():
                try:
                    scene = load_transforms(str(tf))
                    frame_count = len(scene.frames)
                except Exception:
                    pass
            result.append({
                "scene_id": p.name,
                "path": str(p),
                "frame_count": frame_count,
                "has_tree": (p / "tree" / "manifest.json").exists(),
            })
    return result


def get_scene_info(scene_id: str) -> Dict[str, Any]:
    """Return detailed info about a scene."""
    scene_dir = DATA_DIR / scene_id
    if not scene_dir.exists():
        return {"error": f"Scene '{scene_id}' not found"}

    tf_path = scene_dir / "transforms.json"
    frames = []
    intrinsics: dict = {}
    if tf_path.exists():
        scene = load_transforms(str(tf_path))
        intrinsics = {
            "fl_x": scene.fl_x, "fl_y": scene.fl_y,
            "cx": scene.cx,     "cy": scene.cy,
            "w": scene.w,       "h": scene.h,
        }
        frames = [
            {
                "view_id": f.view_id,
                "file_path": f.file_path,
                "position": f.camera_position(),
                "has_depth": f.depth_file_path is not None,
            }
            for f in scene.frames
        ]

    view_ids_with_analysis = []
    views_dir = scene_dir / "views"
    if views_dir.exists():
        view_ids_with_analysis = [p.stem for p in views_dir.glob("*.json")]

    tree_manifest: dict = {}
    manifest_path = scene_dir / "tree" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            tree_manifest = json.load(fh)

    return {
        "scene_id": scene_id,
        "path": str(scene_dir),
        "intrinsics": intrinsics,
        "frames": frames,
        "analyzed_views": sorted(view_ids_with_analysis),
        "tree": tree_manifest,
    }


def init_scene(
    scene_id: str,
    fl_x: float = 800.0,
    fl_y: float = 800.0,
    cx: float = 400.0,
    cy: float = 300.0,
    width: int = 800,
    height: int = 600,
) -> Dict[str, Any]:
    """Initialise a new scene directory with empty transforms.json."""
    scene_dir = DATA_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(exist_ok=True)
    (scene_dir / "depths").mkdir(exist_ok=True)
    (scene_dir / "views").mkdir(exist_ok=True)
    (scene_dir / "tree").mkdir(exist_ok=True)

    tf_path = str(scene_dir / "transforms.json")
    get_or_create_transforms(tf_path, fl_x, fl_y, cx, cy, width, height)

    return {
        "scene_id": scene_id,
        "path": str(scene_dir),
        "status": "initialised",
    }
