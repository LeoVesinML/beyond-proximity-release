"""API routes: scene management."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.mcp.tools.scene_tools import list_scenes, get_scene_info, init_scene

router = APIRouter()


class InitSceneRequest(BaseModel):
    fl_x: float = 800.0
    fl_y: float = 800.0
    cx: float = 400.0
    cy: float = 300.0
    width: int = 800
    height: int = 600


@router.get("")
def api_list_scenes():
    return list_scenes()


@router.get("/{scene_id}")
def api_get_scene(scene_id: str):
    info = get_scene_info(scene_id)
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    return info


@router.post("/{scene_id}/init")
def api_init_scene(scene_id: str, body: InitSceneRequest):
    return init_scene(
        scene_id,
        fl_x=body.fl_x,
        fl_y=body.fl_y,
        cx=body.cx,
        cy=body.cy,
        width=body.width,
        height=body.height,
    )
