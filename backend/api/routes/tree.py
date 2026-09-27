"""API routes: tree visualisation."""
from fastapi import APIRouter, HTTPException

from backend.mcp.tools.tree_tools import get_tree_for_viz, get_node_tool

router = APIRouter()


@router.get("/{scene_id}/viz")
def api_tree_viz(scene_id: str):
    result = get_tree_for_viz(scene_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{scene_id}/node/{node_id}")
def api_get_node(scene_id: str, node_id: str):
    result = get_node_tool(scene_id, node_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
