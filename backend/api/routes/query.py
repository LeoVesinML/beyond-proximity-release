"""API routes: spec-aligned query pipeline, streaming, bbox unprojection."""
import asyncio
import json
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from backend.mcp.tools.query_tools import unproject_bbox_tool, merge_bboxes_tool
from backend.io.annotator import annotate_view, crop_for_refinement, map_refined_bbox
from backend.mcp.tools.tree_tools import (
    get_root_node_tool,
    get_children_tool,
    get_views_in_node_tool,
)
from backend.query.pipeline import (
    build_decomposition_prompt,
    build_traversal_step,
    build_leaf_confirmation_for_view,
)
from backend.query.benchmark import compare_modes, DEFAULT_BENCHMARK_QUERIES

router = APIRouter()


class UnprojectRequest(BaseModel):
    scene_id: str
    view_id: str
    bbox_norm: list
    label: str = ""


@router.post("/unproject")
def api_unproject(body: UnprojectRequest):
    result = unproject_bbox_tool(
        body.scene_id, body.view_id, body.bbox_norm, body.label
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


class MergeBBoxRequest(BaseModel):
    bboxes: list


@router.post("/merge_bboxes")
def api_merge_bboxes(body: MergeBBoxRequest):
    return merge_bboxes_tool(body.bboxes)


# ── 2-D annotation endpoint ─────────────────────────────────────────────────


class AnnotateViewRequest(BaseModel):
    view_id: str
    boxes: list  # [{bbox_2d, label, confidence, color}]
    max_width: int = 1200


@router.post("/{scene_id}/annotate_view")
def api_annotate_view(scene_id: str, body: AnnotateViewRequest):
    """Draw bbox(es) on the scene image and return a base64 PNG data URL.

    bbox_2d values must be normalised 0-1: [x1, y1, x2, y2].
    """
    data_url = annotate_view(scene_id, body.view_id, body.boxes, max_width=body.max_width)
    if data_url is None:
        raise HTTPException(status_code=404, detail=f"Image not found for view '{body.view_id}'")
    return {"view_id": body.view_id, "annotated_image": data_url}


# ── Spec-aligned pipeline (design doc §7) ───────────────────────────────────


class DecompositionRequest(BaseModel):
    query: str


@router.post("/{scene_id}/decomposition")
def api_query_decomposition(scene_id: str, body: DecompositionRequest):
    """§7.2 — LLM produces structured_plan; use it in /traversal and /leaf_view."""
    _ = scene_id  # reserved for future scene-specific hints
    return build_decomposition_prompt(body.query)


class TraversalRequest(BaseModel):
    original_query: str
    node_id: str
    structured_plan: Optional[Dict[str, Any]] = None


@router.post("/{scene_id}/traversal")
def api_query_traversal(scene_id: str, body: TraversalRequest):
    """§7.3 — one internal-node traversal step (children + prompt)."""
    out = build_traversal_step(
        scene_id, body.node_id, body.original_query, body.structured_plan
    )
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


class LeafViewRequest(BaseModel):
    original_query: str
    node_id: str
    view_id: str
    structured_plan: Optional[Dict[str, Any]] = None


@router.post("/{scene_id}/leaf_view")
def api_query_leaf_view(scene_id: str, body: LeafViewRequest):
    """§7.4 — confirm one leaf view; LLM returns found + bbox_2d for unprojection."""
    out = build_leaf_confirmation_for_view(
        scene_id,
        body.node_id,
        body.view_id,
        body.original_query,
        body.structured_plan,
    )
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


# ── Two-stage bbox refinement (crop-and-requery) ────────────────────────────


class RefineBBoxRequest(BaseModel):
    view_id: str
    rough_bbox: list          # [x1, y1, x2, y2] normalised 0-1
    label: str = ""
    padding: float = 0.5      # how much to expand the rough bbox before cropping


@router.post("/{scene_id}/refine_bbox_prompt")
def api_refine_bbox_prompt(scene_id: str, body: RefineBBoxRequest):
    """Stage 1 of two-stage bbox refinement.

    Returns a padded crop of the image centred on rough_bbox, plus a VLM prompt.
    Feed the prompt + crop_b64 to the LLM, get back bbox_2d_in_crop, then call
    /map_refined_bbox to get the final original-image coordinates.
    """
    result = crop_for_refinement(
        scene_id, body.view_id, body.rough_bbox, body.label, body.padding
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Image not found for view '{body.view_id}'")
    return result


class MapRefinedBBoxRequest(BaseModel):
    crop_region: list         # [cx1, cy1, cx2, cy2] normalised in original image
    bbox_in_crop: list        # [rx1, ry1, rx2, ry2] normalised within the crop


@router.post("/{scene_id}/map_refined_bbox")
def api_map_refined_bbox(scene_id: str, body: MapRefinedBBoxRequest):
    """Stage 2 of two-stage bbox refinement.

    Maps the LLM's crop-relative bbox back to original image coordinates.
    """
    mapped = map_refined_bbox(body.crop_region, body.bbox_in_crop)
    return {"bbox_2d_original": mapped}


# ── Graph vs flat benchmark (resources & speed) ────────────────────────────────


class BenchmarkRequest(BaseModel):
    query: Optional[str] = None
    queries: Optional[list[str]] = None


@router.post("/{scene_id}/benchmark")
def api_query_benchmark(scene_id: str, body: BenchmarkRequest):
    """Compare graph-pruned vs flat (all-views) search — counts prompt tokens."""
    queries = body.queries or ([body.query] if body.query else DEFAULT_BENCHMARK_QUERIES)
    results = [compare_modes(scene_id, q) for q in queries]
    n = len(results)
    avg_token_savings = (
        sum(r["savings"]["tokens_pct"] for r in results) / n if n else 0
    )
    return {
        "scene_id": scene_id,
        "queries_run": n,
        "avg_token_savings_pct": round(avg_token_savings, 1),
        "results": results,
    }


# ── Optional keyword demo (not part of the spec algorithm) ──────────────────

_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "why", "me", "my", "we", "can",
    "could", "please", "tell", "show", "get", "locate", "look", "there",
})


def _view_matches_query(view_dict: dict, query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    haystack = json.dumps(view_dict, ensure_ascii=False).lower()
    if q in haystack:
        return True
    tokens = [t for t in re.split(r"[^\w]+", q) if len(t) >= 2]
    keywords = [t for t in tokens if t not in _QUERY_STOPWORDS]
    if not keywords:
        keywords = tokens
    if not keywords:
        return False
    return all(k in haystack for k in keywords)


class RunKeywordDemoRequest(BaseModel):
    query: str = Field(..., description="Substring / token demo only — not spec §7.")


@router.post("/{scene_id}/run_keyword_demo")
async def api_run_keyword_demo(scene_id: str, body: RunKeywordDemoRequest):
    """Dev-only: scan leaf JSONs for keywords. Spec pipeline uses decomposition + LLM."""
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    async def emit(event: dict):
        await manager.broadcast(event)

    await emit({"event": "start", "message": f'[keyword demo] "{query}"'})

    root = get_root_node_tool(scene_id)
    if "error" in root:
        await emit({"event": "not_found", "message": root["error"]})
        return {
            "query": query,
            "found": False,
            "view_id": None,
            "bbox_3d": None,
            "camera_pose": None,
            "confidence": 0.0,
            "explanation": root["error"],
            "mode": "keyword_demo",
        }

    await emit({
        "event": "visit_node",
        "node_id": root["node_id"],
        "node_name": root["name"],
        "message": root["summary"],
    })
    await asyncio.sleep(0.02)

    stack = [root["node_id"]]
    while stack:
        node_id = stack.pop()
        step = get_children_tool(scene_id, node_id, query, "{}")
        if "error" in step:
            continue

        if step.get("is_leaf"):
            await emit({
                "event": "leaf_check",
                "node_id": node_id,
                "node_name": step.get("node_name"),
                "message": "Checking leaf views",
            })
            leaf_data = get_views_in_node_tool(scene_id, node_id, query)
            for view in leaf_data.get("views", []):
                vid = view.get("view_id")
                await emit({
                    "event": "leaf_check",
                    "node_id": node_id,
                    "node_name": step.get("node_name"),
                    "view_id": vid,
                    "message": "Scanning view",
                })
                if _view_matches_query(view, query):
                    explanation = (
                        f"[keyword demo] Matched '{query}' in view {vid} under node "
                        f"{step.get('node_name', node_id)}."
                    )
                    await emit({
                        "event": "found",
                        "node_id": node_id,
                        "node_name": step.get("node_name"),
                        "view_id": vid,
                        "message": explanation,
                    })
                    return {
                        "query": query,
                        "found": True,
                        "view_id": vid,
                        "bbox_3d": None,
                        "camera_pose": None,
                        "confidence": 0.75,
                        "explanation": explanation,
                        "mode": "keyword_demo",
                    }
            continue

        for child in step.get("children", []):
            await emit({
                "event": "descend",
                "node_id": child.get("node_id"),
                "node_name": child.get("name"),
                "message": child.get("summary", ""),
            })
            await emit({
                "event": "visit_node",
                "node_id": child.get("node_id"),
                "node_name": child.get("name"),
                "message": child.get("summary", ""),
            })
            stack.append(child.get("node_id"))
            await asyncio.sleep(0.02)

    explanation = f"[keyword demo] No match for '{query}' in leaf view analyses."
    await emit({"event": "not_found", "message": explanation})
    return {
        "query": query,
        "found": False,
        "view_id": None,
        "bbox_3d": None,
        "camera_pose": None,
        "confidence": 0.0,
        "explanation": explanation,
        "mode": "keyword_demo",
    }


# ── WebSocket for query traversal streaming ──────────────────────────────────


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@router.websocket("/{scene_id}/stream")
async def ws_query_stream(websocket: WebSocket, scene_id: str):
    """WebSocket that Cursor AI pushes traversal events to for frontend viz."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                event = json.loads(data)
                await websocket.send_json({"echo": event})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
