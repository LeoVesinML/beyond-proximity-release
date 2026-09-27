"""API routes: query session log — list, get, and create sessions."""
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.query.live_session import run_query_session_safe
from backend.query.log import list_sessions, get_session, QuerySession

router = APIRouter()


@router.get("/{scene_id}/sessions")
def api_list_sessions(scene_id: str):
    """List all query sessions for a scene (newest first, header only)."""
    return {"sessions": list_sessions(scene_id)}


@router.get("/{scene_id}/sessions/{session_id}")
def api_get_session(scene_id: str, session_id: str):
    """Return full session JSON including all pipeline steps."""
    data = get_session(scene_id, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return data


class CreateSessionRequest(BaseModel):
    original_query: str


@router.post("/{scene_id}/sessions")
def api_create_session(scene_id: str, body: CreateSessionRequest):
    """Create a new (empty) log session. Returns session_id for subsequent log calls."""
    sess = QuerySession(scene_id, body.original_query)
    return {"session_id": sess.session_id}


class RunSessionRequest(BaseModel):
    original_query: str
    step_delay_sec: float = 0.35


@router.post("/{scene_id}/sessions/run")
def api_run_session(scene_id: str, body: RunSessionRequest, background_tasks: BackgroundTasks):
    """Create a session and run the §7 pipeline in the background (Query Flow polls live steps)."""
    sess = QuerySession(scene_id, body.original_query)
    background_tasks.add_task(
        run_query_session_safe,
        scene_id,
        sess.session_id,
        body.original_query,
        step_delay_sec=body.step_delay_sec,
    )
    return {"session_id": sess.session_id, "status": "running"}


class LogDecompositionRequest(BaseModel):
    session_id: str
    structured_plan: Dict[str, Any]
    prompt: str = ""


@router.post("/{scene_id}/sessions/log/decomposition")
def api_log_decomposition(scene_id: str, body: LogDecompositionRequest):
    sess = _load_session(scene_id, body.session_id)
    sess.log_decomposition(body.structured_plan, body.prompt)
    return {"ok": True}


class LogTraversalRequest(BaseModel):
    session_id: str
    node_id: str
    node_name: str
    children_considered: list
    descend_into: list
    reasoning: str = ""


@router.post("/{scene_id}/sessions/log/traversal")
def api_log_traversal(scene_id: str, body: LogTraversalRequest):
    sess = _load_session(scene_id, body.session_id)
    sess.log_traversal(
        body.node_id,
        body.node_name,
        body.children_considered,
        body.descend_into,
        body.reasoning,
    )
    return {"ok": True}


class LogLeafCheckRequest(BaseModel):
    session_id: str
    node_id: str
    node_name: str
    view_id: str
    found: bool
    confidence: str = "low"
    bbox_2d: Optional[list] = None
    matched_object: Optional[str] = None
    explanation: str = ""


@router.post("/{scene_id}/sessions/log/leaf_check")
def api_log_leaf_check(scene_id: str, body: LogLeafCheckRequest):
    sess = _load_session(scene_id, body.session_id)
    sess.log_leaf_check(
        body.node_id,
        body.node_name,
        body.view_id,
        body.found,
        body.confidence,
        body.bbox_2d,
        body.matched_object,
        body.explanation,
    )
    return {"ok": True}


class LogResultRequest(BaseModel):
    session_id: str
    result: Dict[str, Any]


@router.post("/{scene_id}/sessions/log/result")
def api_log_result(scene_id: str, body: LogResultRequest):
    sess = _load_session(scene_id, body.session_id)
    sess.log_result(body.result)
    return {"ok": True}


# ── Helper: reload session from disk so it reflects all prior Cursor steps ────

def _load_session(scene_id: str, session_id: str) -> QuerySession:
    from backend.query.log import get_session as _gs
    data = _gs(scene_id, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    sess = QuerySession.__new__(QuerySession)
    sess.session_id = data["session_id"]
    sess.scene_id = data["scene_id"]
    sess.original_query = data["original_query"]
    sess.started_at = data["started_at"]
    sess.finished_at = data.get("finished_at")
    sess.steps = data.get("steps", [])
    sess.result = data.get("result")
    from backend.query.log import _queries_dir
    sess._path = _queries_dir(scene_id) / f"{session_id}.json"
    return sess
