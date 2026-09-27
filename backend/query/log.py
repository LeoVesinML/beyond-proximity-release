"""Query session logger — persists every pipeline step to disk as JSON.

Directory layout:
  backend/data/scenes/{scene_id}/queries/
    {session_id}.json          ← live-updated during the query

Each session JSON has the structure:
  {
    "session_id": "...",
    "scene_id": "...",
    "original_query": "...",
    "started_at": "<iso>",
    "finished_at": "<iso> | null",
    "steps": [ { "step_type", "ts", ... } ],
    "result": { ... } | null
  }

Step types: decomposition | traversal | leaf_check | found | not_found | error
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR


def _queries_dir(scene_id: str) -> Path:
    d = DATA_DIR / scene_id / "queries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QuerySession:
    """Accumulates pipeline steps and persists them atomically after each update."""

    def __init__(self, scene_id: str, original_query: str, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.scene_id = scene_id
        self.original_query = original_query
        self.started_at = _now()
        self.finished_at: Optional[str] = None
        self.steps: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None
        self._path = _queries_dir(scene_id) / f"{self.session_id}.json"
        self._flush()

    # ── Public helpers ─────────────────────────────────────────────────────

    def log_decomposition(self, structured_plan: dict, prompt: str) -> None:
        self._push({
            "step_type": "decomposition",
            "structured_plan": structured_plan,
            "prompt_excerpt": prompt[:300],
        })

    def log_traversal(
        self,
        node_id: str,
        node_name: str,
        children_considered: List[dict],
        descend_into: List[str],
        reasoning: str,
    ) -> None:
        self._push({
            "step_type": "traversal",
            "node_id": node_id,
            "node_name": node_name,
            "children_count": len(children_considered),
            "children_names": [c.get("name", c.get("node_id", "?")) for c in children_considered],
            "descend_into": descend_into,
            "reasoning": reasoning,
        })

    def log_leaf_check(
        self,
        node_id: str,
        node_name: str,
        view_id: str,
        found: bool,
        confidence: str,
        bbox_2d: Optional[List[float]],
        matched_object: Optional[str],
        explanation: str,
    ) -> None:
        self._push({
            "step_type": "leaf_check",
            "node_id": node_id,
            "node_name": node_name,
            "view_id": view_id,
            "found": found,
            "confidence": confidence,
            "bbox_2d": bbox_2d,
            "matched_object": matched_object,
            "explanation": explanation,
        })

    def log_result(self, result: dict) -> None:
        self.result = result
        self.finished_at = _now()
        self._push({
            "step_type": "found" if result.get("found") else "not_found",
            "view_id": result.get("view_id"),
            "explanation": result.get("explanation", ""),
            "confidence": result.get("confidence", 0.0),
            "bbox_3d": result.get("bbox_3d"),
        })

    def log_error(self, message: str) -> None:
        self.finished_at = _now()
        self._push({"step_type": "error", "message": message})

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "scene_id": self.scene_id,
            "original_query": self.original_query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": self.steps,
            "result": self.result,
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _push(self, step: Dict[str, Any]) -> None:
        step["ts"] = _now()
        self.steps.append(step)
        self._flush()

    def _flush(self) -> None:
        tmp = self._path.with_name(f".{self._path.stem}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        last_error: PermissionError | None = None
        for attempt in range(8):
            try:
                tmp.replace(self._path)
                return
            except PermissionError as exc:
                # Windows can deny replacing a JSON file while the UI is polling it.
                last_error = exc
                time.sleep(0.025 * (attempt + 1))
        try:
            tmp.unlink(missing_ok=True)
        finally:
            if last_error is not None:
                raise last_error


# ── Module-level helpers ───────────────────────────────────────────────────────

def list_sessions(scene_id: str) -> List[dict]:
    """Return all sessions for a scene, newest first (header only — no steps)."""
    d = _queries_dir(scene_id)
    sessions = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            source = data.get("source", "")
            sessions.append({
                # File stem is the API key (matches benchmark *_graph / *_flat filenames).
                "session_id": p.stem,
                "original_query": data["original_query"],
                "started_at": data["started_at"],
                "finished_at": data.get("finished_at"),
                "step_count": len(data.get("steps", [])),
                "found": data.get("result", {}).get("found") if data.get("result") else None,
                "source": source or None,
            })
        except Exception:
            pass
    return sessions


def get_session(scene_id: str, session_id: str) -> Optional[dict]:
    """Return full session JSON."""
    d = _queries_dir(scene_id)
    p = d / f"{session_id}.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    # Legacy benchmark files: JSON session_id may differ from filename stem.
    for candidate in d.glob("*.json"):
        try:
            with open(candidate, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("session_id") == session_id:
                return data
        except Exception:
            pass
    return None
