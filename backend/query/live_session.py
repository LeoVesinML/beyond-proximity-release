"""Run §7 query pipeline and log every step to an existing Query Flow session."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR
from backend.io.captured_semantic_index import (
    QUERY_RUNNER_SCHEMA_COMPATIBLE,
    TREE_MANIFEST_SCHEMA_VERSION,
    load_captured_scene_index,
    load_json,
)
from backend.mcp.tools.query_tools import unproject_bbox_tool
from backend.query.benchmark import (
    _leaf_match_candidate,
    _pick_best_leaf_match,
    _view_matches,
    simulate_graph_search,
)
from backend.query.benchmark_reasoning import _extract_room, _extract_target
from backend.query.log import QuerySession, get_session, _queries_dir
from backend.query.pipeline import build_decomposition_prompt
from backend.query.model_client import StubModelClient
from backend.io.view_store import load_view_analysis
from backend.tree.storage import load_all_nodes


def _is_captured_semantic_index(scene_dir: Path) -> bool:
    manifest_path = scene_dir / "tree" / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        return load_json(manifest_path).get("schema_version") == TREE_MANIFEST_SCHEMA_VERSION
    except (OSError, ValueError):
        return False


def _run_captured_query_session(
    sess: QuerySession,
    query: str,
    scene_dir: Path,
    *,
    step_delay_sec: float,
) -> QuerySession:
    if not QUERY_RUNNER_SCHEMA_COMPATIBLE:
        raise ValueError("Captured ViewJSON query adapter is unavailable")
    index = load_captured_scene_index(
        scene_dir,
        require_complete=True,
        allow_unindexed_frames=True,
    )
    client = StubModelClient()
    answer = client.answer_query(
        {"query": query, "query_type": "object_finding"},
        index["tree"],
        index["views"],
    )
    plan = answer.get("structured_plan", {})
    decomposition = build_decomposition_prompt(query)
    sess.log_decomposition(plan, decomposition["prompt"])
    if step_delay_sec:
        time.sleep(step_delay_sec)

    visited = [str(item) for item in answer.get("visited_nodes", [])]
    for position, node_id in enumerate(visited[:-1]):
        node = index["tree"].get(node_id)
        if not node:
            continue
        children = [
            index["tree"][child_id]
            for child_id in node.get("children_ids", [])
            if child_id in index["tree"]
        ]
        next_id = visited[position + 1]
        sess.log_traversal(
            node_id,
            str(node.get("name", node_id)),
            [{"node_id": child["node_id"], "name": child.get("name", child["node_id"])} for child in children],
            [next_id],
            "[stub] Deterministic lexical traversal over the saved captured semantic index.",
        )
        if step_delay_sec:
            time.sleep(step_delay_sec)

    result = answer.get("result", {}) if isinstance(answer.get("result"), dict) else {}
    selected_view = result.get("selected_view_id")
    if selected_view:
        selected_node = str(result.get("selected_node_id") or "root")
        node = index["tree"].get(selected_node, {})
        sess.log_leaf_check(
            selected_node,
            str(node.get("name", selected_node)),
            str(selected_view),
            found=bool(result.get("found")),
            confidence="stub",
            bbox_2d=result.get("bbox_2d"),
            matched_object=result.get("matched_object"),
            explanation="[stub] " + str(result.get("explanation", "")),
        )

    final = {
        "found": bool(result.get("found")),
        "view_id": selected_view,
        "explanation": "[stub] " + str(result.get("explanation", "")),
        "confidence": float(result.get("confidence", 0.0)),
        "bbox_3d": None,
        "query": query,
        "mode": "stub",
        "semantic_index_mode": index["manifest"].get("semantic_index_mode"),
        "warnings": [*answer.get("warnings", []), *index.get("warnings", [])],
    }
    sess.log_result(final)
    return sess


def load_session_for_update(scene_id: str, session_id: str) -> QuerySession:
    data = get_session(scene_id, session_id)
    if data is None:
        raise ValueError(f"Session '{session_id}' not found for scene '{scene_id}'")
    sess = QuerySession.__new__(QuerySession)
    sess.session_id = data["session_id"]
    sess.scene_id = data["scene_id"]
    sess.original_query = data["original_query"]
    sess.started_at = data["started_at"]
    sess.finished_at = data.get("finished_at")
    sess.steps = []
    sess.result = None
    sess._path = _queries_dir(scene_id) / f"{session_id}.json"
    sess._flush()
    return sess


def run_query_session(
    scene_id: str,
    session_id: str,
    query: Optional[str] = None,
    *,
    step_delay_sec: float = 0.4,
) -> QuerySession:
    """Execute graph search and persist steps to disk for Query Flow polling."""
    sess = load_session_for_update(scene_id, session_id)
    q = (query or sess.original_query).strip()
    scene_path = DATA_DIR / scene_id
    if _is_captured_semantic_index(scene_path):
        return _run_captured_query_session(
            sess,
            q,
            scene_path,
            step_delay_sec=step_delay_sec,
        )

    scene_dir = str(scene_path)
    nodes = load_all_nodes(scene_dir)

    plan = {
        "target": _extract_target(q),
        "location_constraints": _extract_room(q),
        "query_type": "object_finding",
        "output_type": "3d_bbox",
    }
    dec = build_decomposition_prompt(q)
    sess.log_decomposition(plan, dec["prompt"])
    if step_delay_sec:
        time.sleep(step_delay_sec)

    result = simulate_graph_search(scene_id, q, plan)
    path = result.path

    for i, node_id in enumerate(path):
        node = nodes.get(node_id)
        if not node or not node.children_ids:
            continue
        next_id = path[i + 1] if i + 1 < len(path) else None
        if next_id is None:
            continue
        siblings = [nodes[cid] for cid in node.children_ids if cid in nodes]
        next_node = nodes.get(next_id)
        sibling_names = [s.name for s in siblings]
        reasoning = (
            f"At '{node.name}' ({node.node_type.value}). "
            f"Zone summary: {node.summary} "
            f"Descending into '{next_node.name}' — "
            f"summary '{next_node.summary}' matches query constraints. "
            f"Siblings not taken: {', '.join(n for n in sibling_names if n != next_node.name) or 'none'}."
        )
        children_info = [{"node_id": s.node_id, "name": s.name} for s in siblings]
        sess.log_traversal(node_id, node.name, children_info, [next_id], reasoning)
        if step_delay_sec:
            time.sleep(step_delay_sec)

    winner_view = result.view_id
    leaf_candidates = []

    for vid in result.views_checked_ids:
        node_id, node_name = "unknown", "unknown"
        for nid, n in nodes.items():
            if vid in n.view_ids:
                node_id, node_name = nid, n.name
                break
        v = load_view_analysis(scene_dir, vid)
        view_dict = v.model_dump() if v else {}
        matched = bool(v and _view_matches(view_dict, q))
        if matched:
            cand = _leaf_match_candidate(scene_id, node_id, vid, q, view_dict)
            leaf_candidates.append(cand)
            is_winner = vid == winner_view
        else:
            is_winner = False
            cand = None

        sess.log_leaf_check(
            node_id,
            node_name,
            vid,
            found=is_winner and matched,
            confidence=(cand or {}).get("confidence", "high") if matched else "high",
            bbox_2d=(cand or {}).get("bbox_2d") if is_winner and matched else None,
            matched_object=(cand or {}).get("matched_object") if is_winner and matched else None,
            explanation=view_dict.get("scene_summary", ""),
        )
        if step_delay_sec:
            time.sleep(step_delay_sec)

    best = _pick_best_leaf_match(scene_id, leaf_candidates) if leaf_candidates else None
    bbox_3d = None
    if best and best.get("bbox_2d"):
        up = unproject_bbox_tool(scene_id, best["view_id"], best["bbox_2d"], best.get("matched_object") or plan["target"])
        if "bbox_3d" in up:
            bbox_3d = up["bbox_3d"]

    final = {
        "found": result.found,
        "view_id": result.view_id,
        "explanation": (best or {}).get("explanation", "No match in traversed views."),
        "confidence": 0.9 if result.found else 0.0,
        "bbox_3d": bbox_3d,
        "query": q,
    }
    sess.log_result(final)
    return sess


def run_query_session_safe(
    scene_id: str,
    session_id: str,
    query: Optional[str] = None,
    *,
    step_delay_sec: float = 0.35,
) -> None:
    """Run pipeline; persist an error step if execution fails."""
    try:
        run_query_session(
            scene_id,
            session_id,
            query,
            step_delay_sec=step_delay_sec,
        )
    except Exception as exc:
        data = get_session(scene_id, session_id)
        if data is None:
            raise
        sess = QuerySession.__new__(QuerySession)
        sess.session_id = data["session_id"]
        sess.scene_id = data["scene_id"]
        sess.original_query = data["original_query"]
        sess.started_at = data["started_at"]
        sess.finished_at = data.get("finished_at")
        sess.steps = data.get("steps", [])
        sess.result = data.get("result")
        sess._path = _queries_dir(scene_id) / f"{session_id}.json"
        sess.log_error(str(exc))
