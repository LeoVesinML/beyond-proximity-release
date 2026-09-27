"""Spec-aligned query pipeline helpers (design doc §7).

LLM calls happen outside this module (Cursor AI or any client). This layer only
assembles prompts, children metadata, and view JSON for each step.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR
from backend.io.annotator import view_image_usable
from backend.io.view_store import load_view_analysis
from backend.schemas.types import (
    QUERY_DECOMPOSITION_PROMPT,
    TRAVERSAL_TOPDOWN_PROMPT,
    LEAF_CONFIRMATION_SINGLE_VIEW_PROMPT,
)
from backend.tree.storage import load_node


def build_decomposition_prompt(query: str) -> Dict[str, Any]:
    """Step 7.2 — query decomposition."""
    q = query.strip()
    prompt = QUERY_DECOMPOSITION_PROMPT.replace("__QUERY__", q)
    return {
        "phase": "decomposition",
        "original_query": q,
        "prompt": prompt,
        "output_schema_hint": {
            "target": "string",
            "location_constraints": "string | null",
            "attribute_constraints": "object | null",
            "relational_constraints": "list[string] | null",
            "functional_constraints": "string | null",
            "query_type": (
                "object_finding | descriptive | aggregation | "
                "cross_zone_geometric | spatial_relation"
            ),
            "output_type": "3d_bbox | camera_pose | text_answer | count",
        },
        "instruction": (
            "Call an LLM with the prompt above. Use the JSON object it returns as "
            "`structured_plan` for subsequent /traversal and /leaf_view calls."
        ),
    }


def _children_block(children_info: List[dict]) -> str:
    lines = []
    for c in children_info:
        nid = c.get("node_id", "")
        summ = c.get("summary", "")
        lines.append(f'  [{nid}]: {summ}')
    return "\n".join(lines) if lines else "  (no children)"


def build_traversal_step(
    scene_id: str,
    node_id: str,
    original_query: str,
    structured_plan: Optional[dict] = None,
) -> Dict[str, Any]:
    """Step 7.3 — top-down traversal at one internal node."""
    scene_dir = str(DATA_DIR / scene_id)
    node = load_node(scene_dir, node_id)
    if node is None:
        return {"error": f"Node '{node_id}' not found"}

    children_info: List[dict] = []
    for cid in node.children_ids:
        child = load_node(scene_dir, cid)
        if child:
            children_info.append({
                "node_id": cid,
                "name": child.name,
                "summary": child.summary,
                "type": child.node_type.value,
                "view_count": len(child.view_ids),
                "centroid": child.centroid,
            })

    plan = structured_plan if structured_plan is not None else {}
    plan_json = json.dumps(plan, indent=2, ensure_ascii=False)
    children_lines = _children_block(children_info)

    prompt = (
        TRAVERSAL_TOPDOWN_PROMPT.replace("__ORIGINAL_QUERY__", original_query.strip())
        .replace("__STRUCTURED_PLAN_JSON__", plan_json)
        .replace("__CURRENT_NODE_NAME__", node.name)
        .replace("__CURRENT_NODE_SUMMARY__", node.summary)
        .replace("__CHILDREN_BLOCK__", children_lines)
    )

    return {
        "phase": "traversal",
        "node_id": node_id,
        "node_name": node.name,
        "node_summary": node.summary,
        "children": children_info,
        "is_leaf": len(node.children_ids) == 0,
        "view_ids": node.view_ids,
        "prompt": prompt,
        "output_schema_hint": {
            "descend_into": ["<node_id>", "..."],
            "reasoning": "<one sentence>",
        },
        "instruction": (
            "Call an LLM with this prompt. Use descend_into to choose the next "
            "node_id(s). For leaves, call leaf_view for each candidate view_id."
        ),
    }


def build_leaf_confirmation_for_view(
    scene_id: str,
    node_id: str,
    view_id: str,
    original_query: str,
    structured_plan: Optional[dict] = None,
) -> Dict[str, Any]:
    """Step 7.4 — leaf confirmation for a single view."""
    scene_dir = str(DATA_DIR / scene_id)
    node = load_node(scene_dir, node_id)
    if node is None:
        return {"error": f"Node '{node_id}' not found"}
    if view_id not in node.view_ids:
        return {"error": f"view_id '{view_id}' is not in node '{node_id}'"}

    v = load_view_analysis(scene_dir, view_id)
    if v is None:
        return {"error": f"No analysis JSON for view '{view_id}'"}

    view_json = json.dumps(v.model_dump(), indent=2, ensure_ascii=False)
    plan = structured_plan if structured_plan is not None else {}
    plan_json = json.dumps(plan, indent=2, ensure_ascii=False)

    prompt = (
        LEAF_CONFIRMATION_SINGLE_VIEW_PROMPT.replace("__QUERY__", original_query.strip())
        .replace("__STRUCTURED_PLAN_JSON__", plan_json)
        .replace("__VIEW_ID__", view_id)
        .replace("__FULL_VIEW_JSON__", view_json)
    )

    return {
        "phase": "leaf_confirmation",
        "node_id": node_id,
        "view_id": view_id,
        "view_json": v.model_dump(),
        "prompt": prompt,
        "instruction": (
            "Call an LLM with this prompt. If found=true and bbox_2d is set, "
            "call POST /api/query/unproject to obtain bbox_3d."
        ),
    }


# ── Best-view ranking ─────────────────────────────────────────────────────────

def score_view_result(result: dict, scene_id: Optional[str] = None) -> float:
    """Score a single leaf-confirmation result for best-view ranking.

    Returns a float in [0, 1] — higher is better.  Used to select the canonical
    view when multiple views contain the same object.

    Factors (weighted sum):
      - Confidence level   (weight 0.40):  high=1.0 / medium=0.6 / low=0.3
      - BBox area          (weight 0.25):  larger bbox = object fills more of the
                                           frame, indicating closer/cleaner shot.
                                           Capped at 25 % of image area.
      - BBox centrality    (weight 0.15):  bbox centre close to image centre
      - Usable PNG         (weight 0.20):  skip dark placeholder images in Query Flow
    """
    confidence_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    confidence_score = confidence_map.get(result.get("confidence", "low"), 0.3)

    bbox = result.get("bbox_2d")
    bbox_score = 0.0
    centrality_score = 0.0

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        area = max(0.0, (x2 - x1) * (y2 - y1))

        # Scale so that a bbox covering ~25% of the image scores 1.0;
        # penalise very small bboxes (< 1 % of image) lightly.
        if area >= 0.01:
            bbox_score = min(area / 0.25, 1.0)
        else:
            bbox_score = area / 0.01 * 0.1          # near-zero for tiny boxes

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        # Max possible distance from centre to corner ≈ 0.707
        centrality_score = max(0.0, 1.0 - dist / 0.707)

    image_score = 1.0
    if scene_id and result.get("view_id"):
        image_score = 1.0 if view_image_usable(scene_id, result["view_id"]) else 0.0

    return round(
        0.40 * confidence_score
        + 0.25 * bbox_score
        + 0.15 * centrality_score
        + 0.20 * image_score,
        4,
    )


def rank_leaf_results(results: List[dict], scene_id: Optional[str] = None) -> List[dict]:
    """Sort a list of leaf-confirmation results, best first.

    Each item must have at minimum: view_id, found, confidence, bbox_2d.
    Items with found=False are moved to the end.

    Returns the same dicts with an added '_score' key for transparency.
    """
    def sort_key(r: dict) -> float:
        if not r.get("found", False):
            return -1.0
        return score_view_result(r, scene_id)

    ranked = sorted(results, key=sort_key, reverse=True)
    for r in ranked:
        r["_score"] = score_view_result(r, scene_id) if r.get("found") else 0.0
    return ranked
