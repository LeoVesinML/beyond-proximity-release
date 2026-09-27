"""MCP tools: semantic tree — build, save, traverse."""
import json
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR
from backend.geometry.spatial_graph import build_spatial_clusters, cluster_centroid
from backend.io.annotator import crop_for_refinement, map_refined_bbox
from backend.io.nerfstudio import load_transforms
from backend.io.view_store import list_view_summaries, load_view_analysis
from backend.query.pipeline import (
    build_decomposition_prompt,
    build_traversal_step,
    build_leaf_confirmation_for_view,
    rank_leaf_results,
)
from backend.schemas.types import TREE_GROUPING_PROMPT, LEAF_CONFIRMATION_PROMPT
from backend.tree.nodes import NodeType, TreeNode
from backend.tree.storage import (
    get_root_node_id,
    load_node,
    save_node,
    to_d3_tree,
    load_all_nodes,
)


def build_spatial_clusters_tool(
    scene_id: str,
    geometry_assisted: bool = True,
) -> Dict[str, Any]:
    """Return spatial clusters of views + the prompt for Cursor AI to assign zone names.

    When view analyses with `facing` / `visible_landmarks` fields are available a
    semantic-merge pass is applied automatically after the geometric pass.  This
    correctly groups views from opposite ends of the same large room (e.g. the
    stage end and audience end of a ballroom) into a single cluster.

    If geometry_assisted=False all views are returned as a single cluster (ablation).
    """
    scene_dir = DATA_DIR / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.exists():
        return {"error": "Scene not found"}

    scene = load_transforms(str(tf_path))
    positions = {
        f.view_id: f.camera_position()
        for f in scene.frames
        if f.view_id
    }

    raw_summaries = list_view_summaries(str(scene_dir))

    if geometry_assisted:
        # Two-pass: geometric proximity then semantic merge
        clusters = build_spatial_clusters(
            positions,
            view_summaries=raw_summaries if raw_summaries else None,
        )
    else:
        # Ablation: one giant cluster — LLM decides everything
        clusters = [list(positions.keys())]

    # Build a fast lookup for per-view metadata
    summary_map: Dict[str, dict] = {s["view_id"]: s for s in raw_summaries}

    cluster_info = []
    for idx, cluster_views in enumerate(clusters):
        centroid = cluster_centroid(cluster_views, positions)
        view_details = []
        for vid in cluster_views:
            smeta = summary_map.get(vid, {})
            view_details.append({
                "view_id": vid,
                "position": [round(x, 2) for x in positions.get(vid, [0.0, 0.0, 0.0])],
                "summary": smeta.get("summary", ""),
                "room_type": smeta.get("room_type", "unknown"),
                "facing": smeta.get("facing", "unknown"),
                "visible_landmarks": smeta.get("visible_landmarks", []),
            })
        cluster_info.append({
            "cluster_index": idx,
            "centroid": [round(x, 2) for x in centroid],
            "view_count": len(cluster_views),
            "views": view_details,
        })

    # Use .replace instead of str.format so JSON braces in summaries never conflict.
    _clusters_json = json.dumps(cluster_info, indent=2)
    _prompt = (
        TREE_GROUPING_PROMPT
        .replace("{n}", str(len(positions)))
        .replace("{clusters_json}", _clusters_json)
    )

    return {
        "scene_id": scene_id,
        "geometry_assisted": geometry_assisted,
        "cluster_count": len(clusters),
        "clusters": cluster_info,
        "prompt": _prompt,
        "instruction": (
            "Use the clusters and summaries above to decide zone names and hierarchy. "
            "Then call save_node for each node you create, starting from leaves up to root."
        ),
    }


def save_node_tool(scene_id: str, node_dict: dict) -> Dict[str, Any]:
    """Persist a tree node that Cursor AI has created."""
    scene_dir = str(DATA_DIR / scene_id)
    try:
        node = TreeNode.from_dict(node_dict)

        # Attach camera positions for spatial stats
        tf_path = DATA_DIR / scene_id / "transforms.json"
        if tf_path.exists():
            scene = load_transforms(str(tf_path))
            pos_map = {f.view_id: f.camera_position() for f in scene.frames if f.view_id}
            node.camera_positions = [pos_map[v] for v in node.view_ids if v in pos_map]
            node.compute_spatial_stats()

        save_node(scene_dir, node)
        return {"status": "saved", "node_id": node.node_id}
    except Exception as e:
        return {"error": str(e)}


def get_node_tool(scene_id: str, node_id: str) -> Dict[str, Any]:
    """Return full node data."""
    node = load_node(str(DATA_DIR / scene_id), node_id)
    if node is None:
        return {"error": f"Node '{node_id}' not found"}
    return node.to_dict()


def get_root_node_tool(scene_id: str) -> Dict[str, Any]:
    """Return the root node."""
    root_id = get_root_node_id(str(DATA_DIR / scene_id))
    if root_id is None:
        return {"error": "No tree found for scene"}
    return get_node_tool(scene_id, root_id)


def get_query_decomposition_tool(query: str) -> Dict[str, Any]:
    """§7.2 — structured query plan prompt (call LLM before traversal)."""
    return build_decomposition_prompt(query)


def get_children_tool(
    scene_id: str,
    node_id: str,
    query: str,
    structured_plan_json: str = "{}",
) -> Dict[str, Any]:
    """§7.3 — children + top-down traversal prompt (spec: descend_into).

    `query` is the original natural-language query. Pass `structured_plan_json`
    as the JSON object string returned from the decomposition step.
    """
    try:
        plan = json.loads(structured_plan_json) if structured_plan_json.strip() else {}
        if not isinstance(plan, dict):
            plan = {}
    except json.JSONDecodeError:
        plan = {}
    out = build_traversal_step(scene_id, node_id, query, plan)
    if "error" in out:
        return out
    return {
        "node_id": out["node_id"],
        "node_name": out["node_name"],
        "node_summary": out["node_summary"],
        "children": out["children"],
        "is_leaf": out["is_leaf"],
        "view_ids": out["view_ids"],
        "prompt": out["prompt"],
        "output_schema_hint": out.get("output_schema_hint"),
        "instruction": out.get("instruction"),
    }


def get_leaf_confirmation_view_tool(
    scene_id: str,
    node_id: str,
    view_id: str,
    query: str,
    structured_plan_json: str = "{}",
) -> Dict[str, Any]:
    """§7.4 — single-view leaf confirmation prompt."""
    try:
        plan = json.loads(structured_plan_json) if structured_plan_json.strip() else {}
        if not isinstance(plan, dict):
            plan = {}
    except json.JSONDecodeError:
        plan = {}
    return build_leaf_confirmation_for_view(scene_id, node_id, view_id, query, plan)


def get_views_in_node_tool(scene_id: str, node_id: str, query: str) -> Dict[str, Any]:
    """Return full ViewJSON + leaf-confirmation prompt for Cursor AI.

    Called when descending to a leaf node to check whether it contains the query object.
    """
    scene_dir = str(DATA_DIR / scene_id)
    node = load_node(scene_dir, node_id)
    if node is None:
        return {"error": f"Node '{node_id}' not found"}

    views_data = []
    for vid in node.view_ids:
        v = load_view_analysis(scene_dir, vid)
        if v:
            views_data.append(v.model_dump())

    return {
        "node_id": node_id,
        "view_ids": node.view_ids,
        "views": views_data,
        "prompt": LEAF_CONFIRMATION_PROMPT.format(
            query=query,
            view_json=json.dumps(views_data, indent=2),
        ),
    }


def get_tree_for_viz(scene_id: str) -> Dict[str, Any]:
    """Return the full tree as a D3-compatible nested dict."""
    d3 = to_d3_tree(str(DATA_DIR / scene_id))
    if d3 is None:
        return {"error": "No tree found"}
    return d3


def search_summaries(scene_id: str, keyword: str) -> List[dict]:
    """Simple keyword search over all node and view summaries."""
    results = []
    scene_dir = str(DATA_DIR / scene_id)
    kw = keyword.lower()

    # Search tree nodes
    for node in load_all_nodes(scene_dir).values():
        if kw in node.name.lower() or kw in node.summary.lower():
            results.append({
                "type": "node",
                "id": node.node_id,
                "name": node.name,
                "summary": node.summary,
            })

    # Search view analyses
    from backend.io.view_store import list_view_summaries as _ls
    for vs in _ls(scene_dir):
        if kw in vs["summary"].lower():
            results.append({
                "type": "view",
                "id": vs["view_id"],
                "summary": vs["summary"],
            })

    return results


# ── Two-stage bbox refinement ────────────────────────────────────────────────

def get_bbox_refinement_tool(
    scene_id: str,
    view_id: str,
    rough_bbox: List[float],
    label: str = "",
    padding: float = 1.0,
) -> Dict[str, Any]:
    """Stage 1 — crop image around rough_bbox + padding, return crop + VLM prompt.

    Workflow:
      1. Call this tool → get crop_b64 + prompt + crop_region.
      2. Send crop_b64 image + prompt to the VLM.
      3. VLM returns bbox_in_crop ([x1,y1,x2,y2] normalised within the crop).
      4. Call finalize_refined_bbox_tool(crop_region, bbox_in_crop) to map back.
    """
    result = crop_for_refinement(scene_id, view_id, rough_bbox, label, padding)
    if result is None:
        return {"error": f"Image not found for view '{view_id}' in scene '{scene_id}'"}
    return result


def finalize_refined_bbox_tool(
    crop_region: List[float],
    bbox_in_crop: List[float],
) -> Dict[str, Any]:
    """Stage 2 — map crop-relative bbox back to original image normalised coords.

    crop_region: [cx1_n, cy1_n, cx2_n, cy2_n] from get_bbox_refinement_tool
    bbox_in_crop: [rx1, ry1, rx2, ry2] returned by the VLM (0-1 within the crop)
    Returns: {"bbox_2d_original": [x1, y1, x2, y2]}
    """
    mapped = map_refined_bbox(crop_region, bbox_in_crop)
    return {"bbox_2d_original": mapped}


# ── Best-view ranking ─────────────────────────────────────────────────────────

def rank_leaf_results_tool(scene_id: str, results: List[dict]) -> Dict[str, Any]:
    """Rank leaf-confirmation results and identify the best view.

    Pass the full list of leaf-check results (found AND not-found alike).
    Returns them sorted best-first with an added `_score` field, plus a
    `best` shortcut pointing to the top-ranked positive result.

    Score factors: confidence, bbox area, centrality, and usable PNG brightness.

    Use this after collecting all leaf checks for a query to decide:
    - object_finding queries  → display only `best`
    - aggregation queries     → display all items where found=true, ordered by score

    Parameters
    ----------
    scene_id : scene identifier (for image-quality scoring)
    results : list of dicts, each must contain:
        view_id   : str
        found     : bool
        confidence: "high" | "medium" | "low"
        bbox_2d   : [x1, y1, x2, y2] | null
        (any extra fields are preserved unchanged)
    """
    ranked = rank_leaf_results(results, scene_id)
    best = next((r for r in ranked if r.get("found")), None)
    return {
        "ranked": ranked,
        "best": best,
        "found_count": sum(1 for r in ranked if r.get("found")),
    }
