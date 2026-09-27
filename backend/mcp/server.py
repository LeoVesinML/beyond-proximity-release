"""fastmcp server entry point — exposes all infrastructure tools to Cursor AI.

All tools here are PURE INFRASTRUCTURE.  No LLM calls are made inside the backend.
Cursor AI drives all intelligence by calling these tools and using its own vision/language.

Start with:
    conda activate semanticsplat
    python -m backend.mcp.server
"""
from fastmcp import FastMCP

from backend.mcp.tools.scene_tools import list_scenes, get_scene_info, init_scene
from backend.mcp.tools.view_tools import (
    get_unanalyzed_views,
    get_view_image,
    save_view_analysis_tool,
    get_view_full_json,
    list_view_summaries_tool,
    get_view_camera_pose,
)
from backend.mcp.tools.tree_tools import (
    build_spatial_clusters_tool,
    save_node_tool,
    get_node_tool,
    get_root_node_tool,
    get_children_tool,
    get_query_decomposition_tool,
    get_leaf_confirmation_view_tool,
    get_views_in_node_tool,
    get_tree_for_viz,
    search_summaries,
    get_bbox_refinement_tool,
    finalize_refined_bbox_tool,
    rank_leaf_results_tool,
)
from backend.mcp.tools.query_tools import (
    unproject_bbox_tool,
    merge_bboxes_tool,
    get_capture_intrinsics,
    compute_inter_node_distances,
)
from backend.mcp.tools.agent_trace_tools import (
    start_agent_query_trace_tool,
    record_agent_query_step_tool,
    finish_agent_query_trace_tool,
    get_agent_query_trace_tool,
)

mcp = FastMCP(
    name="semantic-splat",
    instructions=(
        "Infrastructure tools for the SemanticSplat 3DGS navigator. "
        "Tools handle file I/O, spatial geometry, and data retrieval. "
        "Cursor AI provides all intelligence (vision, language, reasoning). "
        "For every query, call start_agent_query_trace before decomposition, "
        "record_agent_query_step after each model decision, and "
        "finish_agent_query_trace exactly once."
    ),
)

# ── Scene tools ───────────────────────────────────────────────────────────────
mcp.tool()(list_scenes)
mcp.tool()(get_scene_info)
mcp.tool()(init_scene)

# ── View tools ────────────────────────────────────────────────────────────────
mcp.tool()(get_unanalyzed_views)
mcp.tool()(get_view_image)
mcp.tool(name="save_view_analysis")(save_view_analysis_tool)
mcp.tool(name="get_view_full_json")(get_view_full_json)
mcp.tool(name="list_view_summaries")(list_view_summaries_tool)
mcp.tool()(get_view_camera_pose)

# ── Tree tools ────────────────────────────────────────────────────────────────
mcp.tool(name="build_spatial_clusters")(build_spatial_clusters_tool)
mcp.tool(name="save_node")(save_node_tool)
mcp.tool(name="get_node")(get_node_tool)
mcp.tool(name="get_root_node")(get_root_node_tool)
mcp.tool(name="get_children")(get_children_tool)
mcp.tool(name="get_query_decomposition")(get_query_decomposition_tool)
mcp.tool(name="get_leaf_confirmation_view")(get_leaf_confirmation_view_tool)
mcp.tool(name="get_views_in_node")(get_views_in_node_tool)
mcp.tool()(get_tree_for_viz)
mcp.tool()(search_summaries)

# ── Bbox refinement tools ─────────────────────────────────────────────────────
mcp.tool(name="get_bbox_refinement")(get_bbox_refinement_tool)
mcp.tool(name="finalize_refined_bbox")(finalize_refined_bbox_tool)
mcp.tool(name="rank_leaf_results")(rank_leaf_results_tool)

# ── Query / geometry tools ────────────────────────────────────────────────────
mcp.tool(name="unproject_bbox")(unproject_bbox_tool)
mcp.tool(name="merge_bboxes")(merge_bboxes_tool)
mcp.tool()(get_capture_intrinsics)
mcp.tool()(compute_inter_node_distances)

# Agent trace tools
mcp.tool(name="start_agent_query_trace")(start_agent_query_trace_tool)
mcp.tool(name="record_agent_query_step")(record_agent_query_step_tool)
mcp.tool(name="finish_agent_query_trace")(finish_agent_query_trace_tool)
mcp.tool(name="get_agent_query_trace")(get_agent_query_trace_tool)

if __name__ == "__main__":
    from backend.config import MCP_HOST, MCP_PORT
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
