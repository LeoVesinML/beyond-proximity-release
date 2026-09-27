"""Pydantic schemas shared by the API, MCP tools, and tests."""
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Object-level ─────────────────────────────────────────────────────────────

class ObjectAttributes(BaseModel):
    color: Optional[str] = None
    material: Optional[str] = None
    size: Optional[str] = None
    state: Optional[str] = None
    location_description: Optional[str] = None


class ObjectEntry(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_2d: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2 normalised 0-1
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)


# ── View-level ────────────────────────────────────────────────────────────────

# Valid room types — extend here when new scene types are encountered.
ROOM_TYPE_VALUES = (
    "living_room", "bedroom", "kitchen", "bathroom", "corridor",
    "outdoor", "ballroom", "bar", "reception", "stage", "back_of_house",
    "foyer", "lounge", "service_area", "unknown",
)

# Valid facing directions — which functional area the camera is aimed at.
FACING_VALUES = (
    "stage", "audience", "entrance", "exit", "bar", "reception",
    "screen", "wall", "ceiling", "window", "unknown",
)


class ViewJSON(BaseModel):
    view_id: str
    scene_summary: str
    room_type: str = "unknown"
    lighting: str = "unknown"
    objects: List[ObjectEntry] = Field(default_factory=list)
    spatial_relations: List[str] = Field(default_factory=list)
    functional_context: Optional[str] = None

    # ── Semantic room-boundary fields (Fix 2) ─────────────────────────────
    # `facing` describes the primary subject / functional area the camera
    # points toward.  Used by the semantic merge pass to detect same-room
    # clusters that are geometrically separated (e.g. audience vs stage).
    facing: str = "unknown"

    # `visible_landmarks` lists prominent, named architectural or functional
    # features that are visible in this view (even in the distance).  Shared
    # landmarks across two geometric clusters are strong evidence of same room.
    visible_landmarks: List[str] = Field(default_factory=list)


# ── Capture payload (browser → API) ──────────────────────────────────────────

class CapturePayload(BaseModel):
    scene_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    view_id: str = Field(pattern=r"^v[0-9]+$")
    rgb_b64: str          # base64-encoded PNG
    depth_b64: str        # base64-encoded 32-bit float raw bytes (little-endian)
    transform_matrix: List[List[float]]   # 4×4 camera-to-world
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_ply_url: Optional[str] = None
    captured_at_utc: Optional[str] = None
    pose_coordinate_convention: str = "threejs_world_camera_to_world_rh_y_up_camera_forward_minus_z"
    depth_source: str = "unknown"
    depth_near: float = 0.1
    depth_far: float = 100.0
    depth_valid_pixel_count: Optional[int] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None


# ── 3-D bounding box ─────────────────────────────────────────────────────────

class BBox3D(BaseModel):
    center: Tuple[float, float, float]
    size:   Tuple[float, float, float]
    label:  str = ""


# ── Query plan (from Cursor AI) ──────────────────────────────────────────────

class QueryPlan(BaseModel):
    original_query: str
    target_label: str
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)
    spatial_constraint: Optional[str] = None
    functional_constraint: Optional[str] = None


# ── Query result ─────────────────────────────────────────────────────────────

class QueryResult(BaseModel):
    query: str
    found: bool
    view_id: Optional[str] = None
    bbox_3d: Optional[BBox3D] = None
    camera_pose: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    explanation: str = ""


# ── WebSocket traversal event ─────────────────────────────────────────────────

class TraversalEvent(BaseModel):
    event: str   # "start" | "visit_node" | "descend" | "leaf_check" | "found" | "not_found"
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    view_id: Optional[str] = None
    message: Optional[str] = None


# ── Prompt templates returned by MCP tools to Cursor AI ──────────────────────

VIEW_ANALYSIS_PROMPT = """\
You are analyzing a rendered view from a 3D Gaussian Splatting scene.

Carefully examine the image and produce a structured JSON response with ALL of the following fields:
{
  "view_id": "<view_id>",
  "scene_summary": "<one or two sentence description of what is visible>",
  "room_type": "<see valid values below>",
  "lighting": "<bright|dim|dark|natural|artificial|unknown>",
  "facing": "<see valid values below>",
  "visible_landmarks": ["<landmark name>", ...],
  "objects": [
    {
      "label": "<object name>",
      "confidence": <0.0-1.0>,
      "bbox_2d": [<x1_norm>, <y1_norm>, <x2_norm>, <y2_norm>],
      "attributes": {
        "color": "<color or null>",
        "material": "<material or null>",
        "size": "<small|medium|large or null>",
        "state": "<open|closed|on|off or null>",
        "location_description": "<e.g. on the table, near the window or null>"
      }
    }
  ],
  "spatial_relations": ["<object A> is to the left of <object B>", ...],
  "functional_context": "<one sentence: what is this space used for>"
}

━━━  FIELD DEFINITIONS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

room_type — the functional type of the ROOM THE CAMERA IS PHYSICALLY INSIDE.
  Choose exactly one:
  living_room | bedroom | kitchen | bathroom | corridor | foyer | lounge |
  ballroom | bar | reception | stage | back_of_house | service_area | outdoor | unknown

  Rules:
  - Use "ballroom" for any large event/banquet hall, even if captured from the stage end.
  - Use "bar" for a view centred on a bar or drinks counter.
  - Use "foyer" for lobby-entrance transitions before the main hall.
  - Use "back_of_house" or "service_area" for catering prep, storage, or utility areas.
  - Do NOT use "living_room" for hotel or venue spaces; use the most specific term.

facing — the primary functional area or architectural feature the camera POINTS TOWARD.
  Choose exactly one:
  stage | audience | entrance | exit | bar | reception | screen | wall | ceiling | window | unknown

  Rules:
  - "stage"    — camera faces a stage front, projection screen, or podium wall.
  - "audience" — camera is at/near the stage and faces the audience seating or tables.
  - "entrance" — camera faces the room's main entry point or doorway.
  - "exit"     — camera faces an egress door or emergency exit.
  - "bar"      — camera faces a bar counter.
  - "reception"— camera faces a reception desk or check-in area.
  - "screen"   — camera faces a standalone projection or display screen.
  - "ceiling"  — camera is tilted upward, primarily capturing the ceiling.

visible_landmarks — list every NAMED architectural or functional landmark VISIBLE IN THE IMAGE,
  even if small or in the background. Examples:
  "projection screen", "grand piano", "reception desk", "bar counter",
  "crystal chandelier", "ornate double doors", "willow mural", "banquet tables",
  "marble columns", "exit sign", "fire extinguisher".

  Rules:
  - Include landmarks visible in the distance (e.g. a projection screen seen from the back of the room).
  - Use consistent, canonical names (always "projection screen", not "screen" or "projector screen").
  - List at least one landmark per view; list up to six.

━━━  bbox_2d ACCURACY INSTRUCTIONS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EVERY object, derive bbox_2d using this step-by-step process BEFORE writing the number:
  Step 1 — Horizontal: divide the image into thirds (left 0-0.33, centre 0.33-0.67, right 0.67-1.0).
            Where does the object's left edge fall? Where does its right edge fall?
  Step 2 — Vertical: divide the image into thirds (top 0-0.33, middle 0.33-0.67, bottom 0.67-1.0).
            Where does the object's top edge fall? Where does its bottom edge fall?
  Step 3 — Refine to nearest 0.05 within that third.
  Step 4 — Sanity check: for "above the door" → your y2 must be ≤ the door's y1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rules:
- bbox_2d values are normalised 0-1 (x1/y1 = top-left, x2/y2 = bottom-right).
- Include every clearly visible object; omit vague blobs.
- Return ONLY valid JSON — no markdown fences, no extra text.\
"""

TREE_GROUPING_PROMPT = """\
You are organising {n} visual observations from a 3D Gaussian Splatting scene into a \
semantic hierarchy.

━━━  INPUT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each cluster below was produced by a camera-position proximity algorithm AND a
semantic-merge pass, but the merge pass is conservative.  Your job is to apply
a final layer of human-level semantic reasoning before naming zones.

Spatial clusters:
{clusters_json}

━━━  STEP 0 — SAME-ROOM CHECK (mandatory, run before naming anything)  ━━━━━━━

Before creating any zone, inspect every pair of clusters and ask:
"Could these two separate clusters be different camera positions INSIDE THE SAME
physical room?"

A pair MUST be merged if ANY of the following hold:
  a) Both clusters share the same room_type AND share at least one visible_landmark.
  b) One cluster contains a view with facing="stage" and the other contains a view
     with facing="audience" (or vice versa) — indicating opposite ends of one hall.
  c) A dominant landmark in cluster A (e.g. "projection screen", "stage") appears
     in visible_landmarks of cluster B views, even if cluster B's facing is different.

If you decide two clusters belong to the same room, MERGE them before proceeding.
Record your merge decision briefly in the "merge_reasoning" field.

━━━  STEP 1 — NAMING & SUMMARISING ZONES  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each final cluster (after any merges), decide:
  1. A concise, human-readable zone name using real-world venue/room vocabulary.
     Good: "Ballroom", "Lobby & Bar", "Reception Corridor"
     Bad : "Cluster 0", "Area 1", "Zone near v012"
  2. A one-sentence summary of the zone's function and main features.

Zone-naming heuristics:
  - If the majority of views share room_type="ballroom" → name it "Ballroom" (or sub-divide only if
    there are clear functional sub-areas, e.g. "Stage" vs "Seating Area").
  - If views mix room_type="bar" + "reception" + "corridor" → they likely form one unified lobby/entrance
    space; name it "Lobby & Bar" or "Reception Area".
  - Never create a separate zone purely because the cluster was geometrically isolated — apply Step 0 first.

━━━  STEP 2 — OPTIONAL SUB-GROUPING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sub-group a zone only when it is large AND contains genuinely distinct functional
sub-areas (e.g. a ballroom zone with both stage views and audience-seating views
may optionally have "Stage" and "Seating Area" as regions).

━━━  OUTPUT FORMAT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a JSON array of node descriptors:
[
  {{
    "node_type": "zone",
    "name": "<zone name>",
    "summary": "<one-sentence summary>",
    "merge_reasoning": "<why clusters were merged, or 'no merge' if single cluster>",
    "view_ids": [<all view_ids in this zone, including any merged clusters>],
    "children": [
      {{
        "node_type": "region",
        "name": "<region name>",
        "summary": "<one-sentence summary>",
        "view_ids": [<subset of view_ids>]
      }}
    ]
  }},
  ...
]

Rules:
- Every view_id must appear in exactly one leaf-level node (zone if no children, else region).
- Use ONLY the provided data — do not invent rooms, objects, or relationships.
- "children" array may be omitted or empty if the zone has no sub-groups.
- Return ONLY valid JSON array — no markdown fences, no extra text.\
"""

# §7.2 Query decomposition (structured plan before traversal)
QUERY_DECOMPOSITION_PROMPT = """\
Analyze this query for a 3D scene search system.
Query: '__QUERY__'

Extract:
- target: the primary object or subject being sought
- location_constraints: explicit spatial scope mentioned in the query (or null)
- attribute_constraints: color, size, material, style, etc. (object or null)
- relational_constraints: spatial relationships (near X, on top of Y, to the left of Z) (list or null)
- functional_constraints: what the object is used for (or null)
- query_type: one of [object_finding, descriptive, aggregation, cross_zone_geometric, spatial_relation]
- output_type: one of [3d_bbox, camera_pose, text_answer, count]

Return ONLY valid JSON with those keys.\
"""

# §7.3 Top-down traversal (conservative pruning at each internal node)
TRAVERSAL_TOPDOWN_PROMPT = """\
You are navigating a scene hierarchy to answer:
  "__ORIGINAL_QUERY__"

Query plan (from decomposition step):
__STRUCTURED_PLAN_JSON__

Current node: "__CURRENT_NODE_NAME__"
Summary: "__CURRENT_NODE_SUMMARY__"

Children nodes:
__CHILDREN_BLOCK__

Which children are potentially relevant to answering this query?
- Include a child if it MIGHT contain the answer (be conservative — don't prune if uncertain)
- Exclude a child only if it is CLEARLY irrelevant
- If NO children are relevant, return an empty list (the target does not exist in this branch)

Return JSON: {{"descend_into": ["<node_id>", "..."], "reasoning": "<one sentence>"}}

Return ONLY valid JSON.\
"""

# Legacy alias — same flow as TRAVERSAL_TOPDOWN but uses "visit" key (older clients)
TRAVERSAL_STEP_PROMPT = """\
You are navigating a semantic scene hierarchy to answer: "{query}"

Current node: "{node_name}"
Node summary: "{node_summary}"

Children:
{children_json}

Decide which children to visit next (may be more than one if ambiguous).
Respond with JSON:
{{
  "visit": [<list of node_ids to descend into>],
  "reasoning": "<one sentence>"
}}

Return ONLY valid JSON.\
"""

# §7.4 Leaf confirmation — single view (preferred for precise bbox extraction)
LEAF_CONFIRMATION_SINGLE_VIEW_PROMPT = """\
Query: '__QUERY__'

Query plan:
__STRUCTURED_PLAN_JSON__

Here is the full description of view __VIEW_ID__:
__FULL_VIEW_JSON__

Task:
1. Does this view contain the queried object/subject? (yes/no/partial)
2. If yes or partial: derive a PRECISE bbox_2d using the spatial reasoning steps below.
   DO NOT blindly copy bbox_2d numbers from the JSON — re-derive them from the scene_summary,
   objects[].attributes.location_description, and spatial_relations fields.

   Spatial reasoning steps (work through these in order):
   a) Identify the matched object entry and read its location_description.
   b) Horizontal: which third of the image is the object in? (left 0-0.33 / centre 0.33-0.67 / right 0.67-1.0)
      Estimate x1 and x2 to the nearest 0.05.
   c) Vertical: which third of the image is the object in? (top 0-0.33 / middle 0.33-0.67 / bottom 0.67-1.0)
      Estimate y1 and y2 to the nearest 0.05.
   d) Sanity check against spatial_relations: if "above <ref_object>", your y2 must be ≤ ref_object's y1.
      Adjust if violated.
   e) Scale check: the bbox area should be proportional to the object size attribute
      (small ≤ 5% of image area, medium ≤ 20%, large > 20%).

3. Rate confidence: high/medium/low

Return JSON:
{{
  "found": <true|false>,
  "match": "yes|no|partial",
  "confidence": "high|medium|low",
  "matched_object": "<string or null>",
  "bbox_2d": [<x1_norm>, <y1_norm>, <x2_norm>, <y2_norm>] | null,
  "bbox_reasoning": "<brief summary of the spatial reasoning steps you applied>",
  "explanation": "<one sentence about what was found>"
}}

Return ONLY valid JSON.\
"""

# Batch leaf check (all views in a leaf node — legacy / quick scan)
LEAF_CONFIRMATION_PROMPT = """\
Query: "{query}"

Full view description:
{view_json}

Does this view contain the queried object (accounting for all attributes and context)?
Respond with JSON:
{{
  "found": <true|false>,
  "confidence": <0.0-1.0>,
  "matched_object_label": "<label or null>",
  "bbox_2d": [<x1_norm>, <y1_norm>, <x2_norm>, <y2_norm>],
  "explanation": "<one sentence>"
}}

Return ONLY valid JSON.\
"""
