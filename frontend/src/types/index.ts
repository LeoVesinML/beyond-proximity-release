// Mirrors backend Pydantic schemas

export interface ObjectAttributes {
  color?: string | null;
  material?: string | null;
  size?: string | null;
  state?: string | null;
  location_description?: string | null;
}

export interface ObjectEntry {
  label: string;
  confidence: number;
  bbox_2d?: [number, number, number, number] | null;
  attributes: ObjectAttributes;
}

export interface ViewJSON {
  view_id: string;
  scene_summary: string;
  room_type: string;
  lighting: string;
  objects: ObjectEntry[];
  spatial_relations: string[];
  functional_context?: string | null;
}

export interface FrameEntry {
  file_path: string;
  transform_matrix: number[][];
  view_id: string;
  depth_file_path?: string | null;
}

export interface SceneInfo {
  scene_id: string;
  path: string;
  intrinsics: {
    fl_x: number; fl_y: number;
    cx: number; cy: number;
    w: number; h: number;
  };
  frames: Array<{
    view_id: string;
    file_path: string;
    position: [number, number, number];
    has_depth: boolean;
  }>;
  analyzed_views: string[];
  tree: Record<string, unknown>;
}

export interface CapturedView {
  view_id: string;
  thumbnail: string;          // base64 data URL
  position: [number, number, number];
  timestamp: number;
}

export interface CapturePayload {
  scene_id: string;
  view_id: string;
  rgb_b64: string;
  depth_b64: string;
  transform_matrix: number[][];
  fl_x: number;
  fl_y: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  source_ply_url?: string | null;
  captured_at_utc: string;
  pose_coordinate_convention: string;
  depth_source: string;
  depth_near: number;
  depth_far: number;
  depth_valid_pixel_count: number;
  depth_min: number;
  depth_max: number;
}

export interface BBox3D {
  center: [number, number, number];
  size: [number, number, number];
  label: string;
}

export interface QueryResult {
  query: string;
  found: boolean;
  view_id?: string | null;
  bbox_3d?: BBox3D | null;
  camera_pose?: Record<string, unknown> | null;
  confidence: number;
  explanation: string;
  /** §7 step 1 — show info panel, not "not found". */
  pipeline_phase?: 'decomposition' | 'found' | 'not_found';
  /** base64 data URL of the annotated image (bbox drawn) */
  annotated_image?: string | null;
}

// Tree visualisation

export interface D3TreeNode {
  id: string;
  name: string;
  type: 'root' | 'zone' | 'region' | 'leaf';
  summary: string;
  view_ids: string[];
  centroid: [number, number, number];
  children?: D3TreeNode[];
}

// Query traversal event (WebSocket)
export interface TraversalEvent {
  event: 'start' | 'visit_node' | 'descend' | 'leaf_check' | 'found' | 'not_found';
  node_id?: string;
  node_name?: string;
  view_id?: string;
  message?: string;
}

// ── Query session log (§7 pipeline trace) ────────────────────────────────────

export interface QueryStepDecomposition {
  step_type: 'decomposition';
  ts: string;
  structured_plan: Record<string, unknown>;
  prompt_excerpt: string;
}

export interface QueryStepTraversal {
  step_type: 'traversal';
  ts: string;
  node_id: string;
  node_name: string;
  children_count: number;
  children_names: string[];
  descend_into: string[];
  reasoning: string;
}

export interface QueryStepLeafCheck {
  step_type: 'leaf_check';
  ts: string;
  node_id: string;
  node_name: string;
  view_id: string;
  found: boolean;
  confidence: string;
  bbox_2d: number[] | null;
  matched_object: string | null;
  explanation: string;
  /** base64 data URL loaded lazily when the card is viewed */
  annotated_image?: string | null;
}

export interface QueryStepResult {
  step_type: 'found' | 'not_found';
  ts: string;
  view_id: string | null;
  explanation: string;
  confidence: number;
  bbox_3d: BBox3D | null;
}

export interface QueryStepError {
  step_type: 'error';
  ts: string;
  message: string;
}

export type QueryStep =
  | QueryStepDecomposition
  | QueryStepTraversal
  | QueryStepLeafCheck
  | QueryStepResult
  | QueryStepError;

export interface QuerySessionHeader {
  session_id: string;
  original_query: string;
  started_at: string;
  finished_at: string | null;
  step_count: number;
  found: boolean | null;
  source?: string | null;
}

export interface QuerySession {
  session_id: string;
  scene_id: string;
  original_query: string;
  started_at: string;
  finished_at: string | null;
  steps: QueryStep[];
  result: QueryResult | null;
}
