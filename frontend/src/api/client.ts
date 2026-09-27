import axios from 'axios'
import type { CapturePayload, SceneInfo, D3TreeNode, QueryResult, BBox3D } from '../types'

const api = axios.create({ baseURL: '/api' })

export const healthApi = {
  get: () => api.get<{ status: string; scenes_dir: string; ply_files: string[] }>('/health'),
}

// ── Scenes ────────────────────────────────────────────────────────────────────

export const scenesApi = {
  list: () => api.get<SceneInfo[]>('/scenes'),
  get: (id: string) => api.get<SceneInfo>(`/scenes/${id}`),
  init: (id: string, opts?: {
    fl_x?: number; fl_y?: number; cx?: number; cy?: number;
    width?: number; height?: number;
  }) => api.post(`/scenes/${id}/init`, opts ?? {}),
}

// ── Captures ──────────────────────────────────────────────────────────────────

export const captureApi = {
  save: (payload: CapturePayload) => api.post('/captures/save', payload),
}

// ── Tree ──────────────────────────────────────────────────────────────────────

export const treeApi = {
  getViz: (sceneId: string) => api.get<D3TreeNode>(`/tree/${sceneId}/viz`),
  getNode: (sceneId: string, nodeId: string) => api.get(`/tree/${sceneId}/node/${nodeId}`),
}

// ── Query ─────────────────────────────────────────────────────────────────────

/** Draw 2-D bbox(es) on a scene image, returns base64 annotated PNG. */
export const annotateApi = {
  annotateView: (
    sceneId: string,
    viewId: string,
    boxes: Array<{ bbox_2d: number[]; label?: string; confidence?: number; color?: string }>,
    maxWidth = 1200,
  ) =>
    api.post<{ view_id: string; annotated_image: string }>(
      `/query/${sceneId}/annotate_view`,
      { view_id: viewId, boxes, max_width: maxWidth },
    ),
}

/** §7.2 — LLM returns structured_plan; then use traversal + leaf_view. */
export const queryApi = {
  decomposition: (sceneId: string, query: string) =>
    api.post<{
      phase: string
      original_query: string
      prompt: string
      output_schema_hint: Record<string, unknown>
      instruction: string
    }>(`/query/${sceneId}/decomposition`, { query }),
  traversal: (
    sceneId: string,
    body: { original_query: string; node_id: string; structured_plan?: Record<string, unknown> | null },
  ) =>
    api.post(`/query/${sceneId}/traversal`, body),
  leafView: (
    sceneId: string,
    body: {
      original_query: string
      node_id: string
      view_id: string
      structured_plan?: Record<string, unknown> | null
    },
  ) => api.post(`/query/${sceneId}/leaf_view`, body),
  /** Dev-only substring scan — not spec §7. */
  runKeywordDemo: (sceneId: string, query: string) =>
    api.post<QueryResult & { mode?: string }>(`/query/${sceneId}/run_keyword_demo`, { query }),
  unproject: (sceneId: string, viewId: string, bboxNorm: number[], label?: string) =>
    api.post<{ bbox_3d: BBox3D; view_id: string }>('/query/unproject', {
      scene_id: sceneId,
      view_id: viewId,
      bbox_norm: bboxNorm,
      label: label ?? '',
    }),
  mergeBboxes: (bboxes: BBox3D[]) =>
    api.post<{ merged_bbox_3d: BBox3D }>('/query/merge_bboxes', { bboxes }),
}

// ── Query log ─────────────────────────────────────────────────────────────────

export const queryLogApi = {
  listSessions: (sceneId: string) =>
    api.get<{ sessions: import('../types').QuerySessionHeader[] }>(`/query-log/${sceneId}/sessions`),
  getSession: (sceneId: string, sessionId: string) =>
    api.get<import('../types').QuerySession>(`/query-log/${sceneId}/sessions/${sessionId}`),
  createSession: (sceneId: string, originalQuery: string) =>
    api.post<{ session_id: string }>(`/query-log/${sceneId}/sessions`, { original_query: originalQuery }),
  runSession: (sceneId: string, originalQuery: string, stepDelaySec = 0.35) =>
    api.post<{ session_id: string; status: string }>(`/query-log/${sceneId}/sessions/run`, {
      original_query: originalQuery,
      step_delay_sec: stepDelaySec,
    }),
  logDecomposition: (sceneId: string, sessionId: string, structuredPlan: Record<string, unknown>, prompt?: string) =>
    api.post(`/query-log/${sceneId}/sessions/log/decomposition`, { session_id: sessionId, structured_plan: structuredPlan, prompt: prompt ?? '' }),
  logTraversal: (sceneId: string, body: {
    session_id: string; node_id: string; node_name: string;
    children_considered: { node_id: string; name: string }[];
    descend_into: string[]; reasoning: string;
  }) => api.post(`/query-log/${sceneId}/sessions/log/traversal`, body),
  logLeafCheck: (sceneId: string, body: {
    session_id: string; node_id: string; node_name: string; view_id: string;
    found: boolean; confidence: string; bbox_2d?: number[] | null;
    matched_object?: string | null; explanation: string;
  }) => api.post(`/query-log/${sceneId}/sessions/log/leaf_check`, body),
  logResult: (sceneId: string, sessionId: string, result: import('../types').QueryResult) =>
    api.post(`/query-log/${sceneId}/sessions/log/result`, { session_id: sessionId, result }),
}

// ── WebSocket helper ──────────────────────────────────────────────────────────

export function createQueryStream(sceneId: string, onEvent: (data: unknown) => void): WebSocket {
  const wsUrl = `ws://${window.location.hostname}:8000/api/query/${sceneId}/stream`
  const ws = new WebSocket(wsUrl)
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)) } catch { /* ignore */ }
  }
  return ws
}
