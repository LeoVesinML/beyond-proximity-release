"""Camera proximity graph and spatial clustering for view grouping.

Two-pass clustering strategy
─────────────────────────────
Pass 1 — Geometric:   build_adjacency / connected_components form clusters purely from
          camera-position proximity.  Fast and deterministic but can split a single large
          room into multiple clusters (e.g. stage end vs audience end of a ballroom).

Pass 2 — Semantic merge:  semantic_merge_clusters inspects per-view analysis metadata
          (room_type, facing, visible_landmarks) and merges geometrically separate
          clusters that clearly belong to the same physical room.  Rules are explicit and
          conservative — a merge only happens when two independent signals agree.

Call build_spatial_clusters (which applies both passes) rather than using the
individual helpers directly.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from backend.config import SPATIAL_SCALE_FACTOR


def _positions_array(positions: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    """Return (N×3 array, ordered list of view_ids)."""
    ids = sorted(positions.keys())
    arr = np.array([positions[i] for i in ids], dtype=float)
    return arr, ids


def build_adjacency(
    positions: Dict[str, List[float]],
    scale_factor: float = SPATIAL_SCALE_FACTOR,
) -> Dict[str, List[str]]:
    """Return adjacency dict {view_id: [neighbour_ids]} using adaptive threshold.

    Threshold = scale_factor × median of each point's nearest-neighbour distance.
    """
    if len(positions) < 2:
        return {k: [] for k in positions}

    arr, ids = _positions_array(positions)
    tree = KDTree(arr)

    # k=2: query the point itself + one nearest neighbour
    dists, _ = tree.query(arr, k=min(2, len(arr)))
    if dists.ndim == 1:
        nn_dists = dists
    else:
        nn_dists = dists[:, 1]

    threshold = scale_factor * float(np.median(nn_dists))

    # Build adjacency within threshold
    pairs = tree.query_pairs(threshold)
    adj: Dict[str, List[str]] = {v: [] for v in ids}
    for i, j in pairs:
        adj[ids[i]].append(ids[j])
        adj[ids[j]].append(ids[i])

    return adj


def connected_components(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Return list of connected components (each is a list of view_ids)."""
    visited: set = set()
    components: List[List[str]] = []

    for start in adj:
        if start in visited:
            continue
        component: List[str] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(n for n in adj[node] if n not in visited)
        components.append(sorted(component))

    return components


def build_spatial_clusters(
    positions: Dict[str, List[float]],
    scale_factor: float = SPATIAL_SCALE_FACTOR,
    view_summaries: Optional[List[dict]] = None,
) -> List[List[str]]:
    """Two-pass clustering: geometric proximity → semantic merge.

    Parameters
    ----------
    positions:
        Mapping of view_id → [x, y, z] camera position.
    scale_factor:
        Controls the adjacency threshold (see build_adjacency).
    view_summaries:
        Optional list of per-view summary dicts (from list_view_summaries).
        When provided a second semantic-merge pass is applied that can merge
        geometrically isolated clusters that share the same physical room.
    """
    adj = build_adjacency(positions, scale_factor)
    clusters = connected_components(adj)

    if view_summaries:
        clusters = semantic_merge_clusters(clusters, view_summaries)

    return clusters


# ─── Semantic merge pass ──────────────────────────────────────────────────────

# Complementary facing pairs:  two clusters face *toward each other* across a
# shared room (e.g. stage ↔ audience, entrance ↔ exit).  Matching one of these
# pairs is strong evidence that the two clusters occupy the same room.
_COMPLEMENTARY_FACING = frozenset({
    frozenset({"stage", "audience"}),
    frozenset({"entrance", "exit"}),
    frozenset({"bar", "reception"}),
})

# How many distinct visible_landmarks two clusters must share before we merge.
_LANDMARK_OVERLAP_THRESHOLD = 1


def _cluster_room_types(cluster: List[str], meta: Dict[str, dict]) -> List[str]:
    """Return deduplicated room_type values for the views in *cluster*."""
    seen: set = set()
    out: List[str] = []
    for vid in cluster:
        rt = meta.get(vid, {}).get("room_type", "unknown")
        if rt and rt != "unknown" and rt not in seen:
            seen.add(rt)
            out.append(rt)
    return out


def _cluster_facings(cluster: List[str], meta: Dict[str, dict]) -> List[str]:
    """Return deduplicated facing values for the views in *cluster*."""
    seen: set = set()
    out: List[str] = []
    for vid in cluster:
        f = meta.get(vid, {}).get("facing", "unknown")
        if f and f != "unknown" and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _cluster_landmarks(cluster: List[str], meta: Dict[str, dict]) -> set:
    """Return the union of visible_landmarks for all views in *cluster*."""
    landmarks: set = set()
    for vid in cluster:
        for lm in meta.get(vid, {}).get("visible_landmarks", []):
            if lm:
                landmarks.add(lm.lower().strip())
    return landmarks


def _should_merge(
    a: List[str],
    b: List[str],
    meta: Dict[str, dict],
) -> bool:
    """Return True when clusters *a* and *b* should be merged.

    Merge criterion (either of the following is sufficient):

    Rule 1 — Complementary facing:
      The dominant facing of cluster A and cluster B form a known complementary
      pair (e.g. "stage" + "audience").  This reliably indicates that the two
      clusters photographed the same large room from opposite ends.

    Rule 2 — Same room_type + shared landmark:
      Both clusters share the same dominant room_type AND have at least
      _LANDMARK_OVERLAP_THRESHOLD landmarks in common.
    """
    rt_a = set(_cluster_room_types(a, meta))
    rt_b = set(_cluster_room_types(b, meta))
    facings_a = set(_cluster_facings(a, meta))
    facings_b = set(_cluster_facings(b, meta))
    landmarks_a = _cluster_landmarks(a, meta)
    landmarks_b = _cluster_landmarks(b, meta)

    # Rule 1: complementary facing
    for fa in facings_a:
        for fb in facings_b:
            if frozenset({fa, fb}) in _COMPLEMENTARY_FACING:
                return True

    # Rule 2: shared room_type + shared landmark
    shared_rt = rt_a & rt_b
    if shared_rt:
        shared_lm = landmarks_a & landmarks_b
        if len(shared_lm) >= _LANDMARK_OVERLAP_THRESHOLD:
            return True

    return False


def semantic_merge_clusters(
    clusters: List[List[str]],
    view_summaries: List[dict],
) -> List[List[str]]:
    """Merge geometrically separate clusters that belong to the same room.

    Uses an iterative Union-Find approach so transitive merges are handled
    (A↔B and B↔C causes A, B, C to all merge even if A↔C doesn't trigger).

    Parameters
    ----------
    clusters:
        Output of connected_components — list of view-id lists.
    view_summaries:
        List of dicts returned by io.view_store.list_view_summaries.
        Must include: view_id, room_type, facing, visible_landmarks.
    """
    if len(clusters) <= 1:
        return clusters

    # Build a lookup: view_id → summary dict
    meta: Dict[str, dict] = {s["view_id"]: s for s in view_summaries}

    # Union-Find over cluster indices
    parent = list(range(len(clusters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if find(i) != find(j) and _should_merge(clusters[i], clusters[j], meta):
                union(i, j)

    # Collect merged clusters
    groups: Dict[int, List[str]] = {}
    for idx, cluster in enumerate(clusters):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].extend(cluster)

    return [sorted(v) for v in groups.values()]


def cluster_centroid(view_ids: List[str], positions: Dict[str, List[float]]) -> List[float]:
    """Compute the centroid of a cluster."""
    pts = np.array([positions[v] for v in view_ids if v in positions])
    if len(pts) == 0:
        return [0.0, 0.0, 0.0]
    return pts.mean(axis=0).tolist()
