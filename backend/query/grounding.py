"""Object-level grounding search variants with transparent scoring and fallback.

The public entry point is :func:`search_grounding`.  All variants rank the
same canonical candidates; graph modes only reduce the candidate IDs passed to
the scorer.  Embeddings are optional and never degrade silently to lexical
search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from backend.query.affordance import expand_query_tokens, tokenize


SEARCH_VARIANTS = frozenset({"graph", "graph_fallback", "flat_lexical", "flat_embedding"})
RELATIONS = (
    "near",
    "closest",
    "farthest",
    "between",
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
)
DEFAULT_ANCHOR_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_ANCHOR_MARGIN_THRESHOLD = 0.10
DEFAULT_RELATION_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_RELATION_SCORE_MARGIN = 0.15
DEFAULT_RELATION_BOOST = 0.20

_SYNONYMS: dict[str, frozenset[str]] = {
    "couch": frozenset({"sofa", "settee", "seating"}),
    "sofa": frozenset({"couch", "settee", "seating"}),
    "chair": frozenset({"seat", "seating", "armchair"}),
    "door": frozenset({"doorway", "entrance", "exit"}),
    "doorway": frozenset({"door", "entrance", "exit"}),
    "desk": frozenset({"counter", "reception", "table"}),
    "counter": frozenset({"desk", "reception"}),
    "picture": frozenset({"artwork", "painting", "image"}),
    "artwork": frozenset({"picture", "painting", "display"}),
    "lamp": frozenset({"light", "lighting"}),
    "screen": frozenset({"display", "projection"}),
    "table": frozenset({"desk", "counter"}),
}

_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("left", re.compile(r"\b(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(.+?)(?:[?.!,]|$)", re.I)),
    ("right", re.compile(r"\b(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(.+?)(?:[?.!,]|$)", re.I)),
    ("front", re.compile(r"\b(.+?)\s+(?:is\s+)?(?:in\s+)?front\s+of\s+(.+?)(?:[?.!,]|$)", re.I)),
    ("behind", re.compile(r"\b(.+?)\s+(?:is\s+)?behind\s+(.+?)(?:[?.!,]|$)", re.I)),
    (
        "above",
        re.compile(
            r"\b(.+?)\s+(?:is\s+)?(?:above|on\s+top\s+of)\s+"
            r"(.+?)(?:[?.!,]|$)",
            re.I,
        ),
    ),
    ("below", re.compile(r"\b(.+?)\s+(?:is\s+)?below\s+(.+?)(?:[?.!,]|$)", re.I)),
    ("near", re.compile(r"\b(.+?)\s+(?:is\s+)?near\s+(.+?)(?:[?.!,]|$)", re.I)),
)
_PREFIX_RELATION = re.compile(
    r"\b(?P<relation>near|left of|right of|in front of|behind|above|below)\s+"
    r"(?:the\s+)?(?P<anchor>[a-z0-9][a-z0-9 _-]*?)(?:[?.!,]|$)",
    re.I,
)
_BETWEEN_RELATION = re.compile(
    r"\b(.+?)\s+(?:that\s+(?:is|are)\s+)?"
    r"(?:between|in\s+the\s+(?:center|middle)\s+of)\s+"
    r"(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?:[?.!,]|$)",
    re.I,
)
_DISTANCE_RELATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "closest",
        re.compile(
            r"\b(.+?)\s+(?:that\s+(?:is|are)\s+)?"
            r"(?:is\s+)?(?:closer|closest|close|nearest)\s+to\s+"
            r"(?:the\s+)?(.+?)(?:[?.!,]|$)",
            re.I,
        ),
    ),
    (
        "farthest",
        re.compile(
            r"\b(.+?)\s+(?:that\s+(?:is|are)\s+)?"
            r"(?:is\s+)?(?:farther|farthest|furthest|far(?:\s+away)?)\s+from\s+"
            r"(?:the\s+)?(.+?)(?:[?.!,]|$)",
            re.I,
        ),
    ),
)
_QUERY_FILLER = frozenset(
    {
        "find",
        "select",
        "choose",
        "where",
        "is",
        "are",
        "that",
        "which",
        "the",
        "a",
        "an",
        "object",
        "thing",
        "located",
        "please",
        "show",
        "me",
    }
)


class Embedder(Protocol):
    """Minimal sentence-transformer-compatible protocol."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class RelationQuery:
    target: str
    relation: str | None = None
    anchor: str | None = None
    anchor_secondary: str | None = None


@dataclass
class GroundingCandidate:
    """One canonical object observation from a semantic view/tree."""

    object_id: str
    node_id: str | None
    view_id: str
    ancestor_node_ids: list[str]
    label: str
    text: str
    bbox_2d: tuple[float, float, float, float] | None
    bbox_3d: dict[str, Any] | None
    relations: list[str]
    score: float = 0.0
    score_details: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return self.object_id

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_id"] = self.candidate_id
        return result


@dataclass
class GroundingSearchResult:
    status: str
    variant: str
    query: str
    parsed_query: RelationQuery
    candidates: list[GroundingCandidate]
    candidate_count: int
    searched_candidate_count: int
    fallback_triggered: bool = False
    fallback_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "variant": self.variant,
            "query": self.query,
            "parsed_query": asdict(self.parsed_query),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": self.candidate_count,
            "searched_candidate_count": self.searched_candidate_count,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reasons": self.fallback_reasons,
            "metadata": self.metadata,
        }


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {}


def _object_entries(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = view.get("visible_objects", view.get("objects", []))
    return [_as_mapping(item) for item in raw] if isinstance(raw, list) else []


def _normal_label(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tuple4(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _tree_maps(
    tree: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    nodes = {str(key): _as_mapping(value) for key, value in tree.items()}
    object_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    object_nodes_by_id: dict[str, dict[str, Any]] = {}
    ancestors: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    for node_id, node in nodes.items():
        if isinstance(node.get("parent_id"), str):
            parents[node_id] = str(node["parent_id"])
        for child_id in node.get("children_ids", []):
            parents.setdefault(str(child_id), node_id)
    for node_id, node in nodes.items():
        if str(node.get("node_type", "")) != "object":
            continue
        view_ids = [str(item) for item in node.get("view_ids", [])]
        if view_ids:
            object_nodes[(view_ids[0], _normal_label(str(node.get("name", ""))))] = node
        bbox = _as_mapping(node.get("bbox_3d"))
        if bbox.get("object_id") is not None:
            object_nodes_by_id[str(bbox["object_id"])] = node
        chain: list[str] = []
        parent_id = parents.get(node_id)
        seen: set[str] = set()
        while isinstance(parent_id, str) and parent_id in nodes and parent_id not in seen:
            seen.add(parent_id)
            chain.append(parent_id)
            parent_id = parents.get(parent_id)
        ancestors[node_id] = chain
    return object_nodes, object_nodes_by_id, ancestors


def _object_id(obj: Mapping[str, Any]) -> str | None:
    for key in ("object_id", "instance_id", "id"):
        if obj.get(key) is not None:
            return str(obj[key])
    bbox = _as_mapping(obj.get("bbox_3d"))
    for key in ("object_id", "instance_id", "id"):
        if bbox.get(key) is not None:
            return str(bbox[key])
    attributes = obj.get("attributes", [])
    values = attributes if isinstance(attributes, list) else []
    for value in values:
        match = re.fullmatch(r"(?:object|instance)_id\s*:\s*(.+)", str(value), re.I)
        if match:
            return match.group(1).strip()
    return None


def build_candidates(
    views: Mapping[str, Any],
    tree: Mapping[str, Any] | None = None,
) -> list[GroundingCandidate]:
    """Build the shared, deduplicated object-level candidate universe."""
    tree = tree or {}
    object_nodes, object_nodes_by_id, ancestors = _tree_maps(tree)
    candidates: list[GroundingCandidate] = []
    used_node_ids: set[str] = set()

    for view_key in sorted(views):
        view = _as_mapping(views[view_key])
        view_id = str(view.get("view_id") or view_key)
        view_relations = [str(item) for item in view.get("spatial_relations", [])]
        for index, obj in enumerate(_object_entries(view), start=1):
            label = str(obj.get("label", "")).strip()
            source_object_id = _object_id(obj)
            node = (
                object_nodes_by_id.get(source_object_id or "")
                or object_nodes.get((view_id, _normal_label(label)), {})
            )
            node_id = str(node["node_id"]) if node.get("node_id") else None
            if node_id:
                used_node_ids.add(node_id)
            attributes = obj.get("attributes", [])
            if isinstance(attributes, Mapping):
                attribute_text = [str(value) for value in attributes.values() if value]
            elif isinstance(attributes, list):
                attribute_text = [str(value) for value in attributes]
            else:
                attribute_text = [str(attributes)] if attributes else []
            approx_location = str(obj.get("approx_location") or obj.get("location_description") or "")
            relations = list(dict.fromkeys(view_relations + attribute_text + ([approx_location] if approx_location else [])))
            parts = [label, *attribute_text, approx_location, *view_relations]
            node_bbox = _as_mapping(node.get("bbox_3d"))
            object_id = (
                source_object_id
                or (str(node_bbox["object_id"]) if node_bbox.get("object_id") is not None else None)
                or node_id
                or f"object_{view_id}_{index:03d}_{_normal_label(label).replace(' ', '_')}"
            )
            candidates.append(
                GroundingCandidate(
                    object_id=object_id,
                    node_id=node_id,
                    view_id=view_id,
                    ancestor_node_ids=list(ancestors.get(node_id or "", [])),
                    label=label,
                    text=" ".join(part for part in parts if part),
                    bbox_2d=_tuple4(obj.get("bbox_2d")),
                    bbox_3d=_as_mapping(obj.get("bbox_3d")) or None,
                    relations=relations,
                )
            )

    # Tree-only object nodes remain valid candidates for older indexes.
    for node in object_nodes.values():
        node_id = str(node.get("node_id", ""))
        if not node_id or node_id in used_node_ids:
            continue
        view_ids = [str(item) for item in node.get("view_ids", [])]
        attributes = [str(item) for item in node.get("attributes", [])]
        label = str(node.get("name", ""))
        relations = attributes + ([str(node["approx_location"])] if node.get("approx_location") else [])
        node_bbox = _as_mapping(node.get("bbox_3d"))
        candidates.append(
            GroundingCandidate(
                object_id=str(node_bbox.get("object_id") or node_id),
                node_id=node_id,
                view_id=view_ids[0] if view_ids else "",
                ancestor_node_ids=list(ancestors.get(node_id, [])),
                label=label,
                text=" ".join([label, str(node.get("summary", "")), *relations]),
                bbox_2d=_tuple4(node.get("bbox_2d")),
                bbox_3d=_as_mapping(node.get("bbox_3d")) or None,
                relations=relations,
            )
        )
    return candidates


def validate_bbox_3d(
    bbox: Mapping[str, Any] | None,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None = None,
) -> tuple[bool, str]:
    """Validate finite center/size, positive extents, and optional scene bounds."""
    if not bbox:
        return False, "missing_bbox_3d"
    try:
        center = np.asarray(bbox.get("center"), dtype=float)
        size = np.asarray(bbox.get("size"), dtype=float)
    except (TypeError, ValueError):
        return False, "non_numeric_bbox_3d"
    if center.shape != (3,) or size.shape != (3,):
        return False, "bbox_3d_requires_three_axes"
    if not np.isfinite(center).all() or not np.isfinite(size).all():
        return False, "non_finite_bbox_3d"
    if not (size > 0).all():
        return False, "non_positive_bbox_3d_extent"
    if scene_bounds is not None:
        if isinstance(scene_bounds, Mapping):
            low = np.asarray(scene_bounds.get("min"), dtype=float)
            high = np.asarray(scene_bounds.get("max"), dtype=float)
        else:
            bounds = np.asarray(scene_bounds, dtype=float)
            if bounds.shape != (2, 3):
                return False, "scene_bounds_requires_min_max"
            low, high = bounds
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
            return False, "invalid_scene_bounds"
        bbox_low, bbox_high = center - size / 2.0, center + size / 2.0
        if (bbox_low < low).any() or (bbox_high > high).any():
            return False, "bbox_3d_outside_scene_bounds"
    return True, "valid"


def validate_bbox_2d(bbox: Sequence[float] | None) -> tuple[bool, str]:
    if bbox is None or len(bbox) != 4:
        return False, "missing_bbox_2d"
    values = np.asarray(bbox, dtype=float)
    if not np.isfinite(values).all():
        return False, "non_finite_bbox_2d"
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return False, "non_positive_bbox_2d_extent"
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        return False, "bbox_2d_outside_normalized_bounds"
    return True, "valid"


def parse_relation_query(query: str) -> RelationQuery:
    """Extract a target, optional relation, and anchor from common phrasings."""
    cleaned = " ".join(query.strip().split())
    match = _BETWEEN_RELATION.search(cleaned)
    if match:
        return RelationQuery(
            target=_clean_query_phrase(match.group(1)),
            relation="between",
            anchor=_clean_query_phrase(match.group(2)),
            anchor_secondary=_clean_query_phrase(match.group(3)),
        )
    for relation, pattern in _DISTANCE_RELATIONS:
        match = pattern.search(cleaned)
        if match:
            return RelationQuery(
                target=_clean_query_phrase(match.group(1)),
                relation=relation,
                anchor=_clean_query_phrase(match.group(2)),
            )
    for relation, pattern in _RELATION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            target = _clean_query_phrase(match.group(1))
            anchor = _clean_query_phrase(match.group(2))
            return RelationQuery(target=target, relation=relation, anchor=anchor)
    match = _PREFIX_RELATION.search(cleaned)
    if match:
        relation = match.group("relation").lower().replace("left of", "left").replace("right of", "right")
        relation = relation.replace("in front of", "front")
        before = cleaned[: match.start()]
        target = _clean_query_phrase(before)
        return RelationQuery(target=target, relation=relation, anchor=_clean_query_phrase(match.group("anchor")))
    return RelationQuery(target=_clean_query_phrase(cleaned))


def _clean_query_phrase(text: str) -> str:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _QUERY_FILLER]
    return " ".join(tokens)


def _expanded_tokens(text: str) -> tuple[set[str], set[str], list[str]]:
    base = tokenize(text)
    expanded, affordances = expand_query_tokens(text)
    synonyms: set[str] = set()
    for token in base:
        synonyms.update(_SYNONYMS.get(token, ()))
    return base, expanded | synonyms, affordances


def lexical_score(candidate: GroundingCandidate, target: str) -> tuple[float, dict[str, Any]]:
    """Score exact labels first, then lexical/synonym/affordance overlap."""
    base, expanded, affordances = _expanded_tokens(target)
    label_tokens = tokenize(candidate.label)
    text_tokens = tokenize(candidate.text)
    normalized_target = _normal_label(target)
    normalized_label = _normal_label(candidate.label)
    exact = bool(normalized_target and normalized_target == normalized_label)
    label_overlap = len(base & label_tokens) / max(1, len(base))
    text_overlap = len(base & text_tokens) / max(1, len(base))
    semantic_only = expanded - base
    semantic_overlap = len(semantic_only & text_tokens) / max(1, len(semantic_only))
    score = min(1.0, (0.65 if exact else 0.0) + 0.55 * label_overlap + 0.20 * text_overlap + 0.25 * semantic_overlap)
    return score, {
        "exact_label": exact,
        "label_overlap": round(label_overlap, 6),
        "text_overlap": round(text_overlap, 6),
        "synonym_affordance_overlap": round(semantic_overlap, 6),
        "affordance_keys": affordances,
        "lexical_score": round(score, 6),
    }


def _geometry(
    candidate: GroundingCandidate,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    valid_3d, _ = validate_bbox_3d(candidate.bbox_3d, scene_bounds)
    if valid_3d and candidate.bbox_3d:
        return (
            np.asarray(candidate.bbox_3d["center"], dtype=float),
            np.asarray(candidate.bbox_3d["size"], dtype=float),
            "bbox_3d",
        )
    valid_2d, _ = validate_bbox_2d(candidate.bbox_2d)
    if valid_2d and candidate.bbox_2d:
        x1, y1, x2, y2 = candidate.bbox_2d
        return (
            np.asarray(((x1 + x2) / 2, (y1 + y2) / 2), dtype=float),
            np.asarray((x2 - x1, y2 - y1), dtype=float),
            "bbox_2d",
        )
    return None


def relation_predicate(
    candidate: GroundingCandidate,
    anchor: GroundingCandidate,
    relation: str,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Evaluate a spatial predicate; ``None`` means geometry is unresolved."""
    if relation not in RELATIONS:
        raise ValueError(f"Unsupported relation: {relation}")
    candidate_geometry = _geometry(candidate, scene_bounds)
    anchor_geometry = _geometry(anchor, scene_bounds)
    if candidate_geometry is None or anchor_geometry is None:
        return None, {"source": "none", "reason": "bbox_quality_rejected"}
    center, size, source = candidate_geometry
    anchor_center, anchor_size, anchor_source = anchor_geometry
    if center.shape != anchor_center.shape:
        return None, {"source": "none", "reason": "bbox_dimension_mismatch"}
    if center.shape[0] == 3 and relation in {"left", "right", "front", "behind"}:
        coordinate_frames = {
            str((candidate.bbox_3d or {}).get("coordinate_frame") or ""),
            str((anchor.bbox_3d or {}).get("coordinate_frame") or ""),
        }
        if any(
            frame.startswith(("scannet_", "replica_"))
            for frame in coordinate_frames
        ):
            return None, {
                "source": source,
                "reason": "viewpoint_axis_unavailable",
            }

    delta = center - anchor_center
    tolerance = np.maximum((size + anchor_size) * 0.05, 1e-9)
    if relation == "near":
        scale = max(float(np.linalg.norm((size + anchor_size) / 2.0)), 1e-9)
        normalized_distance = float(np.linalg.norm(delta) / scale)
        matched = normalized_distance <= 1.5
        margin = 1.5 - normalized_distance
        confidence = max(0.0, min(1.0, margin / 1.5)) if matched else 0.0
    else:
        axis = {"left": 0, "right": 0, "above": 1, "below": 1, "front": -1, "behind": -1}[relation]
        if center.shape[0] == 2 and relation in {"front", "behind"}:
            return None, {"source": source, "reason": "depth_axis_unavailable"}
        signed = float(delta[axis])
        # Image y grows downward; world y grows upward.
        direction = {
            "left": -1,
            "right": 1,
            "above": -1 if center.shape[0] == 2 else 1,
            "below": 1 if center.shape[0] == 2 else -1,
            "front": -1,
            "behind": 1,
        }[relation]
        margin = direction * signed - float(tolerance[axis])
        matched = margin > 0
        scale = max(float((size[axis] + anchor_size[axis]) / 2.0), 1e-9)
        confidence = max(0.0, min(1.0, margin / scale)) if matched else 0.0
    return matched, {
        "source": source if source == anchor_source else f"{source}+{anchor_source}",
        "margin": round(float(margin), 6),
        "confidence": round(float(confidence), 6),
    }


def _distance_evidence(
    candidate: GroundingCandidate,
    anchor: GroundingCandidate,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None,
) -> dict[str, Any] | None:
    candidate_geometry = _geometry(candidate, scene_bounds)
    anchor_geometry = _geometry(anchor, scene_bounds)
    if candidate_geometry is None or anchor_geometry is None:
        return None
    center, _, source = candidate_geometry
    anchor_center, anchor_size, anchor_source = anchor_geometry
    if center.shape != anchor_center.shape:
        return None
    scale = max(float(np.linalg.norm(anchor_size)), 1e-9)
    return {
        "source": source if source == anchor_source else f"{source}+{anchor_source}",
        "distance": round(float(np.linalg.norm(center - anchor_center)), 6),
        "normalized_distance": round(
            float(np.linalg.norm(center - anchor_center) / scale),
            6,
        ),
    }


def _between_evidence(
    candidate: GroundingCandidate,
    anchor_a: GroundingCandidate,
    anchor_b: GroundingCandidate,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None,
) -> dict[str, Any] | None:
    candidate_geometry = _geometry(candidate, scene_bounds)
    first_geometry = _geometry(anchor_a, scene_bounds)
    second_geometry = _geometry(anchor_b, scene_bounds)
    if candidate_geometry is None or first_geometry is None or second_geometry is None:
        return None
    center, candidate_size, source = candidate_geometry
    first_center, first_size, first_source = first_geometry
    second_center, second_size, second_source = second_geometry
    if not (center.shape == first_center.shape == second_center.shape):
        return None
    segment = second_center - first_center
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return None
    projection = float(np.dot(center - first_center, segment) / length_squared)
    closest = first_center + projection * segment
    perpendicular = float(np.linalg.norm(center - closest))
    size_scale = max(
        float(
            np.linalg.norm(
                (candidate_size + first_size + second_size) / 3.0
            )
        ),
        1e-9,
    )
    threshold = max(size_scale, 0.25 * length_squared**0.5)
    matched = 0.0 <= projection <= 1.0 and perpendicular <= threshold
    midpoint_score = max(0.0, 1.0 - 2.0 * abs(projection - 0.5))
    perpendicular_score = max(0.0, 1.0 - perpendicular / threshold)
    confidence = (
        0.5 * midpoint_score + 0.5 * perpendicular_score
        if matched
        else 0.0
    )
    return {
        "source": "+".join(
            dict.fromkeys((source, first_source, second_source))
        ),
        "matched": matched,
        "projection": round(projection, 6),
        "perpendicular_distance": round(perpendicular, 6),
        "confidence": round(confidence, 6),
    }


def _relation_text_match(candidate: GroundingCandidate, relation: str, anchor: str) -> bool:
    candidate_terms = tokenize(candidate.label)
    anchor_terms = tokenize(anchor)
    for text in candidate.relations:
        terms = tokenize(text)
        parsed = parse_relation_query(text)
        if (
            parsed.relation == relation
            and parsed.anchor
            and anchor_terms & tokenize(parsed.anchor)
            and parsed.target
            and candidate_terms & tokenize(parsed.target)
        ):
            return True
        # Object attributes frequently omit the subject: "near table".
        starts_with_relation = bool(
            re.match(r"^\s*(?:in\s+front\s+of|left\s+of|right\s+of|near|behind|above|below)\b", text, re.I)
        )
        if starts_with_relation and anchor_terms & terms:
            normalized_relation = "front" if "front" in terms else relation
            if normalized_relation in terms or relation == "front":
                return True
    return False


def _matching_anchors(
    candidates: Iterable[GroundingCandidate],
    anchor_text: str,
) -> list[tuple[float, GroundingCandidate]]:
    object_id_match = re.search(
        r"\b(?:replica\s+|scannet\s+)?(?:object|instance)\s*(?:id\s*)?([a-z0-9_-]+)\b",
        anchor_text,
        re.I,
    )
    explicit_object_id = object_id_match.group(1) if object_id_match else None
    scored = [
        (
            1.0
            if explicit_object_id is not None
            and candidate.object_id.casefold() == explicit_object_id.casefold()
            else lexical_score(candidate, anchor_text)[0],
            candidate,
        )
        for candidate in candidates
    ]
    return [
        (score, candidate)
        for score, candidate in sorted(
            scored,
            key=lambda item: (-item[0], item[1].object_id),
        )
        if score > 0
    ]


def _select_anchors(
    candidates: Iterable[GroundingCandidate],
    anchor_text: str,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[list[GroundingCandidate], dict[str, Any]]:
    scored = _matching_anchors(candidates, anchor_text)
    top_score = scored[0][0] if scored else 0.0
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    score_margin = top_score - second_score
    if not scored or top_score < confidence_threshold:
        return [], {
            "status": "rejected",
            "reason": "low_anchor_confidence",
            "top_score": round(top_score, 6),
            "second_score": round(second_score, 6),
            "score_margin": round(score_margin, 6),
            "candidate_count": len(scored),
            "confidence": 0.0,
        }

    selected = [
        candidate
        for score, candidate in scored
        if score >= confidence_threshold and top_score - score <= margin_threshold
    ]
    if len(selected) == 1:
        confidence = top_score
        status = "unique"
    else:
        # Repeated anchor labels are retained for geometric testing, but their
        # confidence is deliberately capped so they cannot dominate lexical
        # evidence without a high-confidence explicit relation string.
        confidence = min(top_score, 0.5 + max(0.0, score_margin))
        status = "ambiguous"
    return selected, {
        "status": status,
        "reason": None,
        "top_score": round(top_score, 6),
        "second_score": round(second_score, 6),
        "score_margin": round(score_margin, 6),
        "candidate_count": len(scored),
        "selected_count": len(selected),
        "selected_object_ids": [candidate.object_id for candidate in selected],
        "confidence": round(float(confidence), 6),
    }


def rerank_relations(
    candidates: list[GroundingCandidate],
    all_candidates: list[GroundingCandidate],
    parsed: RelationQuery,
    *,
    use_relation: bool = True,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None = None,
    anchor_confidence_threshold: float = DEFAULT_ANCHOR_CONFIDENCE_THRESHOLD,
    anchor_margin_threshold: float = DEFAULT_ANCHOR_MARGIN_THRESHOLD,
    relation_confidence_threshold: float = DEFAULT_RELATION_CONFIDENCE_THRESHOLD,
    relation_score_margin: float = DEFAULT_RELATION_SCORE_MARGIN,
    relation_boost: float = DEFAULT_RELATION_BOOST,
) -> tuple[list[GroundingCandidate], bool, bool]:
    """Apply relations only when anchor, geometry, and score-margin gates pass."""
    if not use_relation or not parsed.relation or not parsed.anchor:
        return candidates, False, False
    anchors, anchor_evidence = _select_anchors(
        all_candidates,
        parsed.anchor,
        confidence_threshold=anchor_confidence_threshold,
        margin_threshold=anchor_margin_threshold,
    )
    anchor_confidence = float(anchor_evidence.get("confidence") or 0.0)
    secondary_anchors: list[GroundingCandidate] = []
    secondary_anchor_evidence: dict[str, Any] | None = None
    secondary_anchor_confidence = 0.0
    if parsed.relation == "between" and parsed.anchor_secondary:
        secondary_anchors, secondary_anchor_evidence = _select_anchors(
            all_candidates,
            parsed.anchor_secondary,
            confidence_threshold=anchor_confidence_threshold,
            margin_threshold=anchor_margin_threshold,
        )
        secondary_anchor_confidence = float(
            secondary_anchor_evidence.get("confidence") or 0.0
        )
    relation_resolved = False
    bbox_rejected = False
    best_lexical_score = max((candidate.score for candidate in candidates), default=0.0)
    lexical_order = [candidate.object_id for candidate in candidates]
    relative_distances: dict[str, float] = {}
    for candidate in candidates:
        base_score = candidate.score
        best_relation = 0.0
        relation_confidence = 0.0
        source = "unresolved"
        evidence: list[dict[str, Any]] = []
        if (
            parsed.relation not in {"closest", "farthest", "between"}
            and _relation_text_match(candidate, parsed.relation, parsed.anchor)
        ):
            best_relation, relation_confidence, source = 1.0, 1.0, "spatial_relation_string"
            evidence.append(
                {
                    "source": source,
                    "matched": True,
                    "confidence": 1.0,
                }
            )
        if parsed.relation in {"closest", "farthest"}:
            distances: list[float] = []
            for anchor in anchors:
                if anchor.object_id == candidate.object_id:
                    continue
                detail = _distance_evidence(candidate, anchor, scene_bounds)
                if detail is None:
                    bbox_rejected = True
                    evidence.append(
                        {
                            "source": "none",
                            "anchor_object_id": anchor.object_id,
                            "matched": None,
                            "reason": "bbox_quality_rejected",
                            "confidence": 0.0,
                        }
                    )
                    continue
                distance = float(detail["normalized_distance"])
                distances.append(distance)
                evidence.append(
                    {
                        "source": detail["source"],
                        "anchor_object_id": anchor.object_id,
                        "matched": True,
                        "normalized_distance": distance,
                    }
                )
            if distances:
                relative_distances[candidate.object_id] = min(distances)
                source = str(evidence[-1]["source"])
        elif parsed.relation == "between":
            for anchor in anchors:
                for secondary_anchor in secondary_anchors:
                    if candidate.object_id in {
                        anchor.object_id,
                        secondary_anchor.object_id,
                    }:
                        continue
                    detail = _between_evidence(
                        candidate,
                        anchor,
                        secondary_anchor,
                        scene_bounds,
                    )
                    if detail is None:
                        bbox_rejected = True
                        continue
                    confidence = (
                        anchor_confidence
                        * secondary_anchor_confidence
                        * float(detail["confidence"])
                    )
                    evidence.append(
                        {
                            **detail,
                            "anchor_object_id": anchor.object_id,
                            "secondary_anchor_object_id": secondary_anchor.object_id,
                            "confidence": round(confidence, 6),
                        }
                    )
                    if detail["matched"] and confidence > relation_confidence:
                        best_relation = 1.0
                        relation_confidence = confidence
                        source = str(detail["source"])
        else:
            for anchor in anchors:
                if anchor.object_id == candidate.object_id:
                    continue
                matched, detail = relation_predicate(candidate, anchor, parsed.relation, scene_bounds)
                if matched is None:
                    bbox_rejected = bbox_rejected or detail.get("reason") == "bbox_quality_rejected"
                    evidence.append(
                        {
                            "source": detail.get("source", "none"),
                            "anchor_object_id": anchor.object_id,
                            "matched": None,
                            "reason": detail.get("reason"),
                            "confidence": 0.0,
                        }
                    )
                    continue
                geometry_confidence = float(detail.get("confidence") or 0.0)
                confidence = anchor_confidence * geometry_confidence if matched else 0.0
                evidence.append(
                    {
                        "source": detail.get("source", "none"),
                        "anchor_object_id": anchor.object_id,
                        "matched": matched,
                        "margin": detail.get("margin"),
                        "confidence": round(confidence, 6),
                    }
                )
                if matched and confidence > relation_confidence:
                    best_relation = 1.0
                    relation_confidence = confidence
                    source = str(detail.get("source"))
        if best_relation == 0.0:
            source = source if candidate.object_id in relative_distances else "unresolved"
        score_margin_eligible = (
            base_score >= best_lexical_score - relation_score_margin
            or bool(candidate.score_details.get("exact_label"))
        )
        relation_applied = bool(
            parsed.relation not in {"closest", "farthest"}
            and
            best_relation > 0
            and relation_confidence >= relation_confidence_threshold
            and score_margin_eligible
        )
        if relation_applied:
            relation_resolved = True
            candidate.score = max(
                0.0,
                min(1.0, candidate.score + relation_boost * best_relation),
            )
        candidate.score_details.update(
            {
                "relation": parsed.relation,
                "anchor": parsed.anchor,
                "anchor_secondary": parsed.anchor_secondary,
                "relation_score": best_relation,
                "relation_source": source,
                "relation_confidence": round(relation_confidence, 6),
                "relation_applied": relation_applied,
                "relation_score_margin_eligible": score_margin_eligible,
                "relation_evidence": evidence,
                "anchor_evidence": anchor_evidence,
                "secondary_anchor_evidence": secondary_anchor_evidence,
                "pre_relation_score": round(base_score, 6),
            }
        )
    if parsed.relation in {"closest", "farthest"} and relative_distances:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.object_id in relative_distances
            and candidate.score_details.get("relation_score_margin_eligible")
        ]
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                relative_distances[candidate.object_id],
                candidate.object_id,
            ),
            reverse=parsed.relation == "farthest",
        )
        if ordered:
            best = relative_distances[ordered[0].object_id]
            ties = [
                candidate
                for candidate in ordered
                if abs(relative_distances[candidate.object_id] - best) <= 1e-9
            ]
            second = (
                relative_distances[ordered[len(ties)].object_id]
                if len(ordered) > len(ties)
                else best
            )
            separation = (
                abs(second - best) / max(abs(second), abs(best), 1e-9)
                if len(ordered) > len(ties)
                else 0.0
            )
            confidence = (
                anchor_confidence
                * (0.5 + 0.5 * min(1.0, separation))
                / len(ties)
            )
            for candidate in ties:
                candidate.score_details["relation_score"] = 1.0
                candidate.score_details["relation_confidence"] = round(
                    confidence,
                    6,
                )
                candidate.score_details["relation_source"] = "relative_bbox_distance"
                candidate.score_details["relation_distance"] = best
                applied = confidence >= relation_confidence_threshold
                candidate.score_details["relation_applied"] = applied
                if applied:
                    relation_resolved = True
                    candidate.score = min(1.0, candidate.score + relation_boost)
    unresolved = (
        not anchors
        or (parsed.relation == "between" and not secondary_anchors)
        or not relation_resolved
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.score,
            -int(bool(item.score_details.get("relation_applied"))),
            -(
                float(item.score_details.get("relation_confidence") or 0.0)
                if item.score_details.get("relation_applied")
                else 0.0
            ),
            item.object_id,
        ),
    )
    relation_order = [candidate.object_id for candidate in ranked]
    changed = lexical_order != relation_order
    for candidate in ranked:
        candidate.score_details["relation_changed_ranking"] = changed
        candidate.score_details["pre_relation_rank"] = (
            lexical_order.index(candidate.object_id) + 1
        )
        candidate.score_details["post_relation_rank"] = (
            relation_order.index(candidate.object_id) + 1
        )
    return ranked, unresolved, bbox_rejected


def _relation_audit(candidates: Sequence[GroundingCandidate]) -> dict[str, Any]:
    affected = [
        candidate
        for candidate in candidates
        if candidate.score_details.get("relation") is not None
    ]
    return {
        "evaluated_candidate_count": len(affected),
        "applied_candidate_count": sum(
            bool(candidate.score_details.get("relation_applied"))
            for candidate in affected
        ),
        "ranking_changed": any(
            bool(candidate.score_details.get("relation_changed_ranking"))
            for candidate in affected
        ),
        "candidate_evidence": [
            {
                "object_id": candidate.object_id,
                "label": candidate.label,
                "relation_score": candidate.score_details.get("relation_score"),
                "relation_confidence": candidate.score_details.get("relation_confidence"),
                "relation_applied": candidate.score_details.get("relation_applied"),
                "relation_source": candidate.score_details.get("relation_source"),
                "pre_relation_rank": candidate.score_details.get("pre_relation_rank"),
                "post_relation_rank": candidate.score_details.get("post_relation_rank"),
            }
            for candidate in affected
        ],
    }


def _node_text(node: Mapping[str, Any]) -> str:
    values = [node.get("name"), node.get("summary"), *(node.get("attributes") or [])]
    return " ".join(str(value) for value in values if value)


def graph_pruned_candidate_ids(
    candidates: list[GroundingCandidate],
    tree: Mapping[str, Any],
    target: str,
    *,
    branch_keep_ratio: float = 0.5,
) -> set[str]:
    """Select candidates under the best-scoring top-level hierarchy branches."""
    if not tree:
        return set()
    nodes = {str(key): _as_mapping(value) for key, value in tree.items()}
    roots = [node for node in nodes.values() if node.get("parent_id") is None or str(node.get("node_type")) == "root"]
    root = roots[0] if roots else {}
    child_ids = [str(item) for item in root.get("children_ids", []) if str(item) in nodes]
    if not child_ids:
        return set()
    query_tokens = _expanded_tokens(target)[1]
    scored: list[tuple[float, str]] = []
    for child_id in child_ids:
        terms = tokenize(_node_text(nodes[child_id]))
        score = len(query_tokens & terms) / max(1, len(query_tokens))
        scored.append((score, child_id))
    best = max((score for score, _ in scored), default=0.0)
    if best <= 0:
        # No hierarchy evidence: deliberately narrow; thresholded fallback can expand.
        selected = {sorted(child_ids)[0]}
    else:
        selected = {child_id for score, child_id in scored if score >= best * branch_keep_ratio}
    return {
        candidate.object_id
        for candidate in candidates
        if candidate.node_id in selected or selected.intersection(candidate.ancestor_node_ids)
    }


def _rank_lexical(candidates: list[GroundingCandidate], target: str) -> list[GroundingCandidate]:
    ranked: list[GroundingCandidate] = []
    for candidate in candidates:
        score, details = lexical_score(candidate, target)
        candidate.score = score
        candidate.score_details = details
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: (-item.score, item.object_id))


def _encode(embedder: Any, texts: Sequence[str]) -> np.ndarray:
    if hasattr(embedder, "encode"):
        try:
            values = embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        except TypeError:
            values = embedder.encode(list(texts))
    elif callable(embedder):
        values = embedder(list(texts))
    else:
        raise TypeError("embedder must be callable or expose encode(texts)")
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] == 0:
        raise ValueError("embedder returned an invalid matrix shape")
    if not np.isfinite(array).all():
        raise ValueError("embedder returned non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embedder returned a zero vector")
    return array / norms


def _load_sentence_transformer(model_name: str, cache_folder: str | None) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu", cache_folder=cache_folder)


def _rank_embedding(
    candidates: list[GroundingCandidate],
    target: str,
    embedder: Embedder | Callable[[Sequence[str]], Any] | None,
    *,
    model_name: str,
    cache_folder: str | None,
) -> tuple[list[GroundingCandidate] | None, dict[str, Any]]:
    metadata = {
        "backend": "injected" if embedder is not None else "sentence_transformers",
        "model": model_name if embedder is None else getattr(embedder, "model_name", "injected"),
        "cache_folder": cache_folder,
        "device": "cpu",
    }
    if embedder is None:
        try:
            embedder = _load_sentence_transformer(model_name, cache_folder)
        except (ImportError, OSError, RuntimeError) as exc:
            metadata["blocked_reason"] = f"{type(exc).__name__}: {exc}"
            return None, metadata
    try:
        matrix = _encode(embedder, [target, *(candidate.text for candidate in candidates)])
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        metadata["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        return None, metadata
    similarities = matrix[1:] @ matrix[0]
    for candidate, similarity in zip(candidates, similarities):
        candidate.score = float((similarity + 1.0) / 2.0)
        candidate.score_details = {"cosine_similarity": round(float(similarity), 6)}
    return sorted(candidates, key=lambda item: (-item.score, item.object_id)), metadata


def _fresh(candidates: Iterable[GroundingCandidate]) -> list[GroundingCandidate]:
    return [
        GroundingCandidate(
            object_id=item.object_id,
            node_id=item.node_id,
            view_id=item.view_id,
            ancestor_node_ids=list(item.ancestor_node_ids),
            label=item.label,
            text=item.text,
            bbox_2d=item.bbox_2d,
            bbox_3d=dict(item.bbox_3d) if item.bbox_3d else None,
            relations=list(item.relations),
        )
        for item in candidates
    ]


def search_grounding(
    bundle: Mapping[str, Any],
    query: str,
    *,
    variant: str = "graph_fallback",
    no_hierarchy: bool = False,
    no_relation: bool = False,
    no_fallback: bool = False,
    top_k: int = 10,
    confidence_threshold: float = 0.45,
    branch_keep_ratio: float = 0.5,
    anchor_confidence_threshold: float = DEFAULT_ANCHOR_CONFIDENCE_THRESHOLD,
    anchor_margin_threshold: float = DEFAULT_ANCHOR_MARGIN_THRESHOLD,
    relation_confidence_threshold: float = DEFAULT_RELATION_CONFIDENCE_THRESHOLD,
    relation_score_margin: float = DEFAULT_RELATION_SCORE_MARGIN,
    relation_boost: float = DEFAULT_RELATION_BOOST,
    scene_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None = None,
    embedder: Embedder | Callable[[Sequence[str]], Any] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_cache: str | None = None,
) -> GroundingSearchResult:
    """Run a grounding variant over a bundle containing ``views`` and ``tree``.

    Ablations are orthogonal: ``no_hierarchy`` disables graph pruning,
    ``no_relation`` disables relation reranking, and ``no_fallback`` prevents
    thresholded expansion.  ``flat_embedding`` returns ``status='blocked'`` if
    its optional backend cannot be loaded.
    """
    if variant not in SEARCH_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(SEARCH_VARIANTS)}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if branch_keep_ratio < 0:
        raise ValueError("branch_keep_ratio must be non-negative")
    for name, value in (
        ("anchor_confidence_threshold", anchor_confidence_threshold),
        ("anchor_margin_threshold", anchor_margin_threshold),
        ("relation_confidence_threshold", relation_confidence_threshold),
        ("relation_score_margin", relation_score_margin),
        ("relation_boost", relation_boost),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    views = bundle.get("views", {})
    tree = bundle.get("tree", {})
    canonical = build_candidates(views if isinstance(views, Mapping) else {}, tree if isinstance(tree, Mapping) else {})
    parsed = parse_relation_query(query)
    metadata: dict[str, Any] = {
        "ablation_flags": {
            "no_hierarchy": no_hierarchy,
            "no_relation": no_relation,
            "no_fallback": no_fallback,
        },
        "same_input_candidate_ids": [candidate.object_id for candidate in canonical],
        "relation_policy": {
            "anchor_confidence_threshold": anchor_confidence_threshold,
            "anchor_margin_threshold": anchor_margin_threshold,
            "relation_confidence_threshold": relation_confidence_threshold,
            "relation_score_margin": relation_score_margin,
            "relation_boost": relation_boost,
        },
    }

    if variant == "flat_embedding":
        ranked, embedding_metadata = _rank_embedding(
            _fresh(canonical),
            parsed.target or query,
            embedder,
            model_name=embedding_model,
            cache_folder=embedding_cache,
        )
        metadata["embedding"] = embedding_metadata
        if ranked is None:
            return GroundingSearchResult(
                status="blocked",
                variant=variant,
                query=query,
                parsed_query=parsed,
                candidates=[],
                candidate_count=len(canonical),
                searched_candidate_count=0,
                metadata=metadata,
            )
        ranked, unresolved, bbox_rejected = rerank_relations(
            ranked,
            _fresh(canonical),
            parsed,
            use_relation=not no_relation,
            scene_bounds=scene_bounds,
            anchor_confidence_threshold=anchor_confidence_threshold,
            anchor_margin_threshold=anchor_margin_threshold,
            relation_confidence_threshold=relation_confidence_threshold,
            relation_score_margin=relation_score_margin,
            relation_boost=relation_boost,
        )
        metadata.update(
            {
                "relation_unresolved": unresolved,
                "bbox_quality_rejected": bbox_rejected,
                "relation_audit": _relation_audit(ranked),
            }
        )
        status = "ok" if ranked and ranked[0].score > 0 else "no_match"
        return GroundingSearchResult(
            status=status,
            variant=variant,
            query=query,
            parsed_query=parsed,
            candidates=ranked[:top_k],
            candidate_count=len(canonical),
            searched_candidate_count=len(canonical),
            metadata=metadata,
        )

    use_graph = variant in {"graph", "graph_fallback"} and not no_hierarchy
    if use_graph:
        selected_ids = graph_pruned_candidate_ids(canonical, tree, parsed.target or query, branch_keep_ratio=branch_keep_ratio)
        initial = [candidate for candidate in canonical if candidate.object_id in selected_ids]
    else:
        initial = canonical
    ranked = _rank_lexical(_fresh(initial), parsed.target or query)
    ranked, relation_unresolved, bbox_rejected = rerank_relations(
        ranked,
        _fresh(canonical),
        parsed,
        use_relation=not no_relation,
        scene_bounds=scene_bounds,
        anchor_confidence_threshold=anchor_confidence_threshold,
        anchor_margin_threshold=anchor_margin_threshold,
        relation_confidence_threshold=relation_confidence_threshold,
        relation_score_margin=relation_score_margin,
        relation_boost=relation_boost,
    )
    fallback_reasons: list[str] = []
    if not ranked or ranked[0].score < confidence_threshold:
        fallback_reasons.append("low_confidence")
    if relation_unresolved:
        fallback_reasons.append("relation_unresolved")
    if bbox_rejected:
        fallback_reasons.append("bbox_quality_rejected")

    fallback_allowed = variant == "graph_fallback" and not no_fallback and use_graph
    fallback_triggered = bool(fallback_allowed and fallback_reasons)
    if fallback_triggered:
        ranked = _rank_lexical(_fresh(canonical), parsed.target or query)
        ranked, relation_unresolved, bbox_rejected = rerank_relations(
            ranked,
            _fresh(canonical),
            parsed,
            use_relation=not no_relation,
            scene_bounds=scene_bounds,
            anchor_confidence_threshold=anchor_confidence_threshold,
            anchor_margin_threshold=anchor_margin_threshold,
            relation_confidence_threshold=relation_confidence_threshold,
            relation_score_margin=relation_score_margin,
            relation_boost=relation_boost,
        )
        if relation_unresolved and "relation_unresolved" not in fallback_reasons:
            fallback_reasons.append("relation_unresolved")
        if bbox_rejected and "bbox_quality_rejected" not in fallback_reasons:
            fallback_reasons.append("bbox_quality_rejected")
    for candidate in ranked:
        valid_3d, reason_3d = validate_bbox_3d(candidate.bbox_3d, scene_bounds)
        valid_2d, reason_2d = validate_bbox_2d(candidate.bbox_2d)
        candidate.score_details["bbox_quality"] = {
            "bbox_3d_valid": valid_3d,
            "bbox_3d_reason": reason_3d,
            "bbox_2d_valid": valid_2d,
            "bbox_2d_reason": reason_2d,
        }
    metadata.update(
        {
            "relation_unresolved": relation_unresolved,
            "bbox_quality_rejected": bbox_rejected,
            "initial_candidate_ids": [candidate.object_id for candidate in initial],
            "relation_audit": _relation_audit(ranked),
        }
    )
    status = "ok" if ranked and ranked[0].score > 0 else "no_match"
    return GroundingSearchResult(
        status=status,
        variant=variant,
        query=query,
        parsed_query=parsed,
        candidates=ranked[:top_k],
        candidate_count=len(canonical),
        searched_candidate_count=len(canonical) if fallback_triggered else len(initial),
        fallback_triggered=fallback_triggered,
        fallback_reasons=fallback_reasons if fallback_triggered else [],
        metadata=metadata,
    )


def available_variants() -> dict[str, dict[str, bool]]:
    """Machine-readable variant/ablation surface for future CLI wiring."""
    return {
        "graph": {"hierarchy": True, "relation": True, "fallback": False, "embedding": False},
        "graph_fallback": {"hierarchy": True, "relation": True, "fallback": True, "embedding": False},
        "flat_lexical": {"hierarchy": False, "relation": True, "fallback": False, "embedding": False},
        "flat_embedding": {"hierarchy": False, "relation": True, "fallback": False, "embedding": True},
    }
