"""Provider-independent model clients for headless SemanticSplat experiments."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_SCHEMA = "semanticsplat.model_cache.v1"
VALID_MODES = {"stub", "live", "cached_live"}


class ModelClientError(RuntimeError):
    """Base error for provider and cache failures."""


class ModelConfigurationError(ModelClientError):
    """Raised when requested model execution is not configured."""


class ModelCacheError(ModelClientError):
    """Raised when cached-live data is missing or has invalid provenance."""


class ModelProviderError(ModelClientError):
    """Raised when a configured live provider request fails."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelClient(ABC):
    """Provider-independent semantic scene and query interface."""

    mode: str
    provider: str | None
    model: str | None
    version: str | None
    temperature: float

    def __init__(
        self,
        *,
        mode: str,
        provider: str | None,
        model: str | None,
        version: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported model mode: {mode}")
        self.mode = mode
        self.provider = provider
        self.model = model
        self.version = version
        self.temperature = temperature
        self._stats: dict[str, Any] = {
            "model_call_count": 0,
            "cache_hits": 0,
            "retry_count": 0,
            "failure_count": 0,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    @abstractmethod
    def describe_view(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a semantic description for one view."""

    @abstractmethod
    def build_tree(
        self,
        view_summaries: dict[str, dict[str, Any]],
        scene_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return a semantic tree keyed by node ID."""

    @abstractmethod
    def answer_query(
        self,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a canonical-query payload without run-level metrics."""

    def stats_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._stats)

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "temperature": self.temperature,
        }

    def _record_usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        aliases = {
            "input_tokens": ("input_tokens", "prompt_tokens", "input"),
            "output_tokens": ("output_tokens", "completion_tokens", "output"),
            "total_tokens": ("total_tokens", "total"),
        }
        for target, keys in aliases.items():
            value = next((usage.get(key) for key in keys if isinstance(usage.get(key), int)), None)
            if value is None:
                continue
            current = self._stats[target]
            self._stats[target] = int(value) + (int(current) if current is not None else 0)


_STOPWORDS = {
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "me", "my", "can", "could", "please",
    "there", "all", "inside", "area", "someone", "people", "guests",
    "place", "visitor", "visitors", "suitable", "around", "near", "by", "with",
    "between", "behind", "surrounded", "side", "sides",
}

_SYNONYMS = {
    "slides": {"projection", "screen", "presentation"},
    "presenter": {"presentation", "podium", "lectern", "screen"},
    "drinks": {"bar", "hospitality"},
    "check": {"reception"},
    "get": {"counter", "desk", "service", "help", "screen", "screens"},
    "ask": {"counter", "desk", "service", "help", "reception"},
    "help": {"counter", "desk", "service", "reception"},
    "emergency": {"exit", "egress"},
    "leave": {"exit", "egress"},
    "catering": {"service", "prep", "plates"},
    "play": {"piano", "performance"},
    "av": {"projector", "equipment", "presentation"},
    "column": {"columns", "pillar", "pillars"},
    "columns": {"column", "pillar", "pillars"},
    "pillar": {"column", "columns", "pillars"},
    "pillars": {"column", "columns", "pillar"},
    "sofa": {"sofas", "couch", "couches"},
    "sofas": {"sofa", "couch", "couches"},
    "sit": {"seat", "seats", "seating", "chair", "chairs", "armchair", "armchairs", "sofa", "sofas", "bench", "benches", "lounge"},
    "sitting": {"seat", "seats", "seating", "chair", "chairs", "armchair", "armchairs", "sofa", "sofas", "bench", "benches", "lounge"},
    "seat": {"sit", "seating", "chair", "chairs", "sofa", "sofas", "bench"},
    "seating": {"sit", "seat", "seats", "chair", "chairs", "sofa", "sofas", "bench", "lounge"},
    "meeting": {"table", "tables", "chair", "chairs", "seating", "banquet", "conference", "round"},
    "exhibit": {"exhibits", "exhibition", "display", "panel", "gallery", "stand", "ruin"},
    "exhibits": {"exhibit", "exhibition", "display", "panel", "gallery", "stand", "ruin"},
    "viewing": {"view", "exhibit", "exhibits", "exhibition", "display", "panel", "gallery", "stand", "ruin"},
    "perform": {"performing", "performance", "stage", "curtain", "floor"},
    "performing": {"perform", "performance", "stage", "curtain", "floor"},
    "watching": {"watch", "performance", "stage", "seating", "audience"},
    "walking": {"walk", "walkway", "sidewalk", "pavement", "path"},
    "walk": {"walking", "walkway", "sidewalk", "pavement", "path"},
    "levels": {"stairs", "staircase", "steps"},
}

_TOKEN_ALIASES = {
    "collumn": "column",
    "collumns": "columns",
}


def _tokens(text: str, *, expand: bool = True) -> set[str]:
    values = {
        _TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    }
    if not expand:
        return values
    expanded = set(values)
    for value in values:
        expanded.update(_SYNONYMS.get(value, set()))
    return expanded


def _view_text(view: dict[str, Any]) -> str:
    pieces: list[str] = [
        str(view.get("scene_summary", "")),
        str(view.get("summary", "")),
        str(view.get("room_type", "")),
        str(view.get("facing", "")),
        str(view.get("functional_context", "")),
        str(view.get("free_text_notes", "")),
    ]
    pieces.extend(str(value) for value in view.get("spatial_relations", []) if value is not None)
    for region in view.get("visible_regions", []):
        if isinstance(region, dict):
            pieces.extend((str(region.get("label", "")), str(region.get("approx_location", ""))))
    for obj in _view_objects(view):
        if not isinstance(obj, dict):
            continue
        pieces.append(str(obj.get("label", "")))
        pieces.append(str(obj.get("approx_location", "")))
        attributes = obj.get("attributes")
        if isinstance(attributes, dict):
            pieces.extend(str(value) for value in attributes.values() if value is not None)
        elif isinstance(attributes, list):
            pieces.extend(str(value) for value in attributes if value is not None)
    for landmark in view.get("landmarks", []):
        if isinstance(landmark, dict):
            pieces.extend(
                (
                    str(landmark.get("label", "")),
                    str(landmark.get("kind", "")),
                    str(landmark.get("approx_location", "")),
                )
            )
    return " ".join(pieces)


def _view_objects(view: dict[str, Any]) -> list[dict[str, Any]]:
    value = view.get("visible_objects")
    if not isinstance(value, list):
        value = view.get("objects", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _object_text(obj: dict[str, Any]) -> str:
    pieces = [
        str(obj.get("label", "")),
        str(obj.get("approx_location", "")),
    ]
    attributes = obj.get("attributes")
    if isinstance(attributes, dict):
        pieces.extend(str(value) for value in attributes.values() if value is not None)
    elif isinstance(attributes, list):
        pieces.extend(str(value) for value in attributes if value is not None)
    return " ".join(pieces)


def _object_id(obj: dict[str, Any]) -> str | None:
    """Return an explicit object identifier without inventing one from a label."""
    for key in ("object_id", "instance_id", "id"):
        value = obj.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value).strip()
            if text:
                return text
    for bbox_key in ("bbox_3d", "bbox_2d"):
        bbox = obj.get(bbox_key)
        if isinstance(bbox, dict):
            for key in ("object_id", "instance_id", "id"):
                value = bbox.get(key)
                if value is not None and not isinstance(value, (dict, list, tuple)):
                    text = str(value).strip()
                    if text:
                        return text
    attributes = obj.get("attributes")
    if isinstance(attributes, dict):
        values = attributes.values()
    elif isinstance(attributes, (list, tuple)):
        values = attributes
    else:
        values = ()
    for value in values:
        match = re.fullmatch(r"\s*(?:object|instance)_id\s*:\s*(\S+)\s*", str(value), re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _score(text: str, query_tokens: set[str]) -> tuple[int, float]:
    haystack = _tokens(text, expand=False)
    overlap = len(query_tokens & haystack)
    ratio = overlap / len(query_tokens) if query_tokens else 0.0
    return overlap, ratio


def _object_rank(
    obj: dict[str, Any],
    query_tokens: set[str],
    exact_query_tokens: set[str] | None = None,
) -> tuple[int, int, int, int, int, float, float, float, float, int]:
    exact_query_tokens = exact_query_tokens or query_tokens
    text_overlap, text_ratio = _score(_object_text(obj), query_tokens)
    label_overlap, label_ratio = _score(str(obj.get("label", "")), query_tokens)
    exact_text_overlap, exact_text_ratio = _score(_object_text(obj), exact_query_tokens)
    exact_label_overlap, exact_label_ratio = _score(str(obj.get("label", "")), exact_query_tokens)
    has_bbox = 1 if isinstance(obj.get("bbox_3d"), dict) else 0
    label_tokens = _tokens(str(obj.get("label", "")), expand=False)
    extra_label_terms = len(label_tokens - exact_query_tokens)
    label_compactness = -extra_label_terms if exact_text_overlap > 0 else 0
    return (
        text_overlap,
        exact_text_overlap,
        label_compactness,
        label_overlap,
        exact_label_overlap,
        text_ratio,
        exact_text_ratio,
        label_ratio,
        exact_label_ratio,
        has_bbox,
    )


def _best_object_rank(
    view: dict[str, Any],
    query_tokens: set[str],
    exact_query_tokens: set[str],
) -> tuple[int, int, int, int, int, float, float, float, float, int]:
    ranks = [_object_rank(obj, query_tokens, exact_query_tokens) for obj in _view_objects(view)]
    return max(ranks) if ranks else (0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)


def _has_full_exact_object_match(views: dict[str, dict[str, Any]], exact_query_tokens: set[str]) -> bool:
    if not exact_query_tokens:
        return False
    for view in views.values():
        for obj in _view_objects(view):
            label_tokens = _tokens(str(obj.get("label", "")), expand=False)
            if len(exact_query_tokens) == 1 and label_tokens != exact_query_tokens:
                continue
            object_tokens = _tokens(_object_text(obj), expand=False)
            if exact_query_tokens <= object_tokens:
                return True
    return False


def _box_bounds(box: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(box, dict):
        return None
    if isinstance(box.get("min"), list) and isinstance(box.get("max"), list):
        lower = box["min"]
        upper = box["max"]
    elif isinstance(box.get("center"), (list, tuple)) and isinstance(box.get("size"), (list, tuple)):
        center = list(box["center"])
        size = list(box["size"])
        if len(center) != 3 or len(size) != 3:
            return None
        lower = [float(center[index]) - float(size[index]) / 2 for index in range(3)]
        upper = [float(center[index]) + float(size[index]) / 2 for index in range(3)]
    else:
        return None
    if len(lower) != 3 or len(upper) != 3:
        return None
    try:
        lo = [float(item) for item in lower]
        hi = [float(item) for item in upper]
    except (TypeError, ValueError):
        return None
    if any(hi[index] <= lo[index] for index in range(3)):
        return None
    return lo, hi


def _merge_bbox_3d(boxes: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    bounds = [_box_bounds(box) for box in boxes]
    valid = [item for item in bounds if item is not None]
    if not valid:
        return None
    lower = [min(item[0][index] for item in valid) for index in range(3)]
    upper = [max(item[1][index] for item in valid) for index in range(3)]
    return {
        "center": [(lower[index] + upper[index]) / 2 for index in range(3)],
        "size": [upper[index] - lower[index] for index in range(3)],
        "label": label,
    }


def _functional_bbox_candidates(
    views: dict[str, dict[str, Any]],
    query_tokens: set[str],
) -> list[tuple[int, float, int, float, str, str, Any, dict[str, Any]]]:
    candidates: list[tuple[int, float, int, float, str, str, Any, dict[str, Any]]] = []
    for view_id, view in views.items():
        view_overlap, view_ratio = _score(_view_text(view), query_tokens)
        if view_overlap <= 0:
            continue
        for obj in _view_objects(view):
            label = str(obj.get("label", ""))
            object_overlap, object_ratio = _score(label, query_tokens)
            bbox_3d = obj.get("bbox_3d")
            if object_overlap <= 0 or not isinstance(bbox_3d, dict):
                continue
            candidates.append((
                object_overlap,
                object_ratio,
                view_overlap,
                view_ratio,
                str(view_id),
                label,
                obj.get("bbox_2d"),
                bbox_3d,
            ))
    candidates.sort(key=lambda item: item[:6], reverse=True)
    return candidates


def _node_text(node: dict[str, Any]) -> str:
    pieces = [
        str(node.get("name", "")),
        str(node.get("summary", "")),
        str(node.get("approx_location", "")),
    ]
    for key in ("attributes", "zone_labels", "observations"):
        value = node.get(key)
        if isinstance(value, dict):
            pieces.extend(str(item) for item in value.values() if item is not None)
        elif isinstance(value, list):
            pieces.extend(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item) for item in value if item is not None)
    return " ".join(pieces)


def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("node_type", node.get("type", "")))


def _node_id(node: dict[str, Any], fallback: str) -> str:
    return str(node.get("node_id") or fallback)


def _reachable_tree(tree: dict[str, dict[str, Any]]) -> tuple[str | None, set[str], dict[str, str]]:
    root_id = next((key for key, node in tree.items() if _node_type(node) == "root"), None)
    if root_id is None:
        return None, set(tree), {}
    reachable: set[str] = set()
    parents: dict[str, str] = {}
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current in reachable or current not in tree:
            continue
        reachable.add(current)
        for child in tree[current].get("children_ids", []):
            child_id = str(child)
            if child_id in tree:
                parents[child_id] = current
                stack.append(child_id)
    return root_id, reachable, parents


def _path_to(root_id: str | None, target_id: str | None, parents: dict[str, str]) -> list[str]:
    if target_id is None:
        return [root_id] if root_id else []
    path = [target_id]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    path.reverse()
    if root_id and (not path or path[0] != root_id):
        path.insert(0, root_id)
    return path


class StubModelClient(ModelClient):
    """Deterministic semantic-index fallback with no external model calls."""

    def __init__(self, *, temperature: float = 0.0) -> None:
        super().__init__(
            mode="stub",
            provider="deterministic_local",
            model="semantic_index_lexical_v1",
            version="1",
            temperature=temperature,
        )

    def describe_view(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        existing = metadata.get("existing_analysis")
        if isinstance(existing, dict):
            return deepcopy(existing)
        view_id = str(metadata.get("view_id") or Path(image_path).stem)
        return {
            "view_id": view_id,
            "scene_summary": "No saved semantic description is available for this view.",
            "room_type": "unknown",
            "lighting": "unknown",
            "objects": [],
            "spatial_relations": [],
            "functional_context": None,
            "facing": "unknown",
            "visible_landmarks": [],
        }

    def build_tree(
        self,
        view_summaries: dict[str, dict[str, Any]],
        scene_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        existing = scene_context.get("existing_tree")
        if isinstance(existing, dict) and existing:
            return deepcopy(existing)
        view_ids = sorted(view_summaries)
        return {
            "root_stub": {
                "node_id": "root_stub",
                "type": "root",
                "name": "Deterministic stub root",
                "summary": "Flat fallback tree because no saved hierarchy was available.",
                "view_ids": [],
                "children_ids": ["leaf_stub_all_views"],
            },
            "leaf_stub_all_views": {
                "node_id": "leaf_stub_all_views",
                "type": "leaf",
                "name": "All indexed views",
                "summary": "Deterministic flat fallback containing every selected view.",
                "view_ids": view_ids,
                "children_ids": [],
            },
        }

    def answer_query(
        self,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(query, dict):
            query_text = str(query.get("query", ""))
            query_type = str(query.get("query_type", "simple_object"))
        else:
            query_text = str(query)
            query_type = "simple_object"

        query_tokens = _tokens(query_text)
        exact_query_tokens = _tokens(query_text, expand=False)
        ranked: list[tuple[Any, ...]] = []
        for view_id, view in views.items():
            overlap, ratio = _score(_view_text(view), query_tokens)
            exact_overlap, exact_ratio = _score(_view_text(view), exact_query_tokens)
            (
                object_text_overlap,
                exact_object_text_overlap,
                label_compactness,
                object_label_overlap,
                exact_object_label_overlap,
                object_text_ratio,
                exact_object_text_ratio,
                object_label_ratio,
                exact_object_label_ratio,
                has_object_bbox,
            ) = (
                _best_object_rank(view, query_tokens, exact_query_tokens)
            )
            ranked.append((
                overlap,
                exact_overlap,
                object_text_overlap,
                exact_object_text_overlap,
                label_compactness,
                object_label_overlap,
                exact_object_label_overlap,
                ratio,
                exact_ratio,
                object_text_ratio,
                exact_object_text_ratio,
                object_label_ratio,
                exact_object_label_ratio,
                has_object_bbox,
                view_id,
                view,
            ))
        ranked.sort(key=lambda item: item[:-1], reverse=True)

        best_row = ranked[0] if ranked else (0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", {})
        best_overlap = int(best_row[0])
        best_ratio = float(best_row[7])
        best_view_id = str(best_row[14])
        best_view = best_row[15]
        found = best_overlap > 0
        if query_type == "negative":
            found = found and _has_full_exact_object_match(views, exact_query_tokens)
        matched_object: str | None = None
        selected_object_id: str | None = None
        bbox_2d: Any = None
        bbox_3d: Any = None
        checked_object_ids = sorted({
            object_id
            for view in views.values()
            for obj in _view_objects(view)
            if (object_id := _object_id(obj)) is not None
        })
        checked_object_count = sum(len(_view_objects(view)) for view in views.values())
        if found:
            object_candidates = []
            for obj in _view_objects(best_view):
                object_candidates.append(
                    (
                        _object_rank(obj, query_tokens, exact_query_tokens),
                        str(obj.get("label", "")),
                        obj,
                    )
                )
            object_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if object_candidates and object_candidates[0][0][0] > 0:
                _, matched_object, selected_object = object_candidates[0]
            else:
                selected_object = next((obj for obj in _view_objects(best_view) if obj.get("label")), None)
                if selected_object is not None:
                    matched_object = str(selected_object.get("label"))
            if selected_object is not None:
                selected_object_id = _object_id(selected_object)
                bbox_2d = selected_object.get("bbox_2d")
                bbox_3d = selected_object.get("bbox_3d")

        merged_functional_views: list[str] = []
        merged_functional_bbox = False
        explicit_functional_object = bool(
            query_type == "functional"
            and matched_object
            and (_tokens(matched_object, expand=False) & exact_query_tokens)
        )
        if found and query_type == "functional" and not explicit_functional_object:
            functional_candidates = _functional_bbox_candidates(views, query_tokens)
            if functional_candidates:
                strongest_overlap = functional_candidates[0][0]
                minimum_overlap = max(1, strongest_overlap - 1)
                selected_candidates = [
                    candidate
                    for candidate in functional_candidates
                    if candidate[0] >= minimum_overlap
                ][:12]
                if len(selected_candidates) > 1:
                    merged_bbox = _merge_bbox_3d(
                        [candidate[7] for candidate in selected_candidates],
                        label="merged functional semantic candidates",
                    )
                    if merged_bbox is not None:
                        bbox_2d = None
                        bbox_3d = merged_bbox
                        selected_object_id = None
                        merged_functional_views = sorted({candidate[4] for candidate in selected_candidates})
                        merged_functional_bbox = True

        root_id, reachable, parents = _reachable_tree(tree)
        selected_node_id = None
        checked_node_ids: list[str] = []
        if found:
            candidates = []
            matched_tokens = _tokens(matched_object or "", expand=False)
            for key, node in tree.items():
                node_id = _node_id(node, key)
                if node_id not in reachable or best_view_id not in as_string_ids(node.get("view_ids")):
                    continue
                checked_node_ids.append(node_id)
                node_text = _node_text(node)
                name_text = str(node.get("name", ""))
                matched_name_overlap, matched_name_ratio = _score(name_text, matched_tokens)
                query_name_overlap, query_name_ratio = _score(name_text, query_tokens)
                match_overlap, match_ratio = _score(node_text, matched_tokens)
                query_overlap, query_ratio = _score(node_text, query_tokens)
                depth = len(_path_to(root_id, node_id, parents))
                node_type = _node_type(node)
                type_priority = {
                    "object": 4,
                    "landmark": 3,
                    "region": 2,
                    "zone": 1,
                    "root": 0,
                }.get(node_type, 0)
                candidates.append((
                    matched_name_overlap,
                    matched_name_ratio,
                    query_name_overlap,
                    query_name_ratio,
                    match_overlap,
                    match_ratio,
                    query_overlap,
                    query_ratio,
                    type_priority,
                    depth,
                    node_id,
                ))
            if candidates:
                candidates.sort(reverse=True)
                selected_node_id = candidates[0][-1]
        visited_nodes = _path_to(root_id, selected_node_id, parents)
        if not found and root_id:
            visited_nodes = [root_id]
            visited_nodes.extend(str(child) for child in tree[root_id].get("children_ids", []) if str(child) in tree)
        if not checked_node_ids:
            checked_node_ids = list(visited_nodes)

        selected_views = sorted(set([best_view_id, *merged_functional_views])) if found else []
        checked_view_ids = sorted(views)
        target = " ".join(sorted(query_tokens))
        warnings = [
            "Stub mode uses deterministic lexical matching over saved semantic descriptions.",
            "This result is not live or cached-live model reasoning.",
        ]
        if merged_functional_bbox:
            warnings.append("Functional query bbox_3d merges multiple matched semantic object boxes.")
        return {
            "structured_plan": {
                "target": target,
                "query_type": query_type,
                "parser": "deterministic_tokenizer_v1",
            },
            "visited_nodes": visited_nodes,
            "traversal_path": visited_nodes,
            "selected_views": selected_views,
            "checked_node_ids": checked_node_ids,
            "checked_view_ids": checked_view_ids,
            "checked_object_ids": checked_object_ids,
            "checked_object_count": checked_object_count,
            "search_variant": "deterministic_lexical_v1",
            "fallback_used": True,
            "fallback_reason": "stub_mode_uses_deterministic_semantic_index_fallback",
            "result": {
                "found": found,
                "matched_object": matched_object,
                "selected_object_id": selected_object_id,
                "selected_node_id": selected_node_id,
                "selected_view_id": best_view_id if found else None,
                "bbox_2d": bbox_2d,
                "bbox_3d": bbox_3d,
                "camera_pose": None,
                "confidence": round(best_ratio, 4) if found else 0.0,
                "explanation": (
                    f"Deterministic lexical fallback selected {best_view_id} with {best_overlap} token matches."
                    if found
                    else "Deterministic lexical fallback found no matching semantic-index tokens."
                ),
            },
            "warnings": warnings,
        }


def as_string_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class CachedModelClient(ModelClient):
    """Replay raw responses whose cache provenance identifies a live source."""

    def __init__(self, cache_dir: str | Path, *, temperature: float = 0.0) -> None:
        super().__init__(
            mode="cached_live",
            provider="cache_replay",
            model=None,
            version=CACHE_SCHEMA,
            temperature=temperature,
        )
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def request_hash(method: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"method": method, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def cache_path(self, method: str, payload: dict[str, Any]) -> Path:
        return self.cache_dir / method / f"{self.request_hash(method, payload)}.json"

    @staticmethod
    def write_cache_entry(
        cache_dir: str | Path,
        method: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        *,
        provider: str,
        model: str,
        version: str | None = None,
        token_usage: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> Path:
        client = CachedModelClient(cache_dir)
        path = client.cache_path(method, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_schema": CACHE_SCHEMA,
            "source_mode": "live",
            "provider": provider,
            "model": model,
            "version": version,
            "request_hash": path.stem,
            "method": method,
            "token_usage": token_usage,
            "response": response,
            "raw_response": raw_response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def _replay(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.cache_path(method, payload)
        if not path.exists():
            self._stats["failure_count"] += 1
            raise ModelCacheError(f"Missing cached-live response: {path}")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._stats["failure_count"] += 1
            raise ModelCacheError(f"Invalid cache file {path}: {exc}") from exc
        if envelope.get("cache_schema") != CACHE_SCHEMA or envelope.get("source_mode") != "live":
            self._stats["failure_count"] += 1
            raise ModelCacheError(f"Cache entry is not verified live provenance: {path}")
        if envelope.get("request_hash") != path.stem or envelope.get("method") != method:
            self._stats["failure_count"] += 1
            raise ModelCacheError(f"Cache request metadata mismatch: {path}")
        response = envelope.get("response")
        if not isinstance(response, dict):
            self._stats["failure_count"] += 1
            raise ModelCacheError(f"Cache response must be a JSON object: {path}")
        self._stats["cache_hits"] += 1
        self._record_usage(envelope.get("token_usage"))
        self.provider = str(envelope.get("provider") or "cache_replay")
        self.model = str(envelope.get("model") or "unknown")
        self.version = str(envelope.get("version")) if envelope.get("version") is not None else None
        return deepcopy(response)

    def describe_view(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        existing = metadata.get("existing_analysis")
        if isinstance(existing, dict):
            return deepcopy(existing)
        payload = {"image_path": str(image_path), "metadata": metadata}
        return self._replay("describe_view", payload)

    def build_tree(
        self,
        view_summaries: dict[str, dict[str, Any]],
        scene_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        existing = scene_context.get("existing_tree")
        if isinstance(existing, dict) and existing:
            return deepcopy(existing)
        payload = {"view_summaries": view_summaries, "scene_context": scene_context}
        response = self._replay("build_tree", payload)
        return response

    def answer_query(
        self,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {"query": query, "tree": tree, "views": views}
        return self._replay("answer_query", payload)


class OpenAIModelClient(ModelClient):
    """Live query reasoning through the OpenAI Responses API."""

    endpoint = "https://api.openai.com/v1/responses"
    reasoning_model_prefixes = ("gpt-5", "o1", "o3", "o4")
    unsupported_reasoning_sampling_parameters = (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_dir: str | Path,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            mode="live",
            provider="openai",
            model=model,
            version="responses_api_v1",
            temperature=temperature,
        )
        self._api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.max_retries = max(0, int(max_retries))
        self.timeout_seconds = float(timeout_seconds)

    def describe_view(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        existing = metadata.get("existing_analysis")
        if isinstance(existing, dict):
            return deepcopy(existing)
        raise ModelConfigurationError(
            "OpenAI live evaluation requires an existing semantic description for every selected view"
        )

    def build_tree(
        self,
        view_summaries: dict[str, dict[str, Any]],
        scene_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        existing = scene_context.get("existing_tree")
        if isinstance(existing, dict) and existing:
            return deepcopy(existing)
        raise ModelConfigurationError(
            "OpenAI live evaluation requires an existing validated semantic tree"
        )

    def answer_query(
        self,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {"query": query, "tree": tree, "views": views}
        request_payload = self._request_payload(query, tree, views)
        raw_response = self._call_provider(request_payload)
        answer = self._parse_answer(raw_response, tree=tree, views=views)
        usage = raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None
        self._record_usage(usage)
        CachedModelClient.write_cache_entry(
            self.cache_dir,
            "answer_query",
            payload,
            answer,
            provider="openai",
            model=str(self.model),
            version=self.version,
            token_usage=usage,
            raw_response=raw_response,
        )
        return answer

    def _request_payload(
        self,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._query_prompt(query, tree, views),
        }
        model = str(self.model or "").strip().lower()
        if not model.startswith(self.reasoning_model_prefixes):
            payload["temperature"] = self.temperature
        else:
            for parameter in self.unsupported_reasoning_sampling_parameters:
                payload.pop(parameter, None)
        return payload

    @staticmethod
    def _compact_tree(tree: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": _node_id(node, key),
                "node_type": _node_type(node),
                "name": node.get("name"),
                "summary": node.get("summary"),
                "view_ids": as_string_ids(node.get("view_ids")),
                "children_ids": as_string_ids(node.get("children_ids")),
            }
            for key, node in sorted(tree.items())
        ]

    @staticmethod
    def _compact_views(views: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        fields = (
            "summary",
            "scene_summary",
            "room_type",
            "visible_regions",
            "visible_objects",
            "objects",
            "landmarks",
            "free_text_notes",
        )
        return {
            view_id: {field: view[field] for field in fields if field in view}
            for view_id, view in sorted(views.items())
        }

    @classmethod
    def _query_prompt(
        cls,
        query: str | dict[str, Any],
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> str:
        context = {
            "query": query,
            "tree": cls._compact_tree(tree),
            "views": cls._compact_views(views),
        }
        return (
            "Answer the scene query using only the supplied manual semantic index. "
            "Do not invent objects, views, nodes, geometry, or ground truth. Return one JSON object only with "
            "structured_plan (object), visited_nodes (array), traversal_path (array), selected_views (array), "
            "checked_node_ids (array), checked_view_ids (array), checked_object_ids (array), search_variant "
            "(string), fallback_used (boolean), fallback_reason (string or null), result (object with found, "
            "matched_object, selected_object_id, selected_node_id, selected_view_id, bbox_2d, bbox_3d, camera_pose, "
            "confidence, explanation), and warnings (array). Use null for unavailable boxes, IDs, and pose. "
            "Confidence must be a number from 0 to 1. The manual index is not independent ground truth.\n\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
        )

    def _call_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._stats["model_call_count"] += 1
            try:
                response = self._request_provider(payload)
                if not isinstance(response, dict):
                    raise ModelProviderError("OpenAI response must be a JSON object")
                return response
            except (ModelProviderError, OSError, ValueError) as exc:
                last_error = exc
                retryable = not isinstance(exc, ModelProviderError) or exc.retryable
                if not retryable or attempt >= self.max_retries:
                    self._stats["failure_count"] += 1
                    break
                self._stats["retry_count"] += 1
                time.sleep(min(2 ** attempt, 4))
        raise ModelProviderError(f"OpenAI request failed after retries: {last_error}") from last_error

    def _request_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SemanticSplat-evaluation/1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            raise ModelProviderError(
                f"OpenAI HTTP {exc.code}: {detail}",
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise ModelProviderError(f"OpenAI network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ModelProviderError("OpenAI returned invalid JSON", retryable=False) from exc

    @staticmethod
    def _output_text(response: dict[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        pieces: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        if not pieces:
            raise ModelProviderError("OpenAI response did not contain output text")
        return "\n".join(pieces).strip()

    @classmethod
    def _parse_answer(
        cls,
        response: dict[str, Any],
        *,
        tree: dict[str, dict[str, Any]],
        views: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        text = cls._output_text(response)
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            answer = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelProviderError("OpenAI output was not valid JSON") from exc
        if not isinstance(answer, dict):
            raise ModelProviderError("OpenAI answer must be a JSON object")
        result = answer.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("found"), bool):
            raise ModelProviderError("OpenAI answer requires result.found as a boolean")
        confidence = result.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ModelProviderError("OpenAI answer requires numeric result.confidence")
        result["confidence"] = max(0.0, min(1.0, float(confidence)))

        warnings = [str(item) for item in answer.get("warnings", []) if item is not None]
        valid_nodes = set(tree)
        valid_views = set(views)
        answer["visited_nodes"] = [str(item) for item in answer.get("visited_nodes", []) if str(item) in valid_nodes]
        answer["selected_views"] = [str(item) for item in answer.get("selected_views", []) if str(item) in valid_views]
        answer["checked_view_ids"] = [
            str(item) for item in answer.get("checked_view_ids", []) if str(item) in valid_views
        ]
        answer["checked_node_ids"] = [
            str(item) for item in answer.get("checked_node_ids", []) if str(item) in valid_nodes
        ]
        answer["traversal_path"] = [
            str(item) for item in answer.get("traversal_path", answer["visited_nodes"]) if str(item) in valid_nodes
        ]
        answer["checked_object_ids"] = as_string_ids(answer.get("checked_object_ids"))
        if result.get("selected_node_id") not in valid_nodes:
            if result.get("selected_node_id") is not None:
                warnings.append("Provider-selected node was absent from the semantic index and was cleared.")
            result["selected_node_id"] = None
        if result.get("selected_view_id") not in valid_views:
            if result.get("selected_view_id") is not None:
                warnings.append("Provider-selected view was absent from the semantic index and was cleared.")
            result["selected_view_id"] = None
        for field in ("matched_object", "selected_object_id", "bbox_2d", "bbox_3d", "camera_pose"):
            result.setdefault(field, None)
        result.setdefault("explanation", "")
        answer["search_variant"] = str(answer.get("search_variant") or "provider_semantic_index")
        answer["fallback_used"] = bool(answer.get("fallback_used", False))
        answer["fallback_reason"] = answer.get("fallback_reason")
        answer["structured_plan"] = answer.get("structured_plan") if isinstance(answer.get("structured_plan"), dict) else {}
        warnings.append("Live provider reasoning used a manual semantic index, not independent ground truth.")
        answer["warnings"] = sorted(set(warnings))
        return answer


def create_model_client(
    mode: str,
    *,
    cache_dir: str | Path | None = None,
    temperature: float = 0.0,
) -> ModelClient:
    """Create a configured client without embedding credentials in code."""
    if temperature != 0:
        raise ModelConfigurationError("Evaluation clients require temperature=0")
    if mode == "stub":
        return StubModelClient(temperature=temperature)
    if mode == "cached_live":
        if cache_dir is None:
            raise ModelConfigurationError("cached_live mode requires a model_cache directory")
        return CachedModelClient(cache_dir, temperature=temperature)
    if mode == "live":
        provider = os.getenv("SEMANTICSPLAT_PROVIDER")
        model = os.getenv("SEMANTICSPLAT_MODEL")
        api_key = os.getenv("SEMANTICSPLAT_API_KEY")
        missing = [
            name
            for name, value in (
                ("SEMANTICSPLAT_PROVIDER", provider),
                ("SEMANTICSPLAT_MODEL", model),
                ("SEMANTICSPLAT_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise ModelConfigurationError(
                "live mode is unavailable; missing environment variables: " + ", ".join(missing)
            )
        if cache_dir is None:
            raise ModelConfigurationError("live mode requires a model_cache directory")
        if str(provider).strip().lower() != "openai":
            raise ModelConfigurationError(
                f"unsupported live provider '{provider}'; supported providers: openai"
            )
        return OpenAIModelClient(
            api_key=str(api_key),
            model=str(model),
            cache_dir=cache_dir,
            temperature=temperature,
        )
    raise ModelConfigurationError(f"Unsupported mode: {mode}")
