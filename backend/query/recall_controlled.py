"""Recall-controlled hierarchical retrieval for the IEEE Access study.

The legacy graph-vs-flat implementation remains unchanged. This module adds a
fully parameterized traversal so method changes can be tuned on a frozen dev
split and evaluated on a held-out test split without rewriting the benchmark.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.query.affordance import tokenize
from backend.query.graph_vs_flat import _node_text, _root_id, hit_at_k, query_has_view_gt
from backend.query.model_client import _view_text
from backend.query.tokens import count_tokens


@dataclass(frozen=True)
class RecallControlledConfig:
    selection_mode: str = "adaptive_mass"
    branch_score_mode: str = "max_leaf_proxy"
    branch_mass_threshold: float = 0.8
    score_temperature: float = 0.25
    max_branches: int = 5
    fallback_child_limit: int = 2
    candidate_budget: int | None = 6
    safety_shortlist_size: int = 2
    confidence_margin_delta: float = 0.0
    minimum_leaf_score: float | None = None
    fallback_to_flat: bool = True
    boundary_epsilon: float = 0.0

    def validate(self) -> None:
        if self.selection_mode not in {"fixed_top2", "adaptive_mass", "oracle"}:
            raise ValueError(f"unsupported selection_mode: {self.selection_mode}")
        if self.branch_score_mode not in {"node_text", "max_leaf_proxy", "top3_leaf_proxy"}:
            raise ValueError(f"unsupported branch_score_mode: {self.branch_score_mode}")
        if not 0 < self.branch_mass_threshold <= 1:
            raise ValueError("branch_mass_threshold must be in (0, 1]")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        if self.max_branches < 1 or self.fallback_child_limit < 1:
            raise ValueError("branch limits must be positive")
        if self.candidate_budget is not None and self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive or None")
        if self.safety_shortlist_size < 0:
            raise ValueError("safety_shortlist_size must be non-negative")
        if self.confidence_margin_delta < 0:
            raise ValueError("confidence_margin_delta must be non-negative")
        if self.boundary_epsilon < 0:
            raise ValueError("boundary_epsilon must be non-negative")


@dataclass
class PreparedScene:
    scene_id: str
    view_ids: list[str]
    view_texts: dict[str, str]
    view_tokens: dict[str, set[str]]
    branch_ids: list[str]
    branch_texts: dict[str, str]
    branch_tokens: dict[str, set[str]]
    branch_views: dict[str, set[str]]
    positions: dict[str, tuple[float, float, float]]


def _score_tokens(text_tokens: set[str], query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(text_tokens & query_tokens)
    return float(overlap) + overlap / len(query_tokens)


def _rank_scores(scores: dict[str, float]) -> list[tuple[float, str]]:
    return sorted(
        ((float(score), str(item_id)) for item_id, score in scores.items()),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )


def _load_positions(scene_dir: Path | None) -> dict[str, tuple[float, float, float]]:
    if scene_dir is None:
        return {}
    transforms = scene_dir / "transforms.json"
    if not transforms.is_file():
        return {}
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    positions: dict[str, tuple[float, float, float]] = {}
    for frame in payload.get("frames", []):
        view_id = str(frame.get("view_id", ""))
        matrix = frame.get("transform_matrix")
        if not view_id or not isinstance(matrix, list) or len(matrix) < 3:
            continue
        try:
            positions[view_id] = (
                float(matrix[0][3]),
                float(matrix[1][3]),
                float(matrix[2][3]),
            )
        except (IndexError, TypeError, ValueError):
            continue
    return positions


def prepare_scene(bundle: dict[str, Any]) -> PreparedScene:
    tree: dict[str, dict[str, Any]] = bundle["tree"]
    views: dict[str, dict[str, Any]] = bundle["views"]
    root = tree[_root_id(bundle["manifest"])]
    branch_ids = [
        str(child_id)
        for child_id in root.get("children_ids", [])
        if str(child_id) in tree
    ]
    view_ids = sorted(str(view_id) for view_id in views)
    view_texts = {view_id: _view_text(views[view_id]) for view_id in view_ids}
    branch_texts = {branch_id: _node_text(tree[branch_id]) for branch_id in branch_ids}
    return PreparedScene(
        scene_id=str(bundle["scene_id"]),
        view_ids=view_ids,
        view_texts=view_texts,
        view_tokens={view_id: tokenize(text) for view_id, text in view_texts.items()},
        branch_ids=branch_ids,
        branch_texts=branch_texts,
        branch_tokens={branch_id: tokenize(text) for branch_id, text in branch_texts.items()},
        branch_views={
            branch_id: {
                str(view_id)
                for view_id in tree[branch_id].get("view_ids", [])
                if str(view_id) in views
            }
            for branch_id in branch_ids
        },
        positions=_load_positions(Path(bundle["scene_dir"]) if bundle.get("scene_dir") else None),
    )


def _soft_branch_views(scene: PreparedScene, epsilon: float) -> dict[str, set[str]]:
    memberships = {branch_id: set(values) for branch_id, values in scene.branch_views.items()}
    if epsilon <= 0 or not scene.positions:
        return memberships

    centroids: dict[str, tuple[float, float, float]] = {}
    for branch_id, view_ids in scene.branch_views.items():
        points = [scene.positions[view_id] for view_id in view_ids if view_id in scene.positions]
        if points:
            centroids[branch_id] = tuple(
                sum(point[axis] for point in points) / len(points) for axis in range(3)
            )
    if len(centroids) < 2:
        return memberships

    for view_id, point in scene.positions.items():
        distances = sorted(
            (
                math.dist(point, centroid),
                branch_id,
            )
            for branch_id, centroid in centroids.items()
        )
        nearest = distances[0][0]
        for distance, branch_id in distances:
            if distance <= nearest + epsilon:
                memberships[branch_id].add(view_id)
    return memberships


def _branch_scores(
    scene: PreparedScene,
    query_tokens: set[str],
    view_scores: dict[str, float],
    mode: str,
    memberships: dict[str, set[str]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for branch_id in scene.branch_ids:
        if mode == "node_text":
            scores[branch_id] = _score_tokens(scene.branch_tokens[branch_id], query_tokens)
            continue
        values = sorted(
            (view_scores.get(view_id, 0.0) for view_id in memberships[branch_id]),
            reverse=True,
        )
        if not values:
            scores[branch_id] = 0.0
        elif mode == "max_leaf_proxy":
            scores[branch_id] = values[0]
        else:
            scores[branch_id] = sum(values[:3]) / min(3, len(values))
    return scores


def _softmax_probabilities(
    ranked: list[tuple[float, str]], temperature: float
) -> dict[str, float]:
    if not ranked:
        return {}
    maximum = ranked[0][0]
    weights = {
        branch_id: math.exp((score - maximum) / temperature)
        for score, branch_id in ranked
    }
    total = sum(weights.values()) or 1.0
    return {branch_id: value / total for branch_id, value in weights.items()}


def _choose_branches(
    scene: PreparedScene,
    ranked: list[tuple[float, str]],
    probabilities: dict[str, float],
    config: RecallControlledConfig,
    expected_views: set[str],
    memberships: dict[str, set[str]],
) -> list[str]:
    if config.selection_mode == "oracle":
        oracle = [
            branch_id
            for branch_id in scene.branch_ids
            if memberships[branch_id] & expected_views
        ]
        if oracle:
            return oracle[: config.max_branches]
    if config.selection_mode == "fixed_top2":
        return [branch_id for _, branch_id in ranked[:2]]

    chosen: list[str] = []
    cumulative = 0.0
    for _, branch_id in ranked:
        chosen.append(branch_id)
        cumulative += probabilities.get(branch_id, 0.0)
        if cumulative >= config.branch_mass_threshold or len(chosen) >= config.max_branches:
            break
    if not chosen:
        chosen = [branch_id for _, branch_id in ranked[: config.fallback_child_limit]]
    return chosen


def reciprocal_rank(ranked_view_ids: Iterable[str], expected_views: Iterable[str]) -> float:
    expected = {str(value) for value in expected_views}
    if not expected:
        return 0.0
    for index, view_id in enumerate(ranked_view_ids, start=1):
        if str(view_id) in expected:
            return 1.0 / index
    return 0.0


def search_prepared_scene(
    scene: PreparedScene,
    query: dict[str, Any],
    config: RecallControlledConfig,
) -> dict[str, Any]:
    config.validate()
    started = time.perf_counter()
    query_text = str(query.get("query", ""))
    query_tokens = tokenize(query_text)
    expected_views = {str(value) for value in query.get("expected_view_ids", []) if value}
    gt_available = query_has_view_gt(query)

    view_scores = {
        view_id: _score_tokens(scene.view_tokens[view_id], query_tokens)
        for view_id in scene.view_ids
    }
    ranked_global = _rank_scores(view_scores)
    memberships = _soft_branch_views(scene, config.boundary_epsilon)
    branch_scores = _branch_scores(
        scene,
        query_tokens,
        view_scores,
        config.branch_score_mode,
        memberships,
    )
    ranked_branches = _rank_scores(branch_scores)
    probabilities = _softmax_probabilities(ranked_branches, config.score_temperature)
    chosen_branches = _choose_branches(
        scene,
        ranked_branches,
        probabilities,
        config,
        expected_views,
        memberships,
    )

    candidates: set[str] = set()
    for branch_id in chosen_branches:
        candidates.update(memberships.get(branch_id, set()))
    safety_ids = [
        view_id
        for score, view_id in ranked_global
        if score > 0
    ][: config.safety_shortlist_size]
    candidates.update(safety_ids)

    probability_order = [probabilities.get(branch_id, 0.0) for _, branch_id in ranked_branches]
    branch_margin = (
        probability_order[0] - probability_order[1]
        if len(probability_order) > 1
        else probability_order[0] if probability_order else 0.0
    )
    top_candidate_score = max((view_scores.get(value, 0.0) for value in candidates), default=0.0)
    fallback_reasons: list[str] = []
    if config.confidence_margin_delta and branch_margin < config.confidence_margin_delta:
        fallback_reasons.append("branch_margin")
    if config.minimum_leaf_score is not None and top_candidate_score < config.minimum_leaf_score:
        fallback_reasons.append("leaf_score")
    fallback_triggered = bool(fallback_reasons and config.fallback_to_flat)
    if fallback_triggered:
        candidates = set(scene.view_ids)

    ranked_candidates = [
        (score, view_id)
        for score, view_id in ranked_global
        if view_id in candidates
    ]
    if not fallback_triggered and config.candidate_budget is not None:
        ranked_candidates = ranked_candidates[: config.candidate_budget]
    checked_ids = [view_id for _, view_id in ranked_candidates]
    positive_ranked = [view_id for score, view_id in ranked_candidates if score > 0]
    context_texts = [scene.view_texts[view_id] for view_id in checked_ids]
    branch_contexts = [scene.branch_texts[branch_id] for branch_id in scene.branch_ids]
    context_chars = sum(len(text) for text in context_texts + branch_contexts)
    context_tokens = sum(count_tokens(text) for text in context_texts + branch_contexts)

    result = {
        "query_id": str(query.get("query_id", "")),
        "scene_id": str(query.get("scene_id", scene.scene_id)),
        "query": query_text,
        "query_type": query.get("query_type"),
        "config": asdict(config),
        "selected_branches": chosen_branches,
        "branch_count": len(scene.branch_ids),
        "selected_branch_count": len(chosen_branches),
        "selected_branch_ratio": (
            len(chosen_branches) / len(scene.branch_ids) if scene.branch_ids else 0.0
        ),
        "branch_scores": {key: round(value, 6) for key, value in branch_scores.items()},
        "branch_probabilities": {key: round(value, 6) for key, value in probabilities.items()},
        "branch_margin": round(branch_margin, 6),
        "top_leaf_score": round(top_candidate_score, 6),
        "safety_shortlist_ids": safety_ids,
        "fallback_triggered": fallback_triggered,
        "fallback_reasons": fallback_reasons,
        "records_checked": len(checked_ids),
        "checked_view_ids": checked_ids,
        "ranked_view_ids": positive_ranked,
        "serialized_context_chars": context_chars,
        "serialized_context_tokens": context_tokens,
        "proxy_view_comparisons": len(scene.view_ids),
        "proxy_branch_comparisons": len(scene.branch_ids),
        "gt_available": gt_available,
        "expected_view_ids": sorted(expected_views),
        "target_in_candidates": bool(expected_views & set(checked_ids)) if gt_available else None,
        "hit_at_1": hit_at_k(positive_ranked, sorted(expected_views), 1) if gt_available else None,
        "hit_at_3": hit_at_k(positive_ranked, sorted(expected_views), 3) if gt_available else None,
        "mrr": reciprocal_rank(positive_ranked, expected_views) if gt_available else None,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 6),
    }
    return result


def search_flat_prepared_scene(
    scene: PreparedScene,
    query: dict[str, Any],
) -> dict[str, Any]:
    """Exhaustively rank every view with the same lexical leaf scorer."""
    started = time.perf_counter()
    query_text = str(query.get("query", ""))
    query_tokens = tokenize(query_text)
    expected_views = {str(value) for value in query.get("expected_view_ids", []) if value}
    gt_available = query_has_view_gt(query)
    ranked = _rank_scores({
        view_id: _score_tokens(scene.view_tokens[view_id], query_tokens)
        for view_id in scene.view_ids
    })
    checked_ids = [view_id for _, view_id in ranked]
    positive_ranked = [view_id for score, view_id in ranked if score > 0]
    context_texts = [scene.view_texts[view_id] for view_id in checked_ids]
    return {
        "query_id": str(query.get("query_id", "")),
        "scene_id": str(query.get("scene_id", scene.scene_id)),
        "query": query_text,
        "query_type": query.get("query_type"),
        "method": "flat_lexical",
        "records_checked": len(checked_ids),
        "checked_view_ids": checked_ids,
        "ranked_view_ids": positive_ranked,
        "top_leaf_score": round(ranked[0][0], 6) if ranked else 0.0,
        "serialized_context_chars": sum(len(text) for text in context_texts),
        "serialized_context_tokens": sum(count_tokens(text) for text in context_texts),
        "proxy_view_comparisons": len(scene.view_ids),
        "proxy_branch_comparisons": 0,
        "gt_available": gt_available,
        "expected_view_ids": sorted(expected_views),
        "target_in_candidates": bool(expected_views & set(checked_ids)) if gt_available else None,
        "hit_at_1": hit_at_k(positive_ranked, sorted(expected_views), 1) if gt_available else None,
        "hit_at_3": hit_at_k(positive_ranked, sorted(expected_views), 3) if gt_available else None,
        "mrr": reciprocal_rank(positive_ranked, expected_views) if gt_available else None,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 6),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def paired_bootstrap_difference(
    left: list[float],
    right: list[float],
    *,
    samples: int = 10_000,
    seed: int = 20260901,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized non-empty samples")
    rng = random.Random(seed)
    size = len(left)
    observed = sum(a - b for a, b in zip(left, right)) / size
    values = []
    for _ in range(samples):
        difference = 0.0
        for _sample_index in range(size):
            index = rng.randrange(size)
            difference += left[index] - right[index]
        values.append(difference / size)
    values.sort()
    low = values[int(0.025 * samples)]
    high = values[min(samples - 1, int(0.975 * samples))]
    return {
        "estimate": round(observed, 6),
        "ci_95": {"low": round(low, 6), "high": round(high, 6)},
        "samples": samples,
        "seed": seed,
    }
