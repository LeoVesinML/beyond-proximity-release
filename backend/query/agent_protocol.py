"""Agent-agnostic semantic traversal and replayable query traces.

The MCP server remains infrastructure-only. Cursor Agent or another language
model can drive the same protocol by calling the MCP tools, while this module
provides a pinned local semantic encoder for reproducible graph-vs-flat runs.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from backend.query.graph_vs_flat import (
    SearchMetrics,
    _count_semantic_items,
    _is_leaf_node,
    _node_text,
    _root_id,
)
from backend.query.model_client import _view_objects, _view_text


TRACE_SCHEMA_VERSION = "semanticsplat.agent_query_trace.v1"
METHOD_PROTOCOL_ID = "mcp_hierarchical_semantic_traversal.v1"
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
TRACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SemanticEncoder(Protocol):
    """Minimal semantic model interface used by the reproducible runner."""

    model_name: str
    backend_name: str
    token_method: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per input string."""

    def token_count(self, text: str) -> int:
        """Return the model tokenizer count for one string."""


class FastEmbedSemanticEncoder:
    """Pinned local BGE encoder with native tokenizer accounting."""

    backend_name = "fastembed_onnx_cpu"

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        cache_dir: Path | str = Path(".cache") / "fastembed",
        *,
        threads: int = 4,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_method = f"fastembed_native_tokenizer:{model_name}"
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(self.cache_dir),
            threads=threads,
            cuda=False,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = [str(text) for text in texts]
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.vstack(list(self._model.embed(values))).astype(np.float32)
        return _normalize_rows(vectors)

    def token_count(self, text: str) -> int:
        return int(self._model.token_count(str(text)))


@dataclass
class AgentStep:
    index: int
    phase: str
    tool_name: str
    request: dict[str, Any]
    response: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    token_method: str | None = None
    created_at: str = field(default_factory=_utc_now)


@dataclass
class AgentTrace:
    trace_id: str
    scene_id: str
    query: str
    query_id: str = ""
    client_name: str = "unspecified_agent"
    model_name: str = "unreported"
    status: str = "running"
    schema_version: str = TRACE_SCHEMA_VERSION
    method_protocol_id: str = METHOD_PROTOCOL_ID
    created_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    result: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accounting"] = trace_accounting(data)
        return data


def new_agent_trace(
    *,
    scene_id: str,
    query: str,
    query_id: str = "",
    client_name: str,
    model_name: str,
    trace_id: str | None = None,
) -> AgentTrace:
    value = trace_id or f"trace_{uuid.uuid4().hex}"
    validate_trace_id(value)
    required_values = {
        "scene_id": scene_id,
        "query": query,
        "client_name": client_name,
        "model_name": model_name,
    }
    for name, item in required_values.items():
        if not str(item).strip():
            raise ValueError(f"{name} must not be empty")
    return AgentTrace(
        trace_id=value,
        scene_id=str(scene_id),
        query=str(query),
        query_id=str(query_id),
        client_name=str(client_name),
        model_name=str(model_name),
    )


def append_agent_step(
    trace: AgentTrace,
    *,
    phase: str,
    tool_name: str,
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    token_method: str | None = None,
) -> AgentStep:
    if trace.status != "running":
        raise ValueError("Cannot append a step to a completed agent trace")
    if not str(phase).strip() or not str(tool_name).strip():
        raise ValueError("phase and tool_name must not be empty")
    if request is not None and not isinstance(request, dict):
        raise ValueError("request must be an object")
    if response is not None and not isinstance(response, dict):
        raise ValueError("response must be an object")
    step = AgentStep(
        index=len(trace.steps),
        phase=str(phase),
        tool_name=str(tool_name),
        request=request or {},
        response=response or {},
        input_tokens=_optional_nonnegative_int(input_tokens, "input_tokens"),
        output_tokens=_optional_nonnegative_int(output_tokens, "output_tokens"),
        latency_ms=_optional_nonnegative_float(latency_ms, "latency_ms"),
        token_method=str(token_method) if token_method else None,
    )
    trace.steps.append(step)
    return step


def complete_agent_trace(
    trace: AgentTrace,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    warnings: Iterable[str] = (),
) -> AgentTrace:
    if trace.status != "running":
        raise ValueError("Agent trace is already finished")
    if status not in {"completed", "failed", "cancelled"}:
        raise ValueError("status must be completed, failed, or cancelled")
    trace.status = status
    trace.result = result or {}
    trace.warnings.extend(str(item) for item in warnings)
    trace.completed_at = _utc_now()
    return trace


def write_agent_trace(trace: AgentTrace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_agent_trace(path: Path) -> AgentTrace:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trace schema: {data.get('schema_version')}")
    trace = AgentTrace(
        trace_id=str(data["trace_id"]),
        scene_id=str(data["scene_id"]),
        query=str(data["query"]),
        query_id=str(data.get("query_id", "")),
        client_name=str(data.get("client_name", "unspecified_agent")),
        model_name=str(data.get("model_name", "unreported")),
        status=str(data.get("status", "running")),
        schema_version=str(data["schema_version"]),
        method_protocol_id=str(data.get("method_protocol_id", METHOD_PROTOCOL_ID)),
        created_at=str(data.get("created_at", _utc_now())),
        completed_at=data.get("completed_at"),
        result=data.get("result"),
        warnings=[str(item) for item in data.get("warnings", [])],
    )
    for item in data.get("steps", []):
        trace.steps.append(
            AgentStep(
                index=int(item["index"]),
                phase=str(item["phase"]),
                tool_name=str(item["tool_name"]),
                request=dict(item.get("request", {})),
                response=dict(item.get("response", {})),
                input_tokens=item.get("input_tokens"),
                output_tokens=item.get("output_tokens"),
                latency_ms=item.get("latency_ms"),
                token_method=item.get("token_method"),
                created_at=str(item.get("created_at", _utc_now())),
            )
        )
    return trace


def trace_accounting(trace: AgentTrace | dict[str, Any]) -> dict[str, Any]:
    steps = trace.steps if isinstance(trace, AgentTrace) else trace.get("steps", [])

    def _value(step: AgentStep | dict[str, Any], key: str) -> Any:
        return getattr(step, key) if isinstance(step, AgentStep) else step.get(key)

    measured_inputs = [
        int(value)
        for step in steps
        if (value := _value(step, "input_tokens")) is not None
    ]
    measured_outputs = [
        int(value)
        for step in steps
        if (value := _value(step, "output_tokens")) is not None
    ]
    latencies = [
        float(value)
        for step in steps
        if (value := _value(step, "latency_ms")) is not None
    ]
    return {
        "step_count": len(steps),
        "model_or_agent_call_count": len(
            [step for step in steps if _value(step, "phase") != "infrastructure"]
        ),
        "input_tokens": sum(measured_inputs) if measured_inputs else None,
        "output_tokens": sum(measured_outputs) if measured_outputs else None,
        "measured_input_step_count": len(measured_inputs),
        "measured_output_step_count": len(measured_outputs),
        "latency_ms": round(sum(latencies), 6) if latencies else None,
    }


def validate_trace_id(trace_id: str) -> str:
    value = str(trace_id)
    if not TRACE_ID_RE.fullmatch(value):
        raise ValueError("Invalid trace_id")
    return value


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _optional_nonnegative_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Semantic encoder must return a 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _rank_scores(scores: np.ndarray, ids: Sequence[str]) -> list[tuple[float, str]]:
    return sorted(
        [(float(score), str(item_id)) for score, item_id in zip(scores, ids)],
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )


def _add_model_context(
    metrics: SearchMetrics,
    encoder: SemanticEncoder,
    texts: Sequence[str],
) -> None:
    metrics.context_chars += sum(len(text) for text in texts)
    tokens = sum(encoder.token_count(text) for text in texts)
    metrics.input_tokens += tokens
    metrics.model_input_tokens += tokens


def _encode_batch(
    metrics: SearchMetrics,
    encoder: SemanticEncoder,
    texts: Sequence[str],
) -> np.ndarray:
    started = time.perf_counter()
    vectors = encoder.encode(texts)
    metrics.model_inference_ms += (time.perf_counter() - started) * 1000
    metrics.model_call_count += 1
    _add_model_context(metrics, encoder, texts)
    return _normalize_rows(vectors)


def _semantic_metrics(
    bundle: dict[str, Any],
    query: str,
    *,
    mode: str,
    query_id: str,
    benchmark_scene_id: str,
    encoder: SemanticEncoder,
) -> SearchMetrics:
    return SearchMetrics(
        mode=mode,
        scene_id=str(bundle["scene_id"]),
        benchmark_scene_id=benchmark_scene_id or str(bundle["scene_id"]),
        query_id=query_id,
        query=query,
        expanded_query=query,
        affordance_keys=[],
        found=False,
        selected_view_id=None,
        selected_node_id=None,
        matched_object=None,
        model_backend=encoder.backend_name,
        model_name=encoder.model_name,
        model_token_method=encoder.token_method,
    )


def _object_text(value: dict[str, Any]) -> str:
    parts = [str(value.get("label", ""))]
    attributes = value.get("attributes")
    if isinstance(attributes, list):
        parts.extend(str(item) for item in attributes)
    for key in ("approx_location", "location_description", "description"):
        if value.get(key):
            parts.append(str(value[key]))
    return " ".join(part for part in parts if part)


def _rank_objects_in_view(
    metrics: SearchMetrics,
    encoder: SemanticEncoder,
    query_vector: np.ndarray,
    view: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], int, float]:
    objects = _view_objects(view)
    if not objects:
        return None, [], 0, 0.0
    object_texts = [_object_text(item) for item in objects]
    before_ms = metrics.model_inference_ms
    object_vectors = _encode_batch(metrics, encoder, object_texts)
    inference_ms = metrics.model_inference_ms - before_ms
    ranked = _rank_scores(
        object_vectors @ query_vector,
        [str(index) for index in range(len(objects))],
    )
    details = [
        {
            "label": str(objects[int(index)].get("label", "")),
            "score": round(score, 6),
        }
        for score, index in ranked
    ]
    label = details[0]["label"] if details else None
    tokens = sum(encoder.token_count(text) for text in object_texts)
    return label, details, tokens, inference_ms


def semantic_flat_search(
    bundle: dict[str, Any],
    query: str,
    encoder: SemanticEncoder,
    *,
    query_id: str = "",
    benchmark_scene_id: str = "",
    minimum_score: float = 0.0,
) -> tuple[SearchMetrics, AgentTrace]:
    """Exhaustive semantic view ranking with the pinned local encoder."""
    started = time.perf_counter()
    metrics = _semantic_metrics(
        bundle,
        query,
        mode="flat_semantic_embedding",
        query_id=query_id,
        benchmark_scene_id=benchmark_scene_id,
        encoder=encoder,
    )
    trace = new_agent_trace(
        scene_id=metrics.scene_id,
        query=query,
        query_id=query_id,
        client_name="local_semantic_runner",
        model_name=encoder.model_name,
    )
    before_query_ms = metrics.model_inference_ms
    query_vector = _encode_batch(metrics, encoder, [query])[0]
    query_inference_ms = metrics.model_inference_ms - before_query_ms
    append_agent_step(
        trace,
        phase="decomposition",
        tool_name="local_semantic_encode_query",
        request={"query": query},
        response={"embedding_dimension": int(query_vector.shape[0])},
        input_tokens=encoder.token_count(query),
        output_tokens=0,
        latency_ms=query_inference_ms,
        token_method=encoder.token_method,
    )
    views: dict[str, dict[str, Any]] = bundle["views"]
    view_ids = sorted(views)
    view_texts = [_view_text(views[view_id]) for view_id in view_ids]
    before_views_ms = metrics.model_inference_ms
    view_vectors = _encode_batch(metrics, encoder, view_texts)
    view_inference_ms = metrics.model_inference_ms - before_views_ms
    ranked = _rank_scores(view_vectors @ query_vector, view_ids)
    metrics.views_checked_ids = view_ids
    metrics.views_checked = len(view_ids)
    for view_id in view_ids:
        regions, objects, landmarks = _count_semantic_items(views[view_id])
        metrics.regions_scanned += regions
        metrics.objects_scanned += objects
        metrics.landmarks_scanned += landmarks
        metrics.semantic_entries_scanned += regions + objects + landmarks
    metrics.all_ranked_view_ids = [view_id for _, view_id in ranked]
    metrics.ranked_view_ids = metrics.all_ranked_view_ids[:3]
    if ranked and ranked[0][0] >= minimum_score:
        metrics.found = True
        metrics.selected_view_id = ranked[0][1]
    append_agent_step(
        trace,
        phase="semantic_ranking",
        tool_name="local_semantic_flat_rank",
        request={"candidate_view_ids": view_ids, "minimum_score": minimum_score},
        response={
            "ranked_view_ids": metrics.all_ranked_view_ids,
            "top_scores": [
                {"view_id": view_id, "score": round(score, 6)}
                for score, view_id in ranked[:5]
            ],
        },
        input_tokens=sum(encoder.token_count(text) for text in view_texts),
        output_tokens=0,
        latency_ms=view_inference_ms,
        token_method=encoder.token_method,
    )
    if metrics.selected_view_id:
        matched, object_ranking, object_tokens, object_inference_ms = _rank_objects_in_view(
            metrics,
            encoder,
            query_vector,
            views[metrics.selected_view_id],
        )
        metrics.matched_object = matched
        append_agent_step(
            trace,
            phase="grounding",
            tool_name="local_semantic_object_rank",
            request={"view_id": metrics.selected_view_id},
            response={"ranked_objects": object_ranking},
            input_tokens=object_tokens,
            output_tokens=0,
            latency_ms=object_inference_ms,
            token_method=encoder.token_method,
        )
    metrics.elapsed_ms = (time.perf_counter() - started) * 1000
    complete_agent_trace(
        trace,
        status="completed",
        result={
            "found": metrics.found,
            "selected_view_id": metrics.selected_view_id,
            "matched_object": metrics.matched_object,
            "ranked_view_ids": metrics.all_ranked_view_ids,
        },
        warnings=["Flat semantic mode scores every indexed view."],
    )
    return metrics, trace


def semantic_graph_search(
    bundle: dict[str, Any],
    query: str,
    encoder: SemanticEncoder,
    *,
    query_id: str = "",
    benchmark_scene_id: str = "",
    keep_margin: float = 0.08,
    minimum_branch_score: float = 0.20,
    max_children: int = 3,
    fallback_child_limit: int = 3,
    minimum_view_score: float = 0.0,
) -> tuple[SearchMetrics, AgentTrace]:
    """Top-down semantic traversal over the captured scene hierarchy."""
    if keep_margin < 0:
        raise ValueError("keep_margin must be non-negative")
    if max_children < 1 or fallback_child_limit < 1:
        raise ValueError("child limits must be at least 1")

    started = time.perf_counter()
    metrics = _semantic_metrics(
        bundle,
        query,
        mode="graph_semantic_embedding",
        query_id=query_id,
        benchmark_scene_id=benchmark_scene_id,
        encoder=encoder,
    )
    trace = new_agent_trace(
        scene_id=metrics.scene_id,
        query=query,
        query_id=query_id,
        client_name="local_semantic_runner",
        model_name=encoder.model_name,
    )
    before_query_ms = metrics.model_inference_ms
    query_vector = _encode_batch(metrics, encoder, [query])[0]
    query_inference_ms = metrics.model_inference_ms - before_query_ms
    append_agent_step(
        trace,
        phase="decomposition",
        tool_name="local_semantic_encode_query",
        request={"query": query},
        response={"embedding_dimension": int(query_vector.shape[0])},
        input_tokens=encoder.token_count(query),
        output_tokens=0,
        latency_ms=query_inference_ms,
        token_method=encoder.token_method,
    )
    tree: dict[str, dict[str, Any]] = bundle["tree"]
    views: dict[str, dict[str, Any]] = bundle["views"]
    stack = [_root_id(bundle["manifest"])]
    visited: set[str] = set()
    scored_views: dict[str, float] = {}

    while stack:
        node_id = stack.pop()
        if node_id in visited or node_id not in tree:
            continue
        visited.add(node_id)
        node = tree[node_id]
        metrics.nodes_visited += 1
        metrics.visited_node_ids.append(node_id)

        if _is_leaf_node(node):
            view_ids = [
                str(view_id)
                for view_id in node.get("view_ids", [])
                if str(view_id) in views and str(view_id) not in scored_views
            ]
            if not view_ids:
                continue
            view_texts = [_view_text(views[view_id]) for view_id in view_ids]
            before_views_ms = metrics.model_inference_ms
            view_vectors = _encode_batch(metrics, encoder, view_texts)
            view_inference_ms = metrics.model_inference_ms - before_views_ms
            ranked_views = _rank_scores(view_vectors @ query_vector, view_ids)
            for score, view_id in ranked_views:
                scored_views[view_id] = score
                metrics.views_checked += 1
                metrics.views_checked_ids.append(view_id)
                regions, objects, landmarks = _count_semantic_items(views[view_id])
                metrics.regions_scanned += regions
                metrics.objects_scanned += objects
                metrics.landmarks_scanned += landmarks
                metrics.semantic_entries_scanned += regions + objects + landmarks
            append_agent_step(
                trace,
                phase="leaf_confirmation",
                tool_name="local_semantic_leaf_rank",
                request={"node_id": node_id, "view_ids": view_ids},
                response={
                    "ranked": [
                        {"view_id": view_id, "score": round(score, 6)}
                        for score, view_id in ranked_views
                    ]
                },
                input_tokens=sum(encoder.token_count(text) for text in view_texts),
                output_tokens=0,
                latency_ms=view_inference_ms,
                token_method=encoder.token_method,
            )
            continue

        children = [
            str(child_id)
            for child_id in node.get("children_ids", [])
            if str(child_id) in tree
        ]
        if not children:
            continue
        child_texts = [_node_text(tree[child_id]) for child_id in children]
        before_children_ms = metrics.model_inference_ms
        child_vectors = _encode_batch(metrics, encoder, child_texts)
        child_inference_ms = metrics.model_inference_ms - before_children_ms
        ranked_children = _rank_scores(child_vectors @ query_vector, children)
        best_score = ranked_children[0][0]
        if best_score < minimum_branch_score:
            chosen = ranked_children[: min(fallback_child_limit, len(ranked_children))]
            reason = "low_confidence_fallback"
        else:
            chosen = [
                item
                for item in ranked_children
                if item[0] >= best_score - keep_margin
            ][:max_children]
            if not chosen:
                chosen = ranked_children[:1]
            reason = "semantic_margin"
        chosen_ids = [child_id for _, child_id in chosen]
        append_agent_step(
            trace,
            phase="traversal",
            tool_name="local_semantic_choose_children",
            request={
                "node_id": node_id,
                "child_ids": children,
                "keep_margin": keep_margin,
                "minimum_branch_score": minimum_branch_score,
                "max_children": max_children,
            },
            response={
                "descend_into": chosen_ids,
                "reason": reason,
                "scores": [
                    {"node_id": child_id, "score": round(score, 6)}
                    for score, child_id in ranked_children
                ],
            },
            input_tokens=sum(encoder.token_count(text) for text in child_texts),
            output_tokens=0,
            latency_ms=child_inference_ms,
            token_method=encoder.token_method,
        )
        for child_id in reversed(chosen_ids):
            stack.append(child_id)

    ranked = sorted(
        [(score, view_id) for view_id, score in scored_views.items()],
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    metrics.all_ranked_view_ids = [view_id for _, view_id in ranked]
    metrics.ranked_view_ids = metrics.all_ranked_view_ids[:3]
    if ranked and ranked[0][0] >= minimum_view_score:
        metrics.found = True
        metrics.selected_view_id = ranked[0][1]
        for node_id in reversed(metrics.visited_node_ids):
            if metrics.selected_view_id in tree[node_id].get("view_ids", []):
                metrics.selected_node_id = node_id
                break
    if metrics.selected_view_id:
        matched, object_ranking, object_tokens, object_inference_ms = _rank_objects_in_view(
            metrics,
            encoder,
            query_vector,
            views[metrics.selected_view_id],
        )
        metrics.matched_object = matched
        append_agent_step(
            trace,
            phase="grounding",
            tool_name="local_semantic_object_rank",
            request={"view_id": metrics.selected_view_id},
            response={"ranked_objects": object_ranking},
            input_tokens=object_tokens,
            output_tokens=0,
            latency_ms=object_inference_ms,
            token_method=encoder.token_method,
        )
    metrics.elapsed_ms = (time.perf_counter() - started) * 1000
    complete_agent_trace(
        trace,
        status="completed",
        result={
            "found": metrics.found,
            "selected_view_id": metrics.selected_view_id,
            "selected_node_id": metrics.selected_node_id,
            "matched_object": metrics.matched_object,
            "ranked_view_ids": metrics.all_ranked_view_ids,
        },
        warnings=[
            "Local semantic mode is a pinned neural embedding traversal, not Cursor Agent or an LLM."
        ],
    )
    return metrics, trace
