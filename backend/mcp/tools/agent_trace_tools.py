"""MCP infrastructure tools for replayable agent query traces."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from backend.query.agent_protocol import (
    append_agent_step,
    complete_agent_trace,
    load_agent_trace,
    new_agent_trace,
    validate_trace_id,
    write_agent_trace,
)


TRACE_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "agent_traces" / "cursor_mcp"


def _trace_path(trace_id: str) -> Path:
    value = validate_trace_id(trace_id)
    return TRACE_ROOT / f"{value}.json"


def start_agent_query_trace_tool(
    scene_id: str,
    query: str,
    client_name: str = "cursor_agent",
    model_name: str = "unreported",
    query_id: str = "",
) -> dict[str, Any]:
    """Start a query trace before Cursor or another agent begins MCP traversal."""
    trace = new_agent_trace(
        scene_id=scene_id,
        query=query,
        query_id=query_id,
        client_name=client_name,
        model_name=model_name,
    )
    path = _trace_path(trace.trace_id)
    write_agent_trace(trace, path)
    return {
        "trace_id": trace.trace_id,
        "schema_version": trace.schema_version,
        "method_protocol_id": trace.method_protocol_id,
        "status": trace.status,
        "trace_path": str(path),
        "instruction": (
            "Pass trace_id to record_agent_query_step after every model decision or MCP call, "
            "then call finish_agent_query_trace exactly once."
        ),
    }


def record_agent_query_step_tool(
    trace_id: str,
    phase: str,
    tool_name: str,
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    token_method: str | None = None,
) -> dict[str, Any]:
    """Append one measured model decision or MCP action to a running trace."""
    path = _trace_path(trace_id)
    if not path.exists():
        return {"error": f"Trace '{trace_id}' not found"}
    trace = load_agent_trace(path)
    step = append_agent_step(
        trace,
        phase=phase,
        tool_name=tool_name,
        request=request,
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        token_method=token_method,
    )
    write_agent_trace(trace, path)
    return {
        "trace_id": trace.trace_id,
        "status": trace.status,
        "recorded_step_index": step.index,
        "step_count": len(trace.steps),
    }


def finish_agent_query_trace_tool(
    trace_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """Finish a trace and persist its result and aggregate accounting."""
    path = _trace_path(trace_id)
    if not path.exists():
        return {"error": f"Trace '{trace_id}' not found"}
    trace = load_agent_trace(path)
    complete_agent_trace(
        trace,
        status=status,
        result=result,
        warnings=warnings,
    )
    write_agent_trace(trace, path)
    data = trace.to_dict()
    return {
        "trace_id": trace.trace_id,
        "status": trace.status,
        "trace_path": str(path),
        "accounting": data["accounting"],
    }


def get_agent_query_trace_tool(trace_id: str) -> dict[str, Any]:
    """Load a trace for replay, audit, or result presentation."""
    path = _trace_path(trace_id)
    if not path.exists():
        return {"error": f"Trace '{trace_id}' not found"}
    return load_agent_trace(path).to_dict()
