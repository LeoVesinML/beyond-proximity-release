"""Tests for agent traces and reproducible local semantic traversal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client

from backend.mcp.tools import agent_trace_tools
from backend.mcp.server import mcp
from backend.query.agent_protocol import (
    TRACE_SCHEMA_VERSION,
    load_agent_trace,
    semantic_flat_search,
    semantic_graph_search,
    trace_accounting,
    write_agent_trace,
)


class FakeSemanticEncoder:
    model_name = "test-semantic-encoder"
    backend_name = "test"
    token_method = "test_whitespace_tokens"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            value = text.lower()
            vectors.append(
                [
                    float(any(token in value for token in ("chair", "seat", "sit"))),
                    float(any(token in value for token in ("exit", "door"))),
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)

    def token_count(self, text: str) -> int:
        return len(text.split())


def _bundle() -> dict:
    return {
        "scene_id": "test-scene",
        "manifest": {"root_node_id": "root"},
        "tree": {
            "root": {
                "node_id": "root",
                "node_type": "root",
                "name": "scene",
                "summary": "indoor scene",
                "children_ids": ["chairs", "exit"],
                "view_ids": [],
            },
            "chairs": {
                "node_id": "chairs",
                "node_type": "leaf",
                "name": "seating area",
                "summary": "chairs and places to sit",
                "children_ids": [],
                "view_ids": ["v001"],
            },
            "exit": {
                "node_id": "exit",
                "node_type": "leaf",
                "name": "exit area",
                "summary": "doors and wayfinding",
                "children_ids": [],
                "view_ids": ["v002"],
            },
        },
        "views": {
            "v001": {
                "view_id": "v001",
                "scene_summary": "A white chair provides a place to sit.",
                "objects": [{"label": "chair"}],
                "visible_regions": [],
                "landmarks": [],
            },
            "v002": {
                "view_id": "v002",
                "scene_summary": "A green exit door and sign.",
                "objects": [{"label": "exit door"}],
                "visible_regions": [],
                "landmarks": [],
            },
        },
    }


def test_semantic_graph_prunes_views_with_same_encoder_as_flat():
    encoder = FakeSemanticEncoder()
    flat, flat_trace = semantic_flat_search(
        _bundle(),
        "find a place to sit",
        encoder,
        query_id="q1",
    )
    graph, graph_trace = semantic_graph_search(
        _bundle(),
        "find a place to sit",
        encoder,
        query_id="q1",
        keep_margin=0.01,
        max_children=1,
    )

    assert flat.selected_view_id == "v001"
    assert graph.selected_view_id == "v001"
    assert flat.views_checked == 2
    assert graph.views_checked == 1
    assert graph.model_call_count >= 3
    assert flat_trace.to_dict()["accounting"]["input_tokens"] == flat.model_input_tokens
    assert graph_trace.to_dict()["accounting"]["input_tokens"] == graph.model_input_tokens


def test_agent_trace_round_trip(tmp_path: Path):
    _, trace = semantic_flat_search(
        _bundle(),
        "find the exit",
        FakeSemanticEncoder(),
        query_id="q2",
    )
    path = tmp_path / "trace.json"
    write_agent_trace(trace, path)
    loaded = load_agent_trace(path)

    assert loaded.schema_version == TRACE_SCHEMA_VERSION
    assert loaded.status == "completed"
    assert loaded.result["selected_view_id"] == "v002"
    assert trace_accounting(loaded)["step_count"] == 3


def test_cursor_mcp_trace_tools_persist_complete_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(agent_trace_tools, "TRACE_ROOT", tmp_path)
    started = agent_trace_tools.start_agent_query_trace_tool(
        "test-scene",
        "find the chair",
        model_name="cursor-model-version",
        query_id="q3",
    )
    trace_id = started["trace_id"]
    recorded = agent_trace_tools.record_agent_query_step_tool(
        trace_id,
        "traversal",
        "get_children",
        request={"node_id": "root"},
        response={"descend_into": ["chairs"]},
        input_tokens=21,
        output_tokens=4,
        latency_ms=12.5,
        token_method="agent_reported",
    )
    finished = agent_trace_tools.finish_agent_query_trace_tool(
        trace_id,
        "completed",
        result={"selected_view_id": "v001"},
    )
    loaded = agent_trace_tools.get_agent_query_trace_tool(trace_id)

    assert recorded["recorded_step_index"] == 0
    assert finished["accounting"]["input_tokens"] == 21
    assert finished["accounting"]["output_tokens"] == 4
    assert loaded["result"]["selected_view_id"] == "v001"
    assert loaded["status"] == "completed"


def test_cursor_mcp_trace_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(agent_trace_tools, "TRACE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid trace_id"):
        agent_trace_tools.get_agent_query_trace_tool("../outside")


@pytest.mark.asyncio
async def test_agent_trace_tools_are_callable_over_mcp_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(agent_trace_tools, "TRACE_ROOT", tmp_path)
    async with Client(mcp) as client:
        tool_names = {tool.name for tool in await client.list_tools()}
        assert {
            "start_agent_query_trace",
            "record_agent_query_step",
            "finish_agent_query_trace",
            "get_agent_query_trace",
        }.issubset(tool_names)

        started = await client.call_tool(
            "start_agent_query_trace",
            {
                "scene_id": "test-scene",
                "query": "find a chair",
                "client_name": "mcp_transport_test",
                "model_name": "test-model",
            },
        )
        trace_id = started.data["trace_id"]
        recorded = await client.call_tool(
            "record_agent_query_step",
            {
                "trace_id": trace_id,
                "phase": "traversal",
                "tool_name": "get_children",
                "request": {"node_id": "root"},
                "response": {"descend_into": ["chairs"]},
                "input_tokens": 10,
                "output_tokens": 2,
                "latency_ms": 5.0,
                "token_method": "agent_reported",
            },
        )
        finished = await client.call_tool(
            "finish_agent_query_trace",
            {
                "trace_id": trace_id,
                "status": "completed",
                "result": {"selected_view_id": "v001"},
            },
        )
        loaded = await client.call_tool(
            "get_agent_query_trace",
            {"trace_id": trace_id},
        )

    assert recorded.data["step_count"] == 1
    assert finished.data["accounting"]["input_tokens"] == 10
    assert loaded.data["status"] == "completed"
    assert loaded.data["result"]["selected_view_id"] == "v001"
