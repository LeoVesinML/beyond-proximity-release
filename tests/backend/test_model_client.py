import json

import pytest

from backend.query.model_client import (
    CACHE_SCHEMA,
    CachedModelClient,
    ModelCacheError,
    ModelConfigurationError,
    ModelProviderError,
    OpenAIModelClient,
    StubModelClient,
    create_model_client,
)


def _tree():
    return {
        "root": {
            "node_id": "root",
            "type": "root",
            "name": "Root",
            "summary": "Whole scene",
            "view_ids": [],
            "children_ids": ["leaf"],
        },
        "leaf": {
            "node_id": "leaf",
            "type": "leaf",
            "name": "Lounge",
            "summary": "Lounge with a red chair",
            "view_ids": ["v001"],
            "children_ids": [],
        },
    }


def _views():
    return {
        "v001": {
            "view_id": "v001",
            "scene_summary": "A red chair in the lounge.",
            "room_type": "lounge",
            "objects": [{"label": "chair", "bbox_2d": [0.1, 0.2, 0.4, 0.8]}],
            "spatial_relations": [],
            "functional_context": "seating",
        }
    }


def test_stub_client_is_explicit_and_has_no_model_usage():
    client = StubModelClient()
    answer = client.answer_query(
        {"query": "Find the red chair", "query_type": "attribute"},
        _tree(),
        _views(),
    )

    assert client.mode == "stub"
    assert answer["result"]["found"] is True
    assert answer["result"]["selected_view_id"] == "v001"
    assert answer["visited_nodes"] == ["root", "leaf"]
    assert any("not live" in warning.lower() for warning in answer["warnings"])
    assert client.stats_snapshot()["model_call_count"] == 0
    assert client.stats_snapshot()["total_tokens"] is None


def test_stub_client_searches_captured_viewjson_fields():
    client = StubModelClient()
    views = {
        "v001": {
            "view_id": "v001",
            "summary": "Conference room seating.",
            "visible_regions": [
                {"label": "conference table area", "approx_location": "center"}
            ],
            "visible_objects": [
                {
                    "label": "chair",
                    "attributes": ["red"],
                    "approx_location": "left of table",
                    "bbox_2d": None,
                }
            ],
            "landmarks": [],
            "free_text_notes": "",
        }
    }
    tree = {
        "root": {
            "node_id": "root",
            "node_type": "root",
            "name": "Root",
            "summary": "Captured scene",
            "view_ids": ["v001"],
            "children_ids": ["object_v001_001_chair"],
        },
        "object_v001_001_chair": {
            "node_id": "object_v001_001_chair",
            "node_type": "object",
            "name": "chair",
            "summary": "Object observation",
            "view_ids": ["v001"],
            "children_ids": [],
        },
    }

    answer = client.answer_query("Find the red chair", tree, views)

    assert answer["result"]["found"] is True
    assert answer["result"]["matched_object"] == "chair"
    assert answer["result"]["selected_view_id"] == "v001"
    assert answer["visited_nodes"] == ["root", "object_v001_001_chair"]


def test_stub_client_normalizes_column_typo_and_aliases():
    client = StubModelClient()
    views = {
        "v001": {
            "view_id": "v001",
            "summary": "Lobby with marble columns beside the piano.",
            "visible_objects": [
                {"label": "columns", "bbox_2d": None},
            ],
            "landmarks": [],
        }
    }
    tree = {
        "root": {
            "node_id": "root",
            "node_type": "root",
            "name": "Root",
            "summary": "Captured scene",
            "view_ids": ["v001"],
            "children_ids": ["object_v001_001_columns"],
        },
        "object_v001_001_columns": {
            "node_id": "object_v001_001_columns",
            "node_type": "object",
            "name": "columns",
            "summary": "Marble columns",
            "view_ids": ["v001"],
            "children_ids": [],
        },
    }

    answer = client.answer_query("Find a collumn", tree, views)

    assert answer["result"]["found"] is True
    assert answer["result"]["matched_object"] == "columns"
    assert answer["result"]["selected_view_id"] == "v001"


def test_stub_client_prefers_exact_object_label_over_ambiguous_combined_label():
    client = StubModelClient()
    views = {
        "v011": {
            "view_id": "v011",
            "summary": "Lobby with elevator doors or golden wall panels.",
            "visible_objects": [
                {
                    "label": "elevator doors or golden wall panels",
                    "attributes": ["golden"],
                    "approx_location": "left side",
                },
            ],
            "landmarks": [],
        },
        "v013": {
            "view_id": "v013",
            "summary": "Lobby with golden elevator doors.",
            "visible_objects": [
                {
                    "label": "elevator doors",
                    "attributes": ["golden", "double"],
                    "approx_location": "center-left",
                },
            ],
            "landmarks": [],
        },
    }
    tree = {
        "root": {
            "node_id": "root",
            "node_type": "root",
            "name": "Root",
            "summary": "Captured scene",
            "view_ids": [],
            "children_ids": ["object_v011_003_ambiguous", "object_v013_002_elevator_doors"],
        },
        "object_v011_003_ambiguous": {
            "node_id": "object_v011_003_ambiguous",
            "node_type": "object",
            "name": "elevator doors or golden wall panels",
            "summary": "Ambiguous combined observation",
            "view_ids": ["v011"],
            "children_ids": [],
        },
        "object_v013_002_elevator_doors": {
            "node_id": "object_v013_002_elevator_doors",
            "node_type": "object",
            "name": "elevator doors",
            "summary": "Golden double doors",
            "view_ids": ["v013"],
            "children_ids": [],
        },
    }

    answer = client.answer_query(
        {"query": "Find the golden elevator doors", "query_type": "attribute"},
        tree,
        views,
    )

    assert answer["result"]["found"] is True
    assert answer["result"]["matched_object"] == "elevator doors"
    assert answer["result"]["selected_view_id"] == "v013"
    assert answer["result"]["selected_node_id"] == "object_v013_002_elevator_doors"


def test_stub_negative_query_does_not_match_single_word_inside_compound_label():
    client = StubModelClient()
    views = {
        "v001": {
            "view_id": "v001",
            "summary": "Outdoor facade with a planted bed and paving.",
            "visible_objects": [
                {"label": "planted bed", "attributes": ["green"], "approx_location": "foreground"},
            ],
            "landmarks": [],
        }
    }
    tree = {
        "root": {
            "node_id": "root",
            "node_type": "root",
            "name": "Root",
            "summary": "Captured scene",
            "view_ids": [],
            "children_ids": ["object_v001_001_planted_bed"],
        },
        "object_v001_001_planted_bed": {
            "node_id": "object_v001_001_planted_bed",
            "node_type": "object",
            "name": "planted bed",
            "summary": "Landscape planting",
            "view_ids": ["v001"],
            "children_ids": [],
        },
    }

    answer = client.answer_query({"query": "Find bed", "query_type": "negative"}, tree, views)

    assert answer["result"]["found"] is False
    assert answer["result"]["matched_object"] is None
    assert answer["selected_views"] == []


def test_cached_client_replays_only_verified_live_entry(tmp_path):
    query = {"query": "Find the red chair", "query_type": "attribute"}
    payload = {"query": query, "tree": _tree(), "views": _views()}
    response = {
        "structured_plan": {"target": "chair"},
        "visited_nodes": ["root", "leaf"],
        "selected_views": ["v001"],
        "result": {"found": True, "selected_view_id": "v001"},
        "warnings": [],
    }
    CachedModelClient.write_cache_entry(
        tmp_path,
        "answer_query",
        payload,
        response,
        provider="test_provider",
        model="test_model",
        version="1",
        token_usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )

    client = CachedModelClient(tmp_path)
    replayed = client.answer_query(query, _tree(), _views())

    assert replayed == response
    assert client.mode == "cached_live"
    assert client.provider == "test_provider"
    assert client.model == "test_model"
    assert client.stats_snapshot()["cache_hits"] == 1
    assert client.stats_snapshot()["model_call_count"] == 0
    assert client.stats_snapshot()["total_tokens"] == 14


def test_cached_client_rejects_stub_provenance(tmp_path):
    client = CachedModelClient(tmp_path)
    payload = {"query": "x", "tree": {}, "views": {}}
    path = client.cache_path("answer_query", payload)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "cache_schema": CACHE_SCHEMA,
            "source_mode": "stub",
            "method": "answer_query",
            "request_hash": path.stem,
            "response": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ModelCacheError, match="not verified live provenance"):
        client.answer_query("x", {}, {})


def test_live_mode_fails_gracefully_without_credentials(monkeypatch):
    for name in ("SEMANTICSPLAT_PROVIDER", "SEMANTICSPLAT_MODEL", "SEMANTICSPLAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ModelConfigurationError, match="missing environment variables"):
        create_model_client("live")


def _openai_response(answer: dict) -> dict:
    return {
        "id": "resp_fixture",
        "model": "fixture-model",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(answer)}],
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    }


def _live_answer() -> dict:
    return {
        "structured_plan": {"target": "chair", "strategy": "semantic_index"},
        "visited_nodes": ["root", "leaf"],
        "selected_views": ["v001"],
        "checked_view_ids": ["v001"],
        "result": {
            "found": True,
            "matched_object": "chair",
            "selected_node_id": "leaf",
            "selected_view_id": "v001",
            "bbox_2d": None,
            "bbox_3d": None,
            "camera_pose": None,
            "confidence": 0.8,
            "explanation": "The manual semantic index identifies a red chair.",
        },
        "warnings": [],
    }


def test_openai_live_client_caches_raw_response_and_real_usage(tmp_path, monkeypatch):
    client = OpenAIModelClient(api_key="secret-fixture", model="fixture-model", cache_dir=tmp_path)
    raw_response = _openai_response(_live_answer())
    monkeypatch.setattr(client, "_request_provider", lambda payload: raw_response)

    assert client.describe_view("unused.png", {"existing_analysis": _views()["v001"]}) == _views()["v001"]
    assert client.build_tree(_views(), {"existing_tree": _tree()}) == _tree()
    answer = client.answer_query(
        {"query": "Find the red chair", "query_type": "attribute"},
        _tree(),
        _views(),
    )

    payload = {
        "query": {"query": "Find the red chair", "query_type": "attribute"},
        "tree": _tree(),
        "views": _views(),
    }
    cache_path = CachedModelClient(tmp_path).cache_path("answer_query", payload)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert answer["result"]["found"] is True
    assert answer["result"]["selected_view_id"] == "v001"
    assert client.stats_snapshot()["model_call_count"] == 1
    assert client.stats_snapshot()["total_tokens"] == 30
    assert envelope["source_mode"] == "live"
    assert envelope["raw_response"] == raw_response
    assert "secret-fixture" not in cache_path.read_text(encoding="utf-8")


def test_openai_live_client_retries_transport_failure(tmp_path, monkeypatch):
    client = OpenAIModelClient(api_key="secret", model="fixture-model", cache_dir=tmp_path)
    responses = iter([ModelProviderError("temporary"), _openai_response(_live_answer())])

    def request(_payload):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(client, "_request_provider", request)
    monkeypatch.setattr("backend.query.model_client.time.sleep", lambda _seconds: None)
    client.answer_query("Find the chair", _tree(), _views())

    assert client.stats_snapshot()["model_call_count"] == 2
    assert client.stats_snapshot()["retry_count"] == 1
    assert client.stats_snapshot()["failure_count"] == 0


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5.5", "o1", "o3-mini", "o4-mini"])
def test_reasoning_models_omit_unsupported_sampling_parameters(tmp_path, monkeypatch, model):
    client = OpenAIModelClient(api_key="secret", model=model, cache_dir=tmp_path)
    captured = {}

    def request(payload):
        captured.update(payload)
        return _openai_response(_live_answer())

    monkeypatch.setattr(client, "_request_provider", request)
    client.answer_query("Find the chair", _tree(), _views())

    assert captured["model"] == model
    assert captured["input"]
    for parameter in OpenAIModelClient.unsupported_reasoning_sampling_parameters:
        assert parameter not in captured


def test_older_models_keep_zero_temperature(tmp_path, monkeypatch):
    client = OpenAIModelClient(api_key="secret", model="gpt-4.1-mini", cache_dir=tmp_path)
    captured = {}

    def request(payload):
        captured.update(payload)
        return _openai_response(_live_answer())

    monkeypatch.setattr(client, "_request_provider", request)
    client.answer_query("Find the chair", _tree(), _views())

    assert captured["temperature"] == 0.0


def test_non_retryable_provider_error_stops_immediately(tmp_path, monkeypatch):
    client = OpenAIModelClient(api_key="secret", model="gpt-5.5", cache_dir=tmp_path)
    calls = 0

    def request(_payload):
        nonlocal calls
        calls += 1
        raise ModelProviderError("bad request", retryable=False)

    monkeypatch.setattr(client, "_request_provider", request)
    with pytest.raises(ModelProviderError, match="bad request"):
        client.answer_query("Find the chair", _tree(), _views())

    assert calls == 1
    assert client.stats_snapshot()["retry_count"] == 0
    assert client.stats_snapshot()["failure_count"] == 1


def test_live_mode_builds_openai_adapter_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTICSPLAT_PROVIDER", "openai")
    monkeypatch.setenv("SEMANTICSPLAT_MODEL", "fixture-model")
    monkeypatch.setenv("SEMANTICSPLAT_API_KEY", "secret")

    client = create_model_client("live", cache_dir=tmp_path)

    assert isinstance(client, OpenAIModelClient)
    assert client.provenance() == {
        "mode": "live",
        "provider": "openai",
        "model": "fixture-model",
        "version": "responses_api_v1",
        "temperature": 0.0,
    }


def test_live_mode_rejects_unsupported_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTICSPLAT_PROVIDER", "unknown")
    monkeypatch.setenv("SEMANTICSPLAT_MODEL", "fixture-model")
    monkeypatch.setenv("SEMANTICSPLAT_API_KEY", "secret")

    with pytest.raises(ModelConfigurationError, match="supported providers: openai"):
        create_model_client("live", cache_dir=tmp_path)


def test_nonzero_temperature_is_rejected():
    with pytest.raises(ModelConfigurationError, match="temperature=0"):
        create_model_client("stub", temperature=0.2)
