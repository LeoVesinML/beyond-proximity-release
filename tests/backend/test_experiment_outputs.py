import json
from pathlib import Path

import numpy as np

from backend.query.model_client import OpenAIModelClient
from scripts.run_experiment import canonical_result, run_experiment


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _prepare_scene(scene_root: Path) -> None:
    scene = scene_root / "fixture_scene"
    _write_json(scene / "transforms.json", {"frames": [{"view_id": "v001", "file_path": "images/v001.png"}]})
    _write_json(scene / "views" / "v001.json", {
        "view_id": "v001",
        "scene_summary": "A red chair in a small lounge.",
        "room_type": "lounge",
        "lighting": "bright",
        "objects": [{"object_id": "chair-1", "label": "chair", "confidence": 1.0, "bbox_2d": [0.1, 0.2, 0.4, 0.8], "attributes": {"color": "red"}}],
        "spatial_relations": [],
        "functional_context": "seating",
        "facing": "wall",
        "visible_landmarks": [],
    })
    _write_json(scene / "tree" / "node_root.json", {
        "node_id": "root",
        "type": "root",
        "name": "Fixture root",
        "summary": "Whole fixture scene",
        "view_ids": [],
        "children_ids": ["leaf_lounge"],
    })
    _write_json(scene / "tree" / "node_leaf_lounge.json", {
        "node_id": "leaf_lounge",
        "type": "leaf",
        "name": "Lounge",
        "summary": "Small lounge with a red chair",
        "view_ids": ["v001"],
        "children_ids": [],
    })


def _prepare_benchmark(path: Path) -> None:
    _write_json(path, {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "fixture_v1",
        "queries": [
            {
                "query_id": "q001",
                "scene_id": "fixture_scene",
                "query": "Find the red chair",
                "query_type": "attribute",
                "expected_output_type": "object",
                "expected_zone_ids": [],
                "expected_node_ids": ["leaf_lounge"],
                "expected_view_ids": ["v001"],
                "expected_object_labels": ["chair"],
                "acceptance_rule": "found=true in v001",
                "notes": "fixture",
                "verification_status": "verified_manual",
                "verification_source": ["fixture"],
            },
            {
                "query_id": "q002",
                "scene_id": "fixture_scene",
                "query": "Find a bicycle",
                "query_type": "negative",
                "expected_output_type": "not_found",
                "expected_zone_ids": [],
                "expected_node_ids": [],
                "expected_view_ids": [],
                "expected_object_labels": ["bicycle"],
                "acceptance_rule": "found=false",
                "notes": "fixture",
                "verification_status": "verified_manual",
                "verification_source": ["fixture"],
            },
        ],
    })


def test_stub_experiment_writes_canonical_outputs_and_resumes(tmp_path):
    scene_root = tmp_path / "scenes"
    benchmark = tmp_path / "benchmark.json"
    output_root = tmp_path / "outputs"
    _prepare_scene(scene_root)
    _prepare_benchmark(benchmark)
    config = tmp_path / "config.yaml"
    _write_json(config, {
        "week": "week2",
        "run_id": "fixture_run",
        "method": "semantic_splat",
        "mode": "stub",
        "scene_root": str(scene_root),
        "scene_ids": ["fixture_scene"],
        "benchmark_path": str(benchmark),
        "output_root": str(output_root),
        "resume": True,
        "view_selection": {"strategy": "existing"},
        "model": {"temperature": 0.0},
    })

    run_dir = run_experiment(config)
    first = json.loads((run_dir / "query_results" / "q001.json").read_text(encoding="utf-8"))
    negative = json.loads((run_dir / "query_results" / "q002.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))

    assert first["schema_version"] == "semanticsplat.query_result.v1"
    assert first["mode"] == "stub"
    assert first["result"]["found"] is True
    assert first["result"]["selected_object_id"] == "chair-1"
    assert first["result"]["selected_view_id"] == "v001"
    assert first["result"]["bbox_availability_status"] == "not_eligible"
    assert first["result"]["bbox_unavailable_reason"] == "geometry_evaluation_not_eligible"
    assert first["traversal_path"] == ["root", "leaf_lounge"]
    assert first["checked_node_ids"]
    assert first["checked_view_ids"] == ["v001"]
    assert first["checked_object_ids"] == ["chair-1"]
    assert first["search_variant"] == "deterministic_lexical_v1"
    assert first["fallback_used"] is True
    assert first["fallback_reason"]
    assert first["metrics_log"]["token_usage"] is None
    assert first["metrics_log"]["estimated_input_tokens"] > 0
    assert first["metrics_log"]["context_size_chars"] > 0
    assert first["metrics_log"]["model_call_count"] == 0
    assert first["metrics_log"]["checked_object_count"] == 1
    assert first["metrics_log"]["construction_cost"] is None
    assert any("not live" in warning.lower() for warning in first["warnings"])
    assert negative["result"]["found"] is False
    assert negative["result"]["bbox_availability_status"] == "not_applicable"
    assert negative["result"]["bbox_unavailable_reason"] == "negative_query"
    assert metrics["coverage"]["result_count"] == 2
    assert metrics["mode_counts"] == {"stub": 2}
    assert metrics["metrics"]["estimated_input_tokens"]["status"] == "measured"
    assert metrics["metrics"]["bbox_3d_iou"]["status"] == "N/A"
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["construction_cost"]["runtime_seconds"] >= 0

    run_experiment(config)
    logs = json.loads((run_dir / "logs.json").read_text(encoding="utf-8"))
    resumed = [event for event in logs["events"] if event["status"] == "resumed"]
    assert {event["stage"] for event in resumed} == {"query:q001", "query:q002"}


def test_no_stub_result_is_marked_live(tmp_path):
    scene_root = tmp_path / "scenes"
    benchmark = tmp_path / "benchmark.json"
    output_root = tmp_path / "outputs"
    _prepare_scene(scene_root)
    _prepare_benchmark(benchmark)
    config = tmp_path / "config.yaml"
    _write_json(config, {
        "run_id": "mode_check",
        "method": "semantic_splat",
        "mode": "stub",
        "scene_root": str(scene_root),
        "scene_ids": ["fixture_scene"],
        "benchmark_path": str(benchmark),
        "output_root": str(output_root),
        "resume": True,
        "query_limit": 1,
        "view_selection": {"strategy": "existing"},
        "model": {"temperature": 0.0},
    })

    run_dir = run_experiment(config)
    result = json.loads(next((run_dir / "query_results").glob("*.json")).read_text(encoding="utf-8"))
    assert result["mode"] == "stub"
    assert result["metrics_log"]["cache_hits"] == 0
    assert result["metrics_log"]["model_call_count"] == 0


def test_query_limit_and_output_directory_overrides(tmp_path):
    scene_root = tmp_path / "scenes"
    benchmark = tmp_path / "benchmark.json"
    configured_output = tmp_path / "configured_outputs"
    override_output = tmp_path / "overridden" / "live_gate_shape"
    _prepare_scene(scene_root)
    _prepare_benchmark(benchmark)
    config = tmp_path / "config.yaml"
    _write_json(config, {
        "run_id": "configured_run",
        "method": "semantic_splat",
        "mode": "stub",
        "scene_root": str(scene_root),
        "scene_ids": ["fixture_scene"],
        "benchmark_path": str(benchmark),
        "output_root": str(configured_output),
        "resume": True,
        "view_selection": {"strategy": "existing"},
        "model": {"temperature": 0.0},
    })

    run_dir = run_experiment(
        config,
        query_limit_override=1,
        output_override=override_output,
    )
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    results = list((run_dir / "query_results").glob("*.json"))

    assert run_dir == override_output
    assert run_config["run_id"] == "live_gate_shape"
    assert run_config["query_limit"] == 1
    assert run_config["output_dir"] == str(override_output)
    assert [path.name for path in results] == ["q001.json"]


def test_live_experiment_records_provider_calls_usage_and_cache(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    benchmark = tmp_path / "benchmark.json"
    output_root = tmp_path / "outputs"
    _prepare_scene(scene_root)
    _prepare_benchmark(benchmark)
    config = tmp_path / "config.yaml"
    _write_json(config, {
        "run_id": "live_fixture",
        "method": "semantic_splat",
        "mode": "live",
        "scene_root": str(scene_root),
        "scene_ids": ["fixture_scene"],
        "benchmark_path": str(benchmark),
        "output_root": str(output_root),
        "resume": True,
        "view_selection": {"strategy": "existing"},
        "model": {"temperature": 0.0},
    })
    answer = {
        "structured_plan": {"target": "chair"},
        "visited_nodes": ["root", "leaf_lounge"],
        "selected_views": ["v001"],
        "checked_view_ids": ["v001"],
        "result": {
            "found": True,
            "matched_object": "chair",
            "selected_node_id": "leaf_lounge",
            "selected_view_id": "v001",
            "bbox_2d": None,
            "bbox_3d": None,
            "camera_pose": None,
            "confidence": 0.9,
            "explanation": "Fixture provider response.",
        },
        "warnings": [],
    }
    raw_response = {
        "id": "resp_fixture",
        "model": "fixture-model",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }
    monkeypatch.setenv("SEMANTICSPLAT_PROVIDER", "openai")
    monkeypatch.setenv("SEMANTICSPLAT_MODEL", "fixture-model")
    monkeypatch.setenv("SEMANTICSPLAT_API_KEY", "secret-fixture")
    monkeypatch.setattr(OpenAIModelClient, "_request_provider", lambda self, payload: raw_response)

    run_dir = run_experiment(config, mode_override="live", query_limit_override=1)
    result = json.loads((run_dir / "query_results" / "q001.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cache_files = list((run_dir / "model_cache" / "answer_query").glob("*.json"))

    assert result["mode"] == "live"
    assert result["metrics_log"]["model_call_count"] == 1
    assert result["metrics_log"]["token_usage"]["total_tokens"] == 20
    assert run_config["model"]["provider"] == "openai"
    assert run_config["model"]["model"] == "fixture-model"
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text(encoding="utf-8"))["source_mode"] == "live"
    assert "secret-fixture" not in cache_files[0].read_text(encoding="utf-8")


def test_canonical_bbox_quality_gate_marks_eligible_missing_and_invalid_boxes_unavailable():
    common = {
        "run_id": "bbox_gate",
        "method": "semantic_splat",
        "mode": "stub",
        "query": {
            "scene_id": "fixture_scene",
            "query_id": "q001",
            "query": "Find the chair",
            "query_type": "simple_object",
            "expected_output_type": "object_3d_bbox",
        },
        "runtime_seconds": 0.01,
        "usage": {},
        "context_metrics": {},
        "depth_validation": {"status": "reliable"},
        "geometry_eval_allowed": True,
    }
    answer = {
        "result": {
            "found": True,
            "selected_object_id": "chair-1",
            "bbox_3d": None,
            "confidence": 1.0,
        }
    }
    missing = canonical_result(answer=answer, **common)
    assert missing["availability_status"] == "unavailable"
    assert missing["availability_reason"] == "bbox_3d_missing"
    assert missing["result"]["bbox_3d"] is None
    assert missing["result"]["bbox_availability_status"] == "unavailable"
    assert missing["result"]["bbox_unavailable_reason"] == "bbox_3d_missing"

    answer["result"]["bbox_3d"] = {
        "center": [0.0, float("nan"), 1.0],
        "size": [1.0, 1.0, 1.0],
        "object_id": "chair-1",
    }
    invalid = canonical_result(answer=answer, **common)
    assert invalid["availability_status"] == "unavailable"
    assert invalid["result"]["bbox_3d"] is None
    assert invalid["result"]["bbox_availability_status"] == "unavailable"
    assert invalid["result"]["bbox_unavailable_reason"] == "bbox_3d_values_non_finite"

    answer["result"]["bbox_3d"] = {
        "center": [0.0, 0.0, 1.0],
        "size": [1.0, 0.0, 1.0],
        "object_id": "chair-1",
    }
    degenerate = canonical_result(answer=answer, **common)
    assert degenerate["availability_status"] == "unavailable"
    assert degenerate["result"]["bbox_3d"] is None
    assert degenerate["result"]["bbox_unavailable_reason"] == "bbox_3d_extents_non_positive"

    answer["result"]["bbox_3d"] = {
        "center": [0.0, 0.0, 1.0],
        "dimensions": [1.0, 2.0, 3.0],
        "object_id": "chair-1",
    }
    available = canonical_result(answer=answer, **common)
    assert available["availability_status"] == "available"
    assert available["result"]["bbox_3d"]["dimensions"] == [1.0, 2.0, 3.0]
    assert available["result"]["bbox_availability_status"] == "available"
    assert available["result"]["bbox_unavailable_reason"] is None


def test_structured_ineligible_scene_batch_is_logged_and_skipped(tmp_path):
    scene_root = tmp_path / "scenes"
    scene = scene_root / "fixture_scene"
    scene.mkdir(parents=True)
    benchmark = tmp_path / "benchmark.json"
    output_root = tmp_path / "outputs"
    validation_report = tmp_path / "fixture_geometry.json"
    _prepare_benchmark(benchmark)
    _write_json(validation_report, {
        "scene_id": "fixture_scene",
        "semantic_evaluation_allowed": False,
        "three_d_localization_allowed": False,
    })
    config = tmp_path / "config.yaml"
    _write_json(config, {
        "week": "week3",
        "run_id": "ineligible_batch",
        "method": "semantic_splat",
        "mode": "cached_live",
        "scene_root": str(scene_root),
        "scene_ids": ["fixture_scene"],
        "scenes": [{
            "scene_id": "fixture_scene",
            "path": str(scene),
            "semantic_eval_allowed": False,
            "geometry_eval_allowed": False,
            "validation_report": str(validation_report),
        }],
        "benchmark_path": str(benchmark),
        "output_root": str(output_root),
        "resume": True,
        "view_selection": {"strategy": "existing"},
        "model": {"temperature": 0.0},
    })

    run_dir = run_experiment(config, mode_override="stub")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "query_results").glob("*.json")]

    assert run_config["status"] == "complete"
    assert run_config["mode"] == "stub"
    assert run_config["scene_ids"] == ["fixture_scene"]
    assert run_config["scene_status"]["fixture_scene"]["status"] == "skipped_ineligible"
    assert run_config["scene_status"]["fixture_scene"]["candidate_view_count"] == 0
    assert metrics["coverage"]["result_count"] == 2
    assert metrics["coverage"]["schema_valid_result_count"] == 2
    assert metrics["coverage"]["unavailable_result_count"] == 2
    assert len(results) == 2
    assert all(result["mode"] == "stub" for result in results)
    assert all(result["availability_status"] == "unavailable" for result in results)
    assert all(result["availability_reason"] for result in results)
    assert all(result["metrics_log"]["model_call_count"] == 0 for result in results)
    assert any("semantic_eval_allowed=false" in warning for warning in run_config["warnings"])


def test_captured_scene_schema_and_benchmark_alias_are_loaded(tmp_path):
    scene_root = tmp_path / "scenes"
    scene = scene_root / "fixture-capture"
    benchmark = tmp_path / "benchmark.json"
    output_root = tmp_path / "outputs"
    validation_report = tmp_path / "fixture_semantic.json"
    _prepare_benchmark(benchmark)
    _write_json(
        scene / "transforms.json",
        {
            "frames": [
                {
                    "view_id": "v001",
                    "file_path": "images/v001.png",
                    "depth_file_path": "depths/v001_depth.npy",
                }
            ]
        },
    )
    (scene / "images").mkdir(parents=True, exist_ok=True)
    (scene / "images" / "v001.png").write_bytes(b"captured-rgb")
    (scene / "depths").mkdir(parents=True, exist_ok=True)
    np.save(scene / "depths" / "v001_depth.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    _write_json(
        scene / "views" / "v001.json",
        {
            "schema_version": "semanticsplat.captured_view_json.v1",
            "view_id": "v001",
            "scene_id": "fixture-capture",
            "summary": "A red chair in a conference area.",
            "visible_regions": [],
            "visible_objects": [
                {
                    "label": "chair",
                    "attributes": ["red"],
                    "approx_location": "center",
                    "bbox_2d": None,
                }
            ],
            "landmarks": [],
            "free_text_notes": "",
        },
    )
    _write_json(
        scene / "tree" / "manifest.json",
        {"schema_version": "semanticsplat.semantic_tree_manifest.v1", "complete": True},
    )
    _write_json(
        scene / "tree" / "node_root.json",
        {
            "node_id": "root",
            "node_type": "root",
            "name": "Root",
            "summary": "Captured root",
            "view_ids": ["v001"],
            "children_ids": ["object_v001_001_chair"],
        },
    )
    _write_json(
        scene / "tree" / "node_object_v001_001_chair.json",
        {
            "node_id": "object_v001_001_chair",
            "node_type": "object",
            "name": "chair",
            "summary": "Red chair",
            "view_ids": ["v001"],
            "children_ids": [],
        },
    )
    _write_json(
        validation_report,
        {
            "scene_id": "fixture-capture",
            "semantic_eval_allowed": True,
            "geometry_eval_allowed": False,
        },
    )
    config = tmp_path / "config.yaml"
    _write_json(
        config,
        {
            "run_id": "captured_alias",
            "method": "semantic_splat",
            "mode": "stub",
            "scene_root": str(scene_root),
            "scene_ids": ["fixture-capture"],
            "scenes": [
                {
                    "scene_id": "fixture-capture",
                    "benchmark_scene_id": "fixture_scene",
                    "path": str(scene),
                    "semantic_eval_allowed": True,
                    "geometry_eval_allowed": False,
                    "validation_report": str(validation_report),
                }
            ],
            "benchmark_path": str(benchmark),
            "output_root": str(output_root),
            "resume": True,
            "query_limit": 1,
            "view_selection": {"strategy": "existing"},
            "model": {"temperature": 0.0},
        },
    )

    run_dir = run_experiment(config)
    result = json.loads((run_dir / "query_results" / "q001.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))

    assert result["scene_id"] == "fixture-capture"
    assert result["result"]["found"] is True
    assert result["result"]["selected_view_id"] == "v001"
    assert run_config["scenes"][0]["benchmark_scene_id"] == "fixture_scene"
    assert run_config["scene_status"]["fixture-capture"]["semantic_index_format"] == "captured_viewjson_v1"
    assert metrics["coverage"]["benchmark_query_count"] == 2
    assert metrics["coverage"]["result_count"] == 1
    assert metrics["coverage"]["missing_result_count"] == 1
    assert metrics["coverage"]["schema_valid_result_count"] == 1
    assert metrics["coverage"]["ground_truth_unavailable_result_count"] == 0
