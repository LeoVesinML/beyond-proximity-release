import json
from pathlib import Path

from scripts.evaluate_results import evaluate, summary_markdown
from scripts.export_compact_metrics import compact_metrics


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _query(*, with_bbox=False):
    query = {
        "query_id": "q001",
        "scene_id": "scene",
        "query": "Find the chair",
        "query_type": "simple_object",
        "expected_output_type": "object",
        "expected_zone_ids": ["zone"],
        "expected_node_ids": ["leaf"],
        "expected_view_ids": ["v001"],
        "expected_object_labels": ["chair"],
        "acceptance_rule": "found=true in v001",
        "notes": "fixture",
        "verification_status": "verified_manual",
        "verification_source": ["fixture"],
    }
    if with_bbox:
        query["gt_3d_reliable"] = True
        query["expected_bbox_3d"] = {"center": [0, 0, 0], "size": [2, 2, 2]}
    return query


def _result(*, bbox=None):
    return {
        "schema_version": "semanticsplat.query_result.v1",
        "run_id": "run",
        "method": "semantic_splat",
        "mode": "stub",
        "scene_id": "scene",
        "query_id": "q001",
        "query": "Find the chair",
        "query_type": "simple_object",
        "structured_plan": {"target": "chair"},
        "visited_nodes": ["root", "zone", "leaf"],
        "selected_views": ["v001"],
        "result": {
            "found": True,
            "matched_object": "chair",
            "selected_node_id": "leaf",
            "selected_view_id": "v001",
            "bbox_2d": [0.1, 0.1, 0.5, 0.5],
            "bbox_3d": bbox,
            "camera_pose": None,
            "confidence": 1.0,
            "explanation": "fixture",
        },
        "metrics_log": {
            "runtime_seconds": 0.25,
            "stage_runtime_seconds": {"answer_query": 0.25},
            "token_usage": None,
            "model_call_count": 0,
            "cache_hits": 0,
            "retry_count": 0,
            "failure_count": 0,
            "visited_node_count": 3,
            "checked_view_count": 1,
            "context_size_chars": 400,
            "prompt_size_chars": 460,
            "estimated_input_tokens": 115,
            "estimated_token_method": "ceil(prompt_size_chars / 4)",
            "construction_cost": None,
        },
        "warnings": [],
    }


def _prepare(tmp_path, *, depth_status="unreliable", with_bbox=False):
    benchmark = tmp_path / "benchmark.json"
    run_dir = tmp_path / "run"
    _write(benchmark, {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "fixture",
        "queries": [_query(with_bbox=with_bbox)],
    })
    _write(run_dir / "run_config.json", {
        "run_id": "run",
        "mode": "stub",
        "depth_validation": {"status": depth_status},
    })
    return benchmark, run_dir


def test_evaluator_reports_hits_and_unavailable_3d_iou(tmp_path):
    benchmark, run_dir = _prepare(tmp_path)
    _write(run_dir / "query_results" / "q001.json", _result())

    summary = evaluate(benchmark, run_dir)

    assert summary["coverage"]["schema_valid_result_count"] == 1
    assert summary["metrics"]["retrieval_success"]["value"] == 1.0
    assert summary["metrics"]["expected_view_hit"]["value"] == 1.0
    assert summary["metrics"]["expected_node_hit"]["value"] == 1.0
    assert summary["metrics"]["expected_zone_hit"]["value"] == 1.0
    assert summary["metrics"]["runtime_seconds"]["mean"] == 0.25
    assert summary["metrics"]["token_usage"]["total"] is None
    assert summary["metrics"]["estimated_input_tokens"]["total"] == 115
    assert summary["metrics"]["context_size_chars"]["mean"] == 400
    assert summary["metrics"]["bbox_3d_iou"]["status"] == "N/A"
    assert summary["metrics"]["bbox_acc_at_0_25"]["status"] == "N/A"
    assert summary["compact_metrics"]["total_queries"] == 1
    assert summary["compact_metrics"]["gt_eligible_queries"] == 1
    assert summary["compact_metrics"]["per_query_type"]["simple"]["retrieval_success"]["value"] == 1.0


def test_3d_iou_requires_reliable_depth_and_gt(tmp_path):
    benchmark, run_dir = _prepare(tmp_path, depth_status="reliable", with_bbox=True)
    _write(
        run_dir / "query_results" / "q001.json",
        _result(bbox={"center": [0, 0, 0], "size": [2, 2, 2]}),
    )

    summary = evaluate(benchmark, run_dir)

    assert summary["metrics"]["bbox_3d_iou"]["status"] == "measured"
    assert summary["metrics"]["bbox_3d_iou"]["value"] == 1.0
    assert summary["metrics"]["bbox_3d_iou"]["denominator"] == 1
    assert summary["metrics"]["bbox_acc_at_0_1"]["value"] == 1.0
    assert summary["metrics"]["bbox_acc_at_0_1"]["numerator"] == 1
    assert summary["metrics"]["bbox_acc_at_0_25"]["value"] == 1.0
    assert summary["metrics"]["bbox_acc_at_0_5"]["value"] == 1.0


def test_missing_bbox_prediction_counts_as_zero_in_acc_denominator(tmp_path):
    benchmark, run_dir = _prepare(tmp_path, depth_status="reliable", with_bbox=True)
    query = _query(with_bbox=True)
    query["source_dataset"] = "Nr3D"
    _write(benchmark, {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "missing_bbox_fixture",
        "queries": [query],
    })
    result = _result(bbox=None)
    result["result"]["found"] = False
    result["result"]["matched_object"] = None
    _write(run_dir / "query_results" / "q001.json", result)

    summary = evaluate(benchmark, run_dir)

    assert summary["metrics"]["bbox_3d_iou"]["value"] == 0.0
    assert summary["metrics"]["bbox_3d_iou"]["denominator"] == 1
    assert summary["metrics"]["bbox_3d_iou"]["missing_prediction_count"] == 1
    assert summary["metrics"]["bbox_acc_at_0_25"]["value"] == 0.0
    assert summary["metrics"]["bbox_acc_at_0_25"]["denominator"] == 1
    assert summary["per_source_dataset"]["Nr3D"]["bbox_acc_at_0_5"]["value"] == 0.0


def test_declared_no_predicted_bbox_keeps_iou_unavailable(tmp_path):
    benchmark, run_dir = _prepare(tmp_path, depth_status="reliable", with_bbox=True)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["bbox_3d_evaluation"] = {
        "status": "unavailable_no_predicted_box",
        "reason": "Method emits a point, not a 3D extent.",
    }
    _write(config_path, config)
    _write(run_dir / "query_results" / "q001.json", _result(bbox=None))

    summary = evaluate(benchmark, run_dir)

    assert summary["metrics"]["bbox_3d_iou"]["status"] == "N/A"
    assert summary["metrics"]["bbox_3d_iou"]["denominator"] == 0
    assert summary["metrics"]["bbox_acc_at_0_25"]["status"] == "N/A"
    assert any("does not emit predicted 3D boxes" in warning for warning in summary["warnings"])


def test_invalid_schema_is_excluded_from_metrics(tmp_path):
    benchmark, run_dir = _prepare(tmp_path)
    invalid = _result()
    invalid["mode"] = "live"
    invalid["schema_version"] = "legacy"
    del invalid["metrics_log"]["checked_view_count"]
    _write(run_dir / "query_results" / "q001.json", invalid)

    summary = evaluate(benchmark, run_dir)

    assert summary["coverage"]["result_count"] == 1
    assert summary["coverage"]["schema_valid_result_count"] == 0
    assert summary["metrics"]["retrieval_success"]["value"] is None
    assert summary["failure_categories"]["baseline adapter issue"] == 1


def test_evaluator_accepts_explicit_results_directory(tmp_path):
    benchmark, run_dir = _prepare(tmp_path)
    results_dir = tmp_path / "external_results"
    _write(results_dir / "q001.json", _result())

    summary = evaluate(benchmark, run_dir, results_dir=results_dir)

    assert summary["coverage"]["schema_valid_result_count"] == 1
    assert summary["per_query"][0]["result_file"].endswith("q001.json")


def test_missing_gt_is_excluded_from_accuracy_metrics(tmp_path):
    benchmark, run_dir = _prepare(tmp_path)
    query = _query()
    query.update({
        "expected_zone_ids": [],
        "expected_node_ids": [],
        "expected_view_ids": [],
        "expected_object_labels": [],
        "notes": "GT unavailable; excluded from accuracy metrics",
        "verification_status": "missing_gt",
        "verification_source": [],
    })
    _write(benchmark, {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "missing_gt_fixture",
        "queries": [query],
    })
    _write(run_dir / "query_results" / "q001.json", _result())

    summary = evaluate(benchmark, run_dir)

    assert summary["coverage"]["schema_valid_result_count"] == 1
    assert summary["coverage"]["ground_truth_unavailable_result_count"] == 1
    assert summary["metrics"]["retrieval_success"]["status"] == "unavailable"
    assert summary["metrics"]["retrieval_success"]["denominator"] == 0
    assert summary["metrics"]["runtime_seconds"]["status"] == "measured"
    assert summary["per_query"][0]["accuracy_metrics_status"] == "unavailable_gt"


def test_unavailable_result_is_schema_valid_but_not_measured(tmp_path):
    benchmark, run_dir = _prepare(tmp_path)
    result = _result()
    result["availability_status"] = "unavailable"
    result["availability_reason"] = "No semantic views are available."
    result["result"]["found"] = False
    result["result"]["matched_object"] = None
    result["result"]["selected_node_id"] = None
    result["result"]["selected_view_id"] = None
    result["selected_views"] = []
    result["visited_nodes"] = []
    result["metrics_log"]["checked_view_count"] = 0
    result["metrics_log"]["visited_node_count"] = 0
    _write(run_dir / "query_results" / "q001.json", result)

    summary = evaluate(benchmark, run_dir)
    markdown = summary_markdown(summary)

    assert summary["coverage"]["result_count"] == 1
    assert summary["coverage"]["schema_valid_result_count"] == 1
    assert summary["coverage"]["unavailable_result_count"] == 1
    assert summary["coverage"]["evaluated_result_count"] == 0
    assert summary["mode_counts"] == {"stub": 1}
    assert summary["metrics"]["retrieval_success"]["status"] == "unavailable"
    assert summary["metrics"]["bbox_3d_iou"]["status"] == "N/A"
    assert summary["metrics"]["bbox_acc_at_0_5"]["status"] == "N/A"
    assert "| retrieval_success | unavailable |" in markdown
    assert "| bbox_3d_iou | N/A |" in markdown
    assert "| Acc@0.5 | N/A |" in markdown


def test_evaluator_scores_object_ids_and_max_bbox_iou(tmp_path):
    benchmark, run_dir = _prepare(tmp_path, depth_status="reliable", with_bbox=True)
    query = _query(with_bbox=True)
    query["expected_object_id"] = "7"
    query["expected_object_ids"] = ["7", "8"]
    query["expected_bboxes_3d"] = [
        {"center": [10, 10, 10], "size": [1, 1, 1], "object_id": "8"},
        {"center": [0, 0, 0], "size": [2, 2, 2], "object_id": "7"},
    ]
    _write(benchmark, {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "multi_bbox_fixture",
        "queries": [query],
    })
    _write(
        run_dir / "query_results" / "q001.json",
        _result(bbox={"center": [0, 0, 0], "size": [2, 2, 2], "object_id": "7"}),
    )

    summary = evaluate(benchmark, run_dir)

    assert summary["metrics"]["expected_object_id_hit"]["value"] == 1.0
    assert summary["metrics"]["bbox_3d_iou"]["value"] == 1.0
    assert summary["metrics"]["bbox_acc_at_0_5"]["value"] == 1.0


def test_public_metrics_export_excludes_per_query_gt_details(tmp_path):
    benchmark, run_dir = _prepare(tmp_path, depth_status="reliable", with_bbox=True)
    _write(
        run_dir / "query_results" / "q001.json",
        _result(bbox={"center": [0, 0, 0], "size": [2, 2, 2]}),
    )
    summary = evaluate(benchmark, run_dir)

    public = compact_metrics(summary)

    assert public["run_id"] == "run"
    assert public["coverage"]["result_count"] == 1
    assert "per_query" not in public
    assert "missing_query_ids" not in public


def test_repository_benchmark_is_balanced_and_five_scene_gt_is_empty():
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads(
        (root / "docs" / "benchmarks" / "benchmark_queries_v1.json").read_text(
            encoding="utf-8"
        )
    )
    queries = benchmark["queries"]
    five_scenes = {"ConferenceHall", "Museume", "outdoor-drone", "outdoor-street", "Theater"}
    expected_types = {"simple_object", "attribute", "relational", "multi_hop", "functional", "negative"}

    type_counts = {query_type: 0 for query_type in expected_types}
    scene_counts = {scene_id: 0 for scene_id in five_scenes}
    for query in queries:
        type_counts[query["query_type"]] += 1
        if query["scene_id"] not in five_scenes:
            continue
        scene_counts[query["scene_id"]] += 1
        assert query["verification_status"] == "missing_gt"
        assert query["notes"] == "GT unavailable; excluded from accuracy metrics"
        assert query["verification_source"] == []
        for field in ("expected_zone_ids", "expected_node_ids", "expected_view_ids", "expected_object_labels"):
            assert query[field] == []

    assert len(queries) == 90
    assert type_counts == {query_type: 15 for query_type in expected_types}
    assert scene_counts == {scene_id: 8 for scene_id in five_scenes}
