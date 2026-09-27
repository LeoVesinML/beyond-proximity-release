import json
from pathlib import Path

from scripts.evaluate_grounding import (
    aggregate_rows,
    evaluate,
    evaluate_query,
    segmentation_metrics,
    write_outputs,
)


def _query(query_id="q1", *, bbox=True, object_id="7", query_type="simple_object"):
    query = {
        "query_id": query_id,
        "scene_id": "scene-a",
        "source_dataset": "fixture",
        "query_type": query_type,
        "verification_status": "verified_official",
        "expected_object_ids": [object_id],
    }
    if bbox:
        query["gt_3d_reliable"] = True
        query["expected_bbox_3d"] = {
            "center": [0, 0, 0],
            "size": [2, 2, 2],
            "object_id": object_id,
        }
    return query


def _result(
    query_id="q1",
    *,
    bbox=None,
    object_id="7",
    found=True,
    relation_correct=None,
    ranked_object_ids=None,
):
    result = {
        "schema_version": "semanticsplat.query_result.v1",
        "run_id": "run",
        "method": "graph",
        "mode": "stub",
        "query_id": query_id,
        "scene_id": "scene-a",
        "query": "fixture",
        "query_type": "simple_object",
        "structured_plan": {},
        "visited_nodes": ["root", "leaf"],
        "selected_views": ["v1"],
        "result": {
            "found": found,
            "selected_node_id": "leaf",
            "object_id": object_id,
            "ranked_object_ids": ranked_object_ids or [object_id],
            "bbox_3d": bbox,
        },
        "metrics_log": {
            "runtime_seconds": 0.2,
            "visited_node_count": 2,
            "checked_view_count": 1,
            "checked_object_count": 4,
            "context_size_chars": 800,
            "estimated_input_tokens": 200,
            "construction_cost": {"seconds": 12},
        },
        "warnings": [],
    }
    if relation_correct is not None:
        result["result"]["relation_correct"] = relation_correct
    return result


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bbox_iou_thresholds_retain_zero_for_missing_prediction():
    exact = evaluate_query(
        _query("exact"),
        _result("exact", bbox={"center": [0, 0, 0], "size": [2, 2, 2]}),
    )
    partial = evaluate_query(
        _query("partial"),
        _result("partial", bbox={"center": [1, 0, 0], "size": [2, 2, 2]}),
    )
    missing = evaluate_query(_query("missing"), _result("missing", bbox=None, found=False))

    metrics = aggregate_rows([exact, partial, missing])

    assert metrics["bbox_iou_3d"]["value"] == 0.444444
    assert metrics["acc_at_0_1"]["value"] == 0.666667
    assert metrics["acc_at_0_25"]["value"] == 0.666667
    assert metrics["acc_at_0_5"]["value"] == 0.333333
    assert metrics["acc_at_0_5"]["denominator"] == 3
    assert missing["bbox_iou_3d"] == 0.0
    assert "bad bbox" in missing["failures"]


def test_recall_at_1_uses_exact_object_id():
    hit = evaluate_query(_query("hit"), _result("hit", bbox=None, object_id="7"))
    miss = evaluate_query(_query("miss"), _result("miss", bbox=None, object_id="chair"))

    metrics = aggregate_rows([hit, miss])

    assert metrics["recall_at_1_exact_object_id"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
        "status": "measured",
    }
    assert "wrong object" in miss["failures"]


def test_recall_at_k_mrr_and_verified_negative_accuracy():
    ranked = evaluate_query(
        _query("ranked"),
        _result(
            "ranked",
            bbox=None,
            object_id="8",
            ranked_object_ids=["8", "9", "7", "10", "11"],
        ),
    )
    negative_query = {
        "query_id": "negative",
        "scene_id": "scene-a",
        "source_dataset": "fixture",
        "query_type": "negative",
        "verification_status": "verified_official_absence",
        "expected_output_type": "not_found",
    }
    negative = evaluate_query(
        negative_query,
        _result("negative", bbox=None, object_id=None, found=False),
    )

    metrics = aggregate_rows([ranked, negative])

    assert metrics["recall_at_1_exact_object_id"]["value"] == 0.0
    assert metrics["recall_at_3_exact_object_id"]["value"] == 1.0
    assert metrics["recall_at_5_exact_object_id"]["value"] == 1.0
    assert metrics["mrr_exact_object_id"]["value"] == 0.333333
    assert metrics["negative_accuracy"]["value"] == 1.0
    assert negative["failures"] == []


def test_failure_aggregation_includes_relation_unavailable_bad_bbox_and_missing_gt(tmp_path):
    queries = [
        _query("wrong-object"),
        _query("wrong-relation", query_type="relational"),
        _query("unavailable"),
        {
            "query_id": "no-gt",
            "scene_id": "scene-a",
            "query_type": "simple_object",
            "verification_status": "missing_gt",
        },
    ]
    benchmark = tmp_path / "benchmark.json"
    results = tmp_path / "results"
    _write(benchmark, {"benchmark_id": "fixture", "queries": queries})
    _write(results / "wrong-object.json", _result("wrong-object", bbox=None, object_id="8"))
    _write(
        results / "wrong-relation.json",
        _result(
            "wrong-relation",
            bbox={"center": [0, 0, 0], "size": [2, 2, 2]},
            relation_correct=False,
        ),
    )
    unavailable = _result("unavailable", bbox=None, found=False)
    unavailable["availability_status"] = "unavailable"
    _write(results / "unavailable.json", unavailable)
    _write(results / "no-gt.json", _result("no-gt", bbox=None))

    summary = evaluate(benchmark, results)

    assert summary["failure_taxonomy"]["wrong object"] == 1
    assert summary["failure_taxonomy"]["wrong relation"] == 1
    assert summary["failure_taxonomy"]["invalid/unavailable prediction"] == 1
    assert summary["failure_taxonomy"]["bad bbox"] == 2
    assert summary["failure_taxonomy"]["missing GT"] == 1


def test_efficiency_and_construction_cost_are_separate(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    results = tmp_path / "results"
    _write(benchmark, {"benchmark_id": "fixture", "queries": [_query()]})
    _write(
        results / "q1.json",
        _result("q1", bbox={"center": [0, 0, 0], "size": [2, 2, 2]}),
    )

    summary = evaluate(benchmark, results)

    efficiency = summary["metrics"]["efficiency"]
    assert efficiency["runtime_seconds"]["mean"] == 0.2
    assert efficiency["checked_nodes"]["total"] == 2
    assert efficiency["checked_views"]["total"] == 1
    assert efficiency["checked_objects"]["total"] == 4
    assert efficiency["context_chars"]["total"] == 800
    assert efficiency["estimated_tokens"]["total"] == 200
    assert "construction_cost" not in efficiency
    assert summary["construction_cost"]["status"] == "reported"
    assert summary["construction_cost"]["per_query_records"] == [{"seconds": 12}]


def test_segmentation_is_explicit_na_without_joint_masks():
    row = evaluate_query(
        _query(),
        _result("q1", bbox={"center": [0, 0, 0], "size": [2, 2, 2]}),
    )

    metrics = segmentation_metrics([row])

    assert metrics["mAcc"]["status"] == "N/A"
    assert metrics["mIoU"]["value"] is None
    assert "not segmentation evidence" in metrics["fmIoU"]["reason"]


def test_segmentation_metrics_are_measured_only_from_arrays():
    query = _query()
    query["segmentation_gt"] = [[0, 0], [1, 1]]
    result = _result("q1", bbox=None)
    result["result"]["segmentation_prediction"] = [[0, 1], [1, 1]]
    row = evaluate_query(query, result)

    metrics = segmentation_metrics([row])

    assert metrics["mAcc"]["value"] == 0.75
    assert metrics["mIoU"]["value"] == 0.583333
    assert metrics["fmIoU"]["value"] == 0.583333
    assert all(record["status"] == "measured" for record in metrics.values())


def test_cli_outputs_include_json_csv_markdown_and_per_query_failures(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    results = tmp_path / "results"
    _write(benchmark, {"benchmark_id": "fixture", "queries": [_query()]})
    _write(results / "q1.json", _result("q1", bbox=None, object_id="8"))
    summary = evaluate(benchmark, results)

    paths = write_outputs(summary, tmp_path / "out")

    assert set(paths) == {"json", "csv", "markdown", "failures"}
    assert all(path.is_file() for path in paths.values())
    failures = json.loads(paths["failures"].read_text(encoding="utf-8"))
    assert failures["per_query_failures"][0]["query_id"] == "q1"
