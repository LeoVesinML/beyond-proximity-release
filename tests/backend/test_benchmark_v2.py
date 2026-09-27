import json
from collections import Counter
from pathlib import Path


def test_benchmark_v2_is_balanced_and_verified_from_viewjson():
    root = Path(__file__).resolve().parents[2]
    benchmark_path = root / "docs" / "benchmarks" / "benchmark_queries_v2.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    queries = benchmark["queries"]

    assert benchmark["benchmark_id"] == "five_capture_manual_semantic_index_v2"
    assert len(queries) == 150
    assert Counter(query["query_type"] for query in queries) == {
        "simple_object": 25,
        "attribute": 25,
        "relational": 25,
        "multi_hop": 25,
        "functional": 25,
        "negative": 25,
    }
    assert Counter(query["scene_id"] for query in queries) == {
        "ConferenceHall": 30,
        "Museume": 30,
        "Theater": 30,
        "outdoor-street": 30,
        "outdoor-drone": 30,
    }
    assert benchmark["sprint_plan_bucket_status"]["intent"]["status"] == "executable"
    assert benchmark["sprint_plan_bucket_status"]["intent"]["benchmark_query_type"] == "functional"
    assert benchmark["sprint_plan_bucket_status"]["intent"]["count"] == 25
    assert benchmark["sprint_plan_bucket_status"]["freshness"]["status"] == "blocked_no_temporal_gt"
    assert all(query["verification_status"] == "verified_from_view_json" for query in queries)
    assert all(query["verification_source"] for query in queries)


def test_benchmark_v2_positive_and_negative_gt_fields_are_scoped():
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads(
        (root / "docs" / "benchmarks" / "benchmark_queries_v2.json").read_text(
            encoding="utf-8"
        )
    )

    positives = [query for query in benchmark["queries"] if query["query_type"] != "negative"]
    negatives = [query for query in benchmark["queries"] if query["query_type"] == "negative"]

    assert positives
    assert negatives
    assert all(query["expected_view_ids"] for query in positives)
    assert all(query["expected_node_ids"] for query in positives)
    assert all(query["expected_zone_ids"] for query in positives)
    assert all(query["expected_zone_labels"] for query in positives)
    assert all(query["expected_object_labels"] for query in positives)
    assert all(query["expected_output_type"] == "not_found" for query in negatives)
    assert all(query["expected_view_ids"] == [] for query in negatives)
    assert all(query["expected_node_ids"] == [] for query in negatives)
    assert all(query["expected_zone_ids"] == [] for query in negatives)


def test_benchmark_v2_manual_zone_gt_covers_all_positive_queries():
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads(
        (root / "docs" / "benchmarks" / "benchmark_queries_v2.json").read_text(
            encoding="utf-8"
        )
    )
    zone_gt = json.loads(
        (root / "docs" / "benchmarks" / "manual_zone_gt_v2.json").read_text(
            encoding="utf-8"
        )
    )

    positives = [query for query in benchmark["queries"] if query["query_type"] != "negative"]
    negatives = [query for query in benchmark["queries"] if query["query_type"] == "negative"]
    zone_ids = {
        zone["zone_id"]
        for scene in zone_gt["scenes"]
        for zone in scene["zones"]
    }

    assert zone_gt["status"] == "manual_zone_reference_tree_native"
    assert zone_gt["positive_queries_with_expected_zones"] == 125
    assert len(positives) == 125
    assert all(set(query["expected_zone_ids"]).issubset(zone_ids) for query in positives)
    assert all(query["expected_zone_ids"] == [] for query in negatives)


def test_benchmark_v2_manual_bbox_gt_covers_all_positive_queries():
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads(
        (root / "docs" / "benchmarks" / "benchmark_queries_v2.json").read_text(
            encoding="utf-8"
        )
    )
    bbox_gt = json.loads(
        (root / "docs" / "benchmarks" / "manual_bbox_gt_v2.json").read_text(
            encoding="utf-8"
        )
    )

    positives = [query for query in benchmark["queries"] if query["query_type"] != "negative"]
    negatives = [query for query in benchmark["queries"] if query["query_type"] == "negative"]

    assert bbox_gt["counts"]["positive_queries"] == 125
    assert bbox_gt["counts"]["query_records_with_bbox_gt"] == 125
    assert bbox_gt["counts"]["negative_queries_not_applicable"] == 25
    assert len(positives) == 125
    assert all(query["gt_3d_reliable"] is True for query in positives)
    assert all(isinstance(query["expected_bbox_3d"], dict) for query in positives)
    assert all(query["expected_bbox_2d_by_view"] for query in positives)
    assert all(query["gt_3d_reliable"] is False for query in negatives)
    assert all(query["expected_bbox_3d"] is None for query in negatives)
    assert all(query["expected_bbox_2d_by_view"] == {} for query in negatives)


def test_scene_local_manual_bbox_gt_copies_exist():
    root = Path(__file__).resolve().parents[2]
    scene_ids = [
        "ConferenceHall-capture-pilot",
        "Museume-capture",
        "Theater-capture",
        "outdoor-street-capture",
        "outdoor-drone-capture",
    ]
    for scene_id in scene_ids:
        path = (
            root
            / "backend"
            / "data"
            / "scenes"
            / scene_id
            / "ground_truth"
            / "manual_bbox_gt_v2.json"
        )
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["scene_id"] == scene_id
        assert data["query_count"] == 25


def test_person2_required_report_files_exist():
    root = Path(__file__).resolve().parents[2]
    required = [
        "person2_coverage_table_v2.md",
        "benchmark_status_v2.md",
        "gt_coverage_report.md",
        "ambiguous_and_negative_review_v2.md",
        "benchmark_v2_changelog.md",
        "final_benchmark_lock_v2.md",
        "benchmark_v2_execution_report.md",
    ]
    for name in required:
        path = root / "docs" / "benchmarks" / name
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip()


def test_captured_trees_have_tree_native_manual_zones():
    root = Path(__file__).resolve().parents[2]
    scene_ids = [
        "ConferenceHall-capture-pilot",
        "Museume-capture",
        "Theater-capture",
        "outdoor-street-capture",
        "outdoor-drone-capture",
    ]
    for scene_id in scene_ids:
        tree_dir = root / "backend" / "data" / "scenes" / scene_id / "tree"
        manifest = json.loads((tree_dir / "manifest.json").read_text(encoding="utf-8"))
        root_node = json.loads((tree_dir / "node_root.json").read_text(encoding="utf-8"))
        zone_ids = [
            node_id
            for node_id in root_node["children_ids"]
            if str(node_id).startswith("manual_zone_")
        ]

        assert manifest["tree_native_zone_nodes"] is True
        assert manifest["manual_zone_node_count"] == len(zone_ids)
        assert zone_ids
        child_counts = []
        for zone_id in zone_ids:
            zone = json.loads((tree_dir / f"node_{zone_id}.json").read_text(encoding="utf-8"))
            assert zone["node_type"] == "zone"
            child_counts.append(len(zone["children_ids"]))
        assert sum(child_counts) > 0


def test_graph_vs_flat_v2_preserves_quality_with_lower_cost():
    root = Path(__file__).resolve().parents[2]
    metrics_path = (
        root
        / "outputs"
        / "week3"
        / "graph_vs_flat_benchmark_v2"
        / "graph_vs_flat_metrics.json"
    )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    graph = summary["graph"]
    flat = summary["flat"]

    assert data["query_count"] == 150
    assert graph["semantic_entries_scanned"] < flat["semantic_entries_scanned"]
    assert graph["context_size_chars"] < flat["context_size_chars"]
    assert graph["estimated_input_tokens"] < flat["estimated_input_tokens"]
    assert graph["latency_ms"] < flat["latency_ms"]
    for metric in (
        "retrieval_success",
        "expected_view_hit",
        "expected_node_hit",
        "expected_zone_hit",
        "expected_object_hit",
        "not_found_correctness",
    ):
        assert graph[metric] == flat[metric]
