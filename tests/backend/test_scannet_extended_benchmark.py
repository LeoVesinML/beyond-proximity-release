import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "docs" / "benchmarks" / "scannet_extended_grounding_v2.json"


def _queries():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    return data, data["queries"]


def test_extended_scannet_benchmark_has_declared_unique_coverage():
    data, queries = _queries()

    assert data["query_count"] == 64
    assert data["base_query_count"] == 48
    assert data["extension_query_count"] == 16
    assert len(queries) == 64
    assert len({query["query_id"] for query in queries}) == 64


def test_extended_scannet_functional_queries_have_official_object_gt():
    _, queries = _queries()
    functional = [
        query for query in queries if query.get("query_type") == "functional"
    ]

    assert len(functional) == 8
    for query in functional:
        assert query["verification_status"] == (
            "verified_official_scannet_gt_functional_mapping"
        )
        assert query["expected_object_ids"]
        assert query["expected_bboxes_3d"]
        assert set(query["expected_object_ids"]) == {
            str(box["object_id"]) for box in query["expected_bboxes_3d"]
        }


def test_extended_scannet_negative_queries_have_verified_absence_gt():
    _, queries = _queries()
    negatives = [
        query for query in queries if query.get("query_type") == "negative"
    ]

    assert len(negatives) == 8
    for query in negatives:
        assert query["negative_gt_verified"] is True
        assert query["verification_status"] == (
            "verified_official_scannet_gt_absence"
        )
        assert query["expected_output_type"] == "not_found"
        assert query["expected_object_ids"] == []
        assert query["expected_bboxes_3d"] == []
