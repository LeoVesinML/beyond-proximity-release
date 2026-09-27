import json
from pathlib import Path

from scripts.audit_relation_reasoning import audit


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _result(query_id, ranked_ids, *, relation_applied):
    selected = ranked_ids[0]
    return {
        "query_id": query_id,
        "structured_plan": {
            "target": "chair",
            "relation": "left",
            "anchor": "table",
            "anchor_secondary": None,
        },
        "result": {
            "selected_object_id": selected,
            "ranked_object_ids": ranked_ids,
            "score_details": {
                "relation_applied": relation_applied,
                "relation_confidence": 0.9 if relation_applied else 0.4,
                "relation_source": "bbox_3d",
            },
            "relation_audit": {
                "ranking_changed": ranked_ids == ["target", "wrong"],
                "evaluated_candidate_count": 2,
            },
        },
    }


def test_relation_audit_counts_only_gated_top1_changes_as_validated(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    with_dir = tmp_path / "with"
    without_dir = tmp_path / "without"
    query_id = "q1"
    _write(
        benchmark,
        {
            "queries": [
                {
                    "query_id": query_id,
                    "query": "Find the chair left of the table.",
                    "scene_id": "scene",
                    "source_dataset": "test",
                    "query_type": "relational",
                    "expected_object_ids": ["target"],
                }
            ]
        },
    )
    _write(
        with_dir / f"{query_id}.json",
        _result(query_id, ["target", "wrong"], relation_applied=True),
    )
    _write(
        without_dir / f"{query_id}.json",
        _result(query_id, ["wrong", "target"], relation_applied=False),
    )

    report = audit(benchmark, with_dir, without_dir)

    assert report["overall"]["beneficial"] == 1
    assert report["overall"]["harmful"] == 0
    assert report["overall"]["validated_relation_changes"] == 1
    assert report["per_query"][0]["query"] == (
        "Find the chair left of the table."
    )
