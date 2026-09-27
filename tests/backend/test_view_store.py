import json

from backend.io.view_store import list_view_summaries


def test_list_view_summaries_adapts_captured_viewjson(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    (views / "v001.json").write_text(
        json.dumps(
            {
                "schema_version": "semanticsplat.captured_view_json.v1",
                "view_id": "v001",
                "summary": "A conference hall with banquet seating.",
                "visible_regions": [{"label": "conference hall area"}],
                "visible_objects": [{"label": "chair"}],
                "landmarks": [{"label": "stage", "kind": "landmark"}],
            }
        ),
        encoding="utf-8",
    )

    assert list_view_summaries(str(tmp_path)) == [
        {
            "view_id": "v001",
            "summary": "A conference hall with banquet seating.",
            "object_count": 1,
            "room_type": "ballroom",
            "facing": "unknown",
            "visible_landmarks": ["stage"],
        }
    ]


def test_list_view_summaries_preserves_legacy_shape(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    (views / "v002.json").write_text(
        json.dumps(
            {
                "view_id": "v002",
                "scene_summary": "A lounge.",
                "objects": [{"label": "sofa"}],
                "room_type": "lounge",
                "facing": "window",
                "visible_landmarks": ["piano"],
            }
        ),
        encoding="utf-8",
    )

    summary = list_view_summaries(str(tmp_path))[0]

    assert summary["summary"] == "A lounge."
    assert summary["room_type"] == "lounge"
    assert summary["visible_landmarks"] == ["piano"]
