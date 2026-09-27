import json
from pathlib import Path

import pytest

from backend.query import live_session, log as query_log
from backend.query.log import QuerySession
from backend.io.captured_semantic_index import load_captured_scene_index
from scripts.build_scene_tree_from_viewjson import build_scene_tree
from scripts.create_manual_viewjson_templates import create_templates


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def prepare_scene(data_dir: Path, *, annotate: bool) -> Path:
    scene = data_dir / "captured"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir(parents=True)
    (scene / "images" / "v001.png").write_bytes(b"rgb")
    (scene / "depths" / "v001_depth.npy").write_bytes(b"depth")
    (scene / "transforms.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "view_id": "v001",
                        "file_path": "images/v001.png",
                        "depth_file_path": "depths/v001_depth.npy",
                        "transform_matrix": IDENTITY,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    create_templates(scene, scene / "views")
    if annotate:
        path = scene / "views" / "v001.json"
        view = json.loads(path.read_text(encoding="utf-8"))
        view.update(
            {
                "summary": "A red chair in the conference room.",
                "visible_objects": [
                    {
                        "label": "chair",
                        "attributes": ["red"],
                        "approx_location": "center",
                        "confidence": None,
                        "bbox_2d": None,
                        "bbox_3d": None,
                        "source": "manual",
                    }
                ],
                "warnings": [],
            }
        )
        path.write_text(json.dumps(view), encoding="utf-8")
    build_scene_tree(scene, scene / "views", scene / "tree")
    return scene


def test_live_session_uses_captured_index_as_explicit_stub(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    prepare_scene(data_dir, annotate=True)
    monkeypatch.setattr(live_session, "DATA_DIR", data_dir)
    monkeypatch.setattr(query_log, "DATA_DIR", data_dir)
    session = QuerySession("captured", "Find the red chair")

    result = live_session.run_query_session(
        "captured", session.session_id, step_delay_sec=0
    )

    assert result.result["found"] is True
    assert result.result["view_id"] == "v001"
    assert result.result["bbox_3d"] is None
    assert result.result["mode"] == "stub"
    assert result.result["semantic_index_mode"] == "manual"
    assert result.result["explanation"].startswith("[stub]")


def test_live_session_rejects_empty_templates_instead_of_negative_result(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    prepare_scene(data_dir, annotate=False)
    monkeypatch.setattr(live_session, "DATA_DIR", data_dir)
    monkeypatch.setattr(query_log, "DATA_DIR", data_dir)
    session = QuerySession("captured", "Find a chair")

    with pytest.raises(ValueError, match="incomplete; manual annotation is required"):
        live_session.run_query_session("captured", session.session_id, step_delay_sec=0)


def test_live_session_ignores_new_capture_until_viewjson_exists(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    scene = prepare_scene(data_dir, annotate=True)
    transforms_path = scene / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    (scene / "images" / "v002.png").write_bytes(b"new-rgb")
    (scene / "depths" / "v002_depth.npy").write_bytes(b"new-depth")
    transforms["frames"].append(
        {
            "view_id": "v002",
            "file_path": "images/v002.png",
            "depth_file_path": "depths/v002_depth.npy",
            "transform_matrix": IDENTITY,
        }
    )
    transforms_path.write_text(json.dumps(transforms), encoding="utf-8")

    monkeypatch.setattr(live_session, "DATA_DIR", data_dir)
    monkeypatch.setattr(query_log, "DATA_DIR", data_dir)

    with pytest.raises(ValueError, match="missing ViewJSON: v002"):
        load_captured_scene_index(scene, require_complete=True)

    session = QuerySession("captured", "Find a chair")
    result = live_session.run_query_session(
        "captured", session.session_id, step_delay_sec=0
    )

    assert result.result["found"] is True
    assert result.result["view_id"] == "v001"
    assert any("v002" in warning for warning in result.result["warnings"])
