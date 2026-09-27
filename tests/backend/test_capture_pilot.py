import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from backend.api.routes import captures
from backend.schemas.types import CapturePayload
from scripts.check_capture_pilot import check_capture
from scripts.check_capture_batch import check_capture_batch
from scripts.validate_scene_geometry import validate_scene


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _png_b64(size=(2, 2)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 80, 40)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _payload(depth: np.ndarray) -> CapturePayload:
    return CapturePayload(
        scene_id="pilot",
        view_id="v001",
        rgb_b64=_png_b64(),
        depth_b64=base64.b64encode(depth.astype("<f4").tobytes()).decode("ascii"),
        transform_matrix=IDENTITY,
        fl_x=2.0,
        fl_y=2.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        source_ply_url="/scenes/ConferenceHall.ply",
        captured_at_utc="2026-06-22T00:00:00+00:00",
        pose_coordinate_convention="threejs_world_camera_to_world_rh_y_up_camera_forward_minus_z",
        depth_source="spark_setDepthColor_log_view_space_splat_center",
        depth_near=0.1,
        depth_far=100.0,
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_capture_route_writes_validated_files_and_metadata(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    scene = data_dir / "pilot"
    scene.mkdir(parents=True)
    monkeypatch.setattr(captures, "DATA_DIR", data_dir)

    response = captures.save_capture(_payload(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)))

    assert response["status"] == "saved"
    assert (scene / "images" / "v001.png").exists()
    assert (scene / "depths" / "v001_depth.npy").exists()
    metadata = json.loads((scene / "captures" / "v001.json").read_text(encoding="utf-8"))
    transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    assert metadata["scene_id"] == "pilot"
    assert metadata["frame_id"] == "v001"
    assert metadata["depth"]["constant"] is False
    assert transforms["w"] == 2
    assert transforms["h"] == 2
    assert transforms["frames"][0]["depth_file_path"] == "depths/v001_depth.npy"

    report = check_capture(scene)
    assert report["scaling_allowed"] is True
    assert report["semantic_eval_allowed"] is False
    assert report["geometry_eval_allowed"] is False
    assert report["intrinsics_match_canvas"] is True


def test_capture_route_rejects_constant_depth(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    (data_dir / "pilot").mkdir(parents=True)
    monkeypatch.setattr(captures, "DATA_DIR", data_dir)

    with pytest.raises(HTTPException, match="Depth is constant") as error:
        captures.save_capture(_payload(np.full((2, 2), 0.1, dtype=np.float32)))

    assert error.value.status_code == 422
    assert not (data_dir / "pilot" / "images" / "v001.png").exists()


def test_validator_requires_semantic_index_and_3d_gt(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    scene = data_dir / "pilot"
    scene.mkdir(parents=True)
    monkeypatch.setattr(captures, "DATA_DIR", data_dir)
    captures.save_capture(_payload(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)))

    report = validate_scene(scene)
    assert report["geometry_inputs_valid"] is True
    assert report["semantic_evaluation_allowed"] is False
    assert report["three_d_localization_allowed"] is False

    _write_json(scene / "views" / "v001.json", {"view_id": "v001"})
    _write_json(scene / "tree" / "manifest.json", {"root_id": "root", "node_ids": ["root"]})
    _write_json(scene / "tree" / "node_root.json", {"node_id": "root"})
    report = validate_scene(scene)
    assert report["semantic_evaluation_allowed"] is True
    assert report["three_d_localization_allowed"] is False


def test_batch_checker_writes_valid_and_unavailable_reports(tmp_path, monkeypatch):
    data_dir = tmp_path / "scenes"
    valid_scene = data_dir / "valid-capture"
    valid_scene.mkdir(parents=True)
    missing_scene = data_dir / "not-captured"
    monkeypatch.setattr(captures, "DATA_DIR", data_dir)
    payload = _payload(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    payload.scene_id = valid_scene.name
    captures.save_capture(payload)

    out_dir = tmp_path / "reports"
    reports = check_capture_batch([valid_scene, missing_scene], out_dir)

    assert len(reports) == 2
    assert reports[0]["scaling_allowed"] is True
    assert reports[1]["scaling_allowed"] is False
    assert reports[1]["warnings"][0].startswith("Capture validation unavailable:")
    assert (out_dir / "valid-capture_capture_check.json").exists()
    assert (out_dir / "not-captured_capture_check.json").exists()
