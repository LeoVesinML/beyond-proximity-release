"""Tests for backend/io/nerfstudio.py"""
import json
import os
import tempfile

import pytest

from backend.io.nerfstudio import (
    FrameEntry,
    NerfstudioScene,
    get_or_create_transforms,
    load_transforms,
    save_transforms,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
SAMPLE_TRANSFORMS = os.path.join(FIXTURES, "sample_transforms.json")


class TestFrameEntry:
    def test_camera_position(self):
        m = [[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]]
        f = FrameEntry(file_path="img.png", transform_matrix=m, view_id="v1")
        assert f.camera_position() == [3, 4, 5]

    def test_rotation_matrix(self):
        m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        f = FrameEntry(file_path="img.png", transform_matrix=m)
        rot = f.rotation_matrix_3x3()
        assert rot == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_to_from_dict_roundtrip(self):
        m = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        f = FrameEntry(
            file_path="images/v001.png",
            transform_matrix=m,
            view_id="v001",
            depth_file_path="depths/v001_depth.npy",
        )
        d = f.to_dict()
        f2 = FrameEntry.from_dict(d)
        assert f2.view_id == "v001"
        assert f2.camera_position() == [1, 2, 3]
        assert f2.depth_file_path == "depths/v001_depth.npy"


class TestNerfstudioScene:
    def test_add_and_get_frame(self):
        scene = NerfstudioScene(fl_x=800, fl_y=800, cx=400, cy=300, w=800, h=600)
        m = [[1, 0, 0, 0]] * 4
        scene.add_frame(FrameEntry(file_path="img.png", transform_matrix=m, view_id="v1"))
        assert scene.get_frame("v1") is not None
        assert scene.get_frame("v999") is None

    def test_to_from_dict_roundtrip(self):
        scene = NerfstudioScene(fl_x=600, fl_y=600, cx=300, cy=200, w=600, h=400)
        scene.add_frame(FrameEntry(
            file_path="images/v001.png",
            transform_matrix=[[1, 0, 0, 0]] * 4,
            view_id="v001",
        ))
        d = scene.to_dict()
        scene2 = NerfstudioScene.from_dict(d)
        assert scene2.fl_x == 600
        assert len(scene2.frames) == 1
        assert scene2.frames[0].view_id == "v001"


class TestIOFunctions:
    def test_save_and_load_roundtrip(self, tmp_path):
        scene = NerfstudioScene(fl_x=800, fl_y=800, cx=400, cy=300, w=800, h=600)
        scene.add_frame(FrameEntry(
            file_path="images/v001.png",
            transform_matrix=[[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            view_id="v001",
        ))
        path = str(tmp_path / "transforms.json")
        save_transforms(scene, path)
        loaded = load_transforms(path)
        assert loaded.fl_x == 800
        assert loaded.get_frame("v001").camera_position()[0] == 5

    def test_load_fixture(self):
        scene = load_transforms(SAMPLE_TRANSFORMS)
        assert len(scene.frames) == 3
        assert scene.get_frame("v001") is not None
        assert scene.get_frame("v002").camera_position()[0] == 2.0

    def test_get_or_create_creates_new(self, tmp_path):
        path = str(tmp_path / "new_transforms.json")
        assert not os.path.exists(path)
        scene = get_or_create_transforms(path)
        assert os.path.exists(path)
        assert scene.fl_x == 800.0

    def test_get_or_create_loads_existing(self, tmp_path):
        path = str(tmp_path / "transforms.json")
        scene = NerfstudioScene(fl_x=999, fl_y=999, cx=500, cy=400, w=1000, h=800)
        save_transforms(scene, path)
        loaded = get_or_create_transforms(path)
        assert loaded.fl_x == 999
