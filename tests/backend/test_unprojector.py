"""Tests for backend/geometry/unprojector.py"""
import numpy as np
import pytest

from backend.geometry.unprojector import (
    bbox_2d_to_3d,
    depth_stats_in_bbox,
    merge_bboxes,
    unproject_point,
)
from backend.schemas.types import BBox3D


class TestUnprojectPoint:
    def test_identity_camera(self):
        c2w = np.eye(4)
        # Principal ray at cx, cy with depth 1 → should be [0, 0, 1] in world space
        fl_x, fl_y, cx, cy = 800.0, 800.0, 400.0, 300.0
        p = unproject_point(cx, cy, 1.0, fl_x, fl_y, cx, cy, c2w)
        assert abs(p[0]) < 1e-6
        assert abs(p[1]) < 1e-6
        assert abs(p[2] - 1.0) < 1e-6

    def test_translated_camera(self):
        c2w = np.eye(4)
        c2w[0, 3] = 5.0  # camera translated 5 units along X
        fl_x, fl_y, cx, cy = 800.0, 800.0, 400.0, 300.0
        p = unproject_point(cx, cy, 1.0, fl_x, fl_y, cx, cy, c2w)
        assert abs(p[0] - 5.0) < 1e-6
        assert abs(p[2] - 1.0) < 1e-6


class TestDepthStats:
    def test_uniform_depth(self):
        depth = np.ones((100, 100), dtype=np.float32) * 3.0
        med, std = depth_stats_in_bbox(depth, 0.1, 0.1, 0.9, 0.9, 100, 100)
        assert abs(med - 3.0) < 1e-4
        assert abs(std) < 1e-4

    def test_empty_region(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        med, std = depth_stats_in_bbox(depth, 0.0, 0.0, 0.5, 0.5, 100, 100)
        assert med == 0.0


class TestBbox2dTo3d:
    def test_basic(self):
        depth = np.ones((600, 800), dtype=np.float32) * 2.0
        c2w = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        bbox = bbox_2d_to_3d(
            bbox_norm=(0.1, 0.1, 0.9, 0.9),
            depth_map=depth,
            fl_x=800.0,
            fl_y=800.0,
            cx=400.0,
            cy=300.0,
            transform_matrix=c2w,
            width=800,
            height=600,
        )
        assert bbox is not None
        assert abs(bbox.center[2] - 2.0) < 0.2

    def test_zero_depth_returns_none(self):
        depth = np.zeros((600, 800), dtype=np.float32)
        c2w = [[1, 0, 0, 0]] * 4
        result = bbox_2d_to_3d(
            (0.2, 0.2, 0.8, 0.8), depth, 800.0, 800.0, 400.0, 300.0, c2w, 800, 600
        )
        assert result is None


class TestMergeBboxes:
    def test_merge_two(self):
        b1 = BBox3D(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        b2 = BBox3D(center=(2.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        merged = merge_bboxes([b1, b2])
        assert merged is not None
        assert abs(merged.center[0] - 1.0) < 1e-6
        assert abs(merged.size[0] - 4.0) < 1e-6

    def test_empty(self):
        assert merge_bboxes([]) is None

    def test_single(self):
        b = BBox3D(center=(1.0, 2.0, 3.0), size=(1.0, 1.0, 1.0))
        merged = merge_bboxes([b])
        assert merged.center == b.center
