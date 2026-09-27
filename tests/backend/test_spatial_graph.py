"""Tests for backend/geometry/spatial_graph.py"""
import pytest

from backend.geometry.spatial_graph import (
    build_adjacency,
    build_spatial_clusters,
    cluster_centroid,
    connected_components,
)


class TestBuildAdjacency:
    def test_two_close_views(self):
        positions = {
            "v1": [0.0, 0.0, 0.0],
            "v2": [0.1, 0.0, 0.0],
        }
        adj = build_adjacency(positions, scale_factor=10.0)
        assert "v2" in adj["v1"]

    def test_two_far_views(self):
        # With 3 points: v1-v2 close (d=0.1), v3 far (d=100).
        # Median NN dist ≈ 0.1, threshold = 0.5 × 0.1 = 0.05 → v3 isolated.
        positions = {
            "v1": [0.0,   0.0, 0.0],
            "v2": [0.1,   0.0, 0.0],
            "v3": [100.0, 0.0, 0.0],
        }
        adj = build_adjacency(positions, scale_factor=0.5)
        assert "v3" not in adj["v1"]
        assert "v3" not in adj["v2"]

    def test_single_view(self):
        positions = {"v1": [0.0, 0.0, 0.0]}
        adj = build_adjacency(positions)
        assert adj == {"v1": []}

    def test_empty(self):
        assert build_adjacency({}) == {}


class TestConnectedComponents:
    def test_two_clusters(self):
        adj = {
            "a": ["b"],
            "b": ["a"],
            "c": ["d"],
            "d": ["c"],
        }
        comps = connected_components(adj)
        assert len(comps) == 2
        flat = {v for c in comps for v in c}
        assert flat == {"a", "b", "c", "d"}

    def test_single_component(self):
        adj = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        comps = connected_components(adj)
        assert len(comps) == 1
        assert set(comps[0]) == {"a", "b", "c"}

    def test_isolated_nodes(self):
        adj = {"a": [], "b": [], "c": []}
        comps = connected_components(adj)
        assert len(comps) == 3


class TestBuildSpatialClusters:
    def test_naturally_two_clusters(self):
        positions = {
            "v1": [0.0, 0.0, 0.0],
            "v2": [0.2, 0.0, 0.0],
            "v3": [10.0, 0.0, 0.0],
            "v4": [10.2, 0.0, 0.0],
        }
        clusters = build_spatial_clusters(positions)
        assert len(clusters) == 2

    def test_single_cluster(self):
        positions = {f"v{i}": [float(i), 0.0, 0.0] for i in range(5)}
        clusters = build_spatial_clusters(positions)
        # All close together → one cluster
        assert len(clusters) == 1


class TestClusterCentroid:
    def test_centroid(self):
        positions = {
            "v1": [0.0, 0.0, 0.0],
            "v2": [2.0, 0.0, 0.0],
        }
        c = cluster_centroid(["v1", "v2"], positions)
        assert abs(c[0] - 1.0) < 1e-6

    def test_missing_view(self):
        positions = {"v1": [1.0, 0.0, 0.0]}
        c = cluster_centroid(["v1", "v999"], positions)
        assert len(c) == 3
