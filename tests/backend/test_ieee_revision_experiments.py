"""Unit tests for IEEE revision experiment helpers."""
import numpy as np

from scripts.run_ieee_public_scale_ann import _quality, _query_indices, _top_indices


def _records():
    return [
        {"scene_id": "s1", "object_id": "a", "record_key": "r0"},
        {"scene_id": "s1", "object_id": "a", "record_key": "r1"},
        {"scene_id": "s1", "object_id": "b", "record_key": "r2"},
        {"scene_id": "s2", "object_id": "a", "record_key": "r3"},
    ]


def test_top_indices_excludes_query_record():
    ranked = _top_indices(np.array([1.0, 0.8, 0.7], dtype=np.float32), 2, excluded=0)
    assert ranked == [1, 2]


def test_scale_quality_requires_same_scene_and_object():
    records = _records()
    quality = _quality([3, 1, 2], records, records[0])
    assert quality["hit_at_1"] is False
    assert quality["hit_at_3"] is True
    assert quality["mrr"] == 0.5


def test_scale_query_selection_requires_an_alternate_description():
    selected = _query_indices(_records(), size=4, limit=10)
    assert set(selected) == {0, 1}
