from pathlib import Path

import numpy as np

from scripts.run_predicted_map_pair import _select_branches, _split_spatial


def test_spatial_split_is_complete_disjoint_and_bounded() -> None:
    centers = np.asarray([[float(index), 0.0, 0.0] for index in range(31)])
    branches = _split_spatial(list(range(31)), centers, max_leaf_size=8)
    flattened = [index for branch in branches for index in branch]
    assert sorted(flattened) == list(range(31))
    assert len(flattened) == len(set(flattened))
    assert max(map(len, branches)) <= 8


def test_adaptive_branch_selection_respects_mass_and_cap() -> None:
    selected = _select_branches(np.asarray([2.0, 1.0, 0.0]), threshold=0.7, cap=2)
    assert selected == [0]


def test_predicted_map_inputs_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "outputs/baselines/conceptgraphs_scannet_full_v1/run_config.json").is_file()
    assert (root / "outputs/baselines/conceptgraphs_replica_full_v1/run_config.json").is_file()
