from scripts.run_ieee_sensitivity import SWEEPS


def test_sensitivity_sweeps_cover_selected_recall_controls() -> None:
    fields = {field for field, _values in SWEEPS.values()}
    assert fields == {
        "branch_mass_threshold",
        "score_temperature",
        "max_branches",
        "safety_shortlist_size",
        "candidate_budget",
        "boundary_epsilon",
    }
    assert all(len(values) >= 4 for _field, values in SWEEPS.values())
