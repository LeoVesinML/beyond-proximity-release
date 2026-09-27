#!/usr/bin/env python3
"""Validate journal-local assets without rewriting the IEEE Access manuscript."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "ieee-access"

REQUIRED = (
    "main.tex",
    "refs.bib",
    "ieeeaccess.cls",
    "tables/new_experiment_macros.tex",
    "tables/reviewer_revision_macros.tex",
    "tables/ieee_revision_macros.tex",
    "figures/fig_tree_traversal.pdf",
    "figures/fig_agent_mcp_workflow.pdf",
    "figures/fig_scene_gallery.pdf",
    "figures/fig_raw_rgbd_hierarchy.pdf",
    "figures/fig_reviewer_controls.pdf",
    "figures/fig_scorer_frontiers.pdf",
    "figures/fig_mitigation_ablation.pdf",
    "figures/fig_public_scale_ann.pdf",
    "figures/fig_predicted_map_pair.pdf",
    "figures/fig_sensitivity.pdf",
)


def main() -> int:
    missing = [relative for relative in REQUIRED if not (PAPER / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "IEEE Access journal-local assets are missing: " + ", ".join(missing)
        )
    print("IEEE Access source is journal-owned; all required local assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
