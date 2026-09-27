#!/usr/bin/env python3
"""Export reviewer-control figures and LaTeX macros from frozen outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#1769aa"
GREEN = "#18794e"
ORANGE = "#c44e0b"
GRAY = "#667085"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finish(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=240)
    plt.close(fig)


def scaling_figure(data: dict, stem: Path) -> None:
    measured = data["measured"]
    extra = data["extrapolated"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), constrained_layout=True)
    definitions = (("views", "Views checked/query"), ("tokens", "Serialized tokens/query"))
    for ax, (key, label) in zip(axes, definitions, strict=True):
        x = [row["view_count"] for row in measured]
        for method, color, marker in (("flat", ORANGE, "s"), ("graph", GREEN, "o")):
            y = [row[f"{method}_{key}"] for row in measured]
            ax.plot(x, y, marker=marker, color=color, linewidth=1.6, label=f"{method.title()} measured")
            xx = [x[-1], *[row["view_count"] for row in extra]]
            yy = [y[-1], *[row[f"{method}_{key}"] for row in extra]]
            ax.plot(xx, yy, linestyle="--", color=color, linewidth=1.1, alpha=0.8)
            ax.scatter(xx[1:], yy[1:], marker=marker, facecolors="white", edgecolors=color, zorder=3)
        ax.axvspan(97, 1000, color="#f2f4f7", alpha=0.7, zorder=0)
        ax.axvline(97, color=GRAY, linestyle=":", linewidth=1)
        ax.text(105, ax.get_ylim()[1] * 0.92, "extrapolated", color=GRAY, fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("Collection size (views)")
        ax.set_ylabel(label)
        ax.grid(color="#e4e7ec", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    finish(fig, stem)


def reviewer_controls_figure(relation: dict, hierarchy: dict, stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), constrained_layout=True)
    ax = axes[0]
    values = [relation["without_relation"]["acc_at_0_25"], relation["with_relation"]["acc_at_0_25"]]
    ax.plot([0, 1], values, color="#98a2b3", linewidth=2)
    ax.scatter([0, 1], values, s=75, color=[GRAY, GREEN], edgecolor="white", zorder=3)
    for x, value in enumerate(values):
        ax.text(x, value + 0.018, f"{value:.3f}", ha="center", fontsize=8.5)
    delta = relation["paired_bootstrap"]["acc_at_0_25"]
    ax.text(0.5, 0.37, f"paired delta {delta['estimate']:+.3f}\n95% CI [{delta['ci_95']['low']:+.3f}, {delta['ci_95']['high']:+.3f}]", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["No relation", "Relation"])
    ax.set_ylabel("Acc@0.25 (48 matched queries)")
    ax.set_ylim(0.34, 0.64)
    ax.grid(axis="y", color="#e4e7ec", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) Frozen-candidate relation ablation", loc="left", fontweight="bold")

    ax = axes[1]
    names = ("Pose", "Pose+semantic", "Cursor+MCP")
    keys = ("pose_only", "pose_semantic_merge", "cursor_agent_mcp")
    actual = []
    random_values = []
    for key in keys:
        rows = [row for row in hierarchy["structure"]["per_scene"] if row["variant"] == key]
        null = [row for row in hierarchy["structure"]["random_matched_complexity"] if row["variant"] == key]
        actual.append(np.mean([row["pairwise_f1"] for row in rows]))
        random_values.append(np.mean([row["random_pairwise_f1_mean"] for row in null]))
    projection = np.mean([row["pairwise_f1"] for row in hierarchy["structure"]["per_scene"] if row["variant"] == "disjoint_manual_projection"])
    y = np.arange(len(names))[::-1]
    for yi, low, high in zip(y, random_values, actual, strict=True):
        ax.plot([low, high], [yi, yi], color="#cbd5e1", linewidth=3, solid_capstyle="round")
    ax.scatter(random_values, y, marker="x", s=48, color=GRAY, label="Matched random")
    ax.scatter(actual, y, marker="o", s=52, color=BLUE, edgecolor="white", label="Observed")
    ax.axvline(projection, color=GREEN, linestyle="--", linewidth=1.3, label="Disjoint reference")
    ax.set_yticks(y, names)
    ax.set_xlabel("Macro pairwise F1")
    ax.set_xlim(0.35, 0.95)
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) Representation-aware structure controls", loc="left", fontweight="bold")
    finish(fig, stem)


def macros(scaling: dict, relation: dict, hierarchy: dict) -> str:
    last = scaling["measured"][-1]
    projection = np.mean([row["pairwise_f1"] for row in hierarchy["structure"]["per_scene"] if row["variant"] == "disjoint_manual_projection"])
    random_cursor = np.mean([row["random_pairwise_f1_mean"] for row in hierarchy["structure"]["random_matched_complexity"] if row["variant"] == "cursor_agent_mcp"])
    budget6 = {row["variant"]: row for row in hierarchy["matched_budgets"] if row["budget"] == 6}
    values = {
        "ScaleMeasuredViews": str(last["view_count"]),
        "ScaleFlatEntries": f"{last['flat_entries']:.0f}",
        "ScaleGraphEntries": f"{last['graph_entries']:.1f}",
        "ScaleFlatTokens": f"{last['flat_tokens']:.0f}",
        "ScaleGraphTokens": f"{last['graph_tokens']:.0f}",
        "MatchedRelationWith": f"{relation['with_relation']['acc_at_0_25']:.3f}",
        "MatchedRelationWithHitThree": f"{relation['with_relation']['hit_at_3']:.3f}",
        "MatchedRelationWithout": f"{relation['without_relation']['acc_at_0_25']:.3f}",
        "MatchedRelationWithoutHitThree": f"{relation['without_relation']['hit_at_3']:.3f}",
        "MatchedRelationDelta": f"{relation['paired_bootstrap']['acc_at_0_25']['estimate']:.3f}",
        "MatchedRelationLow": f"{relation['paired_bootstrap']['acc_at_0_25']['ci_95']['low']:.3f}",
        "MatchedRelationHigh": f"{relation['paired_bootstrap']['acc_at_0_25']['ci_95']['high']:.3f}",
        "DisjointReferenceFOne": f"{projection:.3f}",
        "CursorRandomFOne": f"{random_cursor:.3f}",
        "BudgetSixManualHitThree": f"{budget6['manual_reference']['hit_at_3']:.3f}",
        "BudgetSixPoseHitThree": f"{budget6['pose_only']['hit_at_3']:.3f}",
        "BudgetSixSemanticHitThree": f"{budget6['pose_semantic_merge']['hit_at_3']:.3f}",
        "BudgetSixCursorHitThree": f"{budget6['cursor_agent_mcp']['hit_at_3']:.3f}",
    }
    return "% Generated by scripts/export_reviewer_revision.py.\n" + "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling", type=Path, default=ROOT / "outputs/reviewer_revision/scaling_v1/scaling_summary.json")
    parser.add_argument("--relation", type=Path, default=ROOT / "outputs/reviewer_revision/matched_relation_v1/matched_relation_summary.json")
    parser.add_argument("--hierarchy", type=Path, default=ROOT / "outputs/reviewer_revision/matched_hierarchy_v1/matched_hierarchy_summary.json")
    parser.add_argument("--paper", type=Path, default=ROOT / "papers/twinworld")
    args = parser.parse_args()
    scaling, relation, hierarchy = read(args.scaling), read(args.relation), read(args.hierarchy)
    figures = args.paper / "figures"
    scaling_figure(scaling, figures / "fig_reviewer_scaling")
    reviewer_controls_figure(relation, hierarchy, figures / "fig_reviewer_controls")
    (args.paper / "tables/reviewer_revision_macros.tex").write_text(macros(scaling, relation, hierarchy), encoding="utf-8")


if __name__ == "__main__":
    main()
