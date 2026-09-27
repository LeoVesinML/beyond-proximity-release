#!/usr/bin/env python3
"""Generate evidence-backed overview figures for the academic preprint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import export_twinworld_figures as figures


ROOT = Path(__file__).resolve().parent.parent
GREEN = "#237A4B"
BLUE = "#1D5FA7"
ORANGE = "#C84B0E"
INK = "#1F2933"
MUTED = "#667085"
GRID = "#D9DEE7"
PALE_GREEN = "#EAF5EF"
PALE_BLUE = "#EAF2FB"
PALE_ORANGE = "#FFF0E8"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".pdf", {}),
        (".png", {"dpi": 260}),
    ):
        target = out_dir / f"{stem}{suffix}"
        temporary = out_dir / f".{stem}.tmp{suffix}"
        fig.savefig(temporary, bbox_inches="tight", pad_inches=0.03, **kwargs)
        if target.is_file() and target.read_bytes() == temporary.read_bytes():
            temporary.unlink()
        else:
            temporary.replace(target)
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.45,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=7.6,
        color=MUTED,
        linespacing=1.2,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.35,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def render_agent_mcp_protocol(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.94,
        "QUERY-TIME AGENT PROTOCOL",
        fontsize=11,
        fontweight="bold",
        color=BLUE,
    )
    top_y = 0.60
    width = 0.16
    height = 0.22
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    boxes = [
        ("Natural-language query", "target, attributes,\nanchor, intent", PALE_BLUE, BLUE),
        ("Agent decomposition", "structured query and\nsearch constraints", PALE_BLUE, BLUE),
        ("MCP hierarchy access", "root/children tools;\nscene records stay local", PALE_GREEN, GREEN),
        ("Expand or prune", "conservative branch\nselection with trace", PALE_GREEN, GREEN),
        ("Candidate evidence", "leaf views, objects,\nboxes, relations", PALE_GREEN, GREEN),
    ]
    for x, (title, body, face, edge) in zip(xs, boxes):
        _box(ax, (x, top_y), (width, height), title, body, face=face, edge=edge)
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, (left + width, top_y + height / 2), (right, top_y + height / 2))

    lower_y = 0.18
    _box(
        ax,
        (0.22, lower_y),
        (0.23, 0.21),
        "Semantic confirmation",
        "shared ranker; optional\nvalidated relation evidence",
        face=PALE_ORANGE,
        edge=ORANGE,
    )
    _box(
        ax,
        (0.54, lower_y),
        (0.20, 0.21),
        "Grounded result",
        "ranked IDs, supporting view,\nbox and confidence",
        face=PALE_ORANGE,
        edge=ORANGE,
    )
    _box(
        ax,
        (0.82, lower_y),
        (0.16, 0.21),
        "Replayable trace",
        "calls, branches, tokens,\nlatency and fallback",
        face="#F4F5F7",
        edge=MUTED,
    )
    ax.plot([0.90, 0.90, 0.335], [top_y, 0.50, 0.50], color=GREEN, linewidth=1.35)
    _arrow(ax, (0.335, 0.50), (0.335, lower_y + 0.21), color=GREEN)
    _arrow(ax, (0.45, lower_y + 0.105), (0.54, lower_y + 0.105), color=ORANGE)
    _arrow(ax, (0.74, lower_y + 0.105), (0.82, lower_y + 0.105), color=MUTED)

    ax.text(
        0.02,
        0.47,
        "CLIENTS",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.02,
        0.405,
        "Cursor Agent: interactive product client\n"
        "Pinned BGE encoder: reproducible semantic client\n"
        "Lexical overlap: controlled baseline only",
        fontsize=8.4,
        color=MUTED,
        va="top",
        linespacing=1.35,
    )
    ax.text(
        0.02,
        0.08,
        "One-time hierarchy construction is evaluated separately from per-query traversal.",
        fontsize=8.3,
        color=MUTED,
    )
    _save(fig, out_dir, "fig_agent_mcp_protocol")


def render_agent_semantic_results(out_dir: Path) -> None:
    data = _load(
        ROOT
        / "outputs"
        / "agent_semantic"
        / "five_scene_four_variant_v1"
        / "metrics_summary.json"
    )
    keys = [
        "flat_lexical",
        "graph_lexical",
        "flat_semantic_embedding",
        "graph_semantic_embedding",
    ]
    labels = ["Flat\nlexical", "Graph\nlexical", "Flat\nsemantic", "Graph\nsemantic"]
    colors = [ORANGE, GREEN, "#E58A5B", "#56A779"]
    tokens = [data["methods"][key]["efficiency"]["input_tokens"] for key in keys]
    views = [data["methods"][key]["efficiency"]["views_checked"] for key in keys]
    hit1 = [data["methods"][key]["quality"]["hit_at_1"] for key in keys]
    hit3 = [data["methods"][key]["quality"]["hit_at_3"] for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(keys))
    axes[0].bar(x, tokens, color=colors, width=0.66)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Input tokens per query")
    axes[0].set_title("Measured query context")
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_axisbelow(True)
    for index, value in enumerate(tokens):
        axes[0].text(index, value + 55, f"{value:,.0f}", ha="center", fontsize=8)
        axes[0].text(
            index,
            max(value * 0.48, 180),
            f"{views[index]:.1f} views",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )

    bar_width = 0.34
    axes[1].bar(x - bar_width / 2, hit1, bar_width, color=BLUE, label="hit@1")
    axes[1].bar(x + bar_width / 2, hit3, bar_width, color=GREEN, label="hit@3")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Retrieval quality (n=125)")
    axes[1].set_title("Verified-view retrieval")
    axes[1].grid(axis="y", color=GRID, linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower left")
    for index, (v1, v3) in enumerate(zip(hit1, hit3)):
        axes[1].text(index - bar_width / 2, v1 + 0.018, f"{v1:.3f}", ha="center", fontsize=7)
        axes[1].text(index + bar_width / 2, v3 + 0.018, f"{v3:.3f}", ha="center", fontsize=7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Same scenes, records, queries, and quality labels; graph variants change traversal only",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    _save(fig, out_dir, "fig_agent_semantic_results")


def render_hierarchy_comparison(out_dir: Path) -> None:
    data = _load(
        ROOT
        / "docs"
        / "experiments"
        / "hierarchy_construction"
        / "four_variant_v1"
        / "hierarchy_evaluation.json"
    )
    keys = ["manual_reference", "pose_only", "pose_semantic_merge", "cursor_agent_mcp"]
    labels = ["Manual\nreference", "Pose\nonly", "Pose +\nsemantic", "Cursor Agent\n+ MCP"]
    colors = [MUTED, BLUE, ORANGE, GREEN]
    variants = [data["variants"][key] for key in keys]
    hit3 = [item["query_metrics"]["hit_at_3"] for item in variants]
    tokens = [item["query_metrics"]["mean_input_tokens"] for item in variants]
    agreement = [item["assignment_agreement"]["weighted_best_zone_jaccard"] for item in variants]
    zones = [item["structure"]["zone_count"] for item in variants]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(keys))
    axes[0].bar(x, hit3, color=colors, width=0.66)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("hit@3 (n=125)")
    axes[0].set_title("Query quality by hierarchy")
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_axisbelow(True)
    for index, value in enumerate(hit3):
        axes[0].text(index, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)

    label_offsets = [(6, 6), (-4, 8), (-98, 8), (6, 8)]
    for index, (xv, yv) in enumerate(zip(tokens, agreement)):
        axes[1].scatter(
            xv,
            yv,
            s=110 + zones[index] * 4,
            color=colors[index],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axes[1].annotate(
            labels[index].replace("\n", " "),
            (xv, yv),
            xytext=label_offsets[index],
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[1].set_xlabel("Estimated query tokens")
    axes[1].set_ylabel("Best-zone Jaccard to manual reference")
    axes[1].set_title("Structure agreement versus query context")
    axes[1].set_xlim(970, 1870)
    axes[1].set_ylim(0.48, 1.025)
    axes[1].grid(color=GRID, linewidth=0.7)
    axes[1].text(
        0.02,
        0.04,
        "Marker area encodes zone count",
        transform=axes[1].transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Four hierarchy-construction variants over the same 97 views and 150 queries",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    _save(fig, out_dir, "fig_hierarchy_comparison")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "papers" / "beyond-proximity" / "figures",
    )
    args = parser.parse_args()
    figures.OUT = args.out_dir.resolve()
    figures._setup()
    method_pdf = figures.OUT / "fig_method_workflow.pdf"
    method_png = figures.OUT / "fig_method_workflow.png"
    if not method_pdf.is_file() or not method_png.is_file():
        figures.render_system_overview()
    render_agent_mcp_protocol(figures.OUT)
    render_agent_semantic_results(figures.OUT)
    render_hierarchy_comparison(figures.OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
