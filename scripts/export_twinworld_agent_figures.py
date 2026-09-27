#!/usr/bin/env python3
"""Export TwinWorld figures and LaTeX macros from the new frozen experiments."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
GREEN = "#18794e"
BLUE = "#1769aa"
ORANGE = "#c44e0b"
GRAY = "#667085"
PALETTE = (GRAY, BLUE, GREEN)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
    }
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finish(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    targets = (
        (stem.with_suffix(".pdf"), "pdf", {"metadata": {"CreationDate": None}}),
        (stem.with_suffix(".png"), "png", {"dpi": 240}),
    )
    for target, file_format, options in targets:
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format=file_format,
            bbox_inches="tight",
            **options,
        )
        rendered = buffer.getvalue()
        if target.exists() and target.read_bytes() == rendered:
            continue
        temporary = target.with_name(f"{target.name}.tmp")
        temporary.write_bytes(rendered)
        temporary.replace(target)
    plt.close(fig)


def annotate(
    ax: plt.Axes,
    bars,
    *,
    digits: int = 2,
    skip_indices: frozenset[int] = frozenset(),
    font_size: float = 6.6,
    inside: bool = False,
) -> list[Text]:
    labels = []
    for index, bar in enumerate(bars):
        if index in skip_indices:
            continue
        value = bar.get_height()
        y = value * 0.92 if inside else value
        labels.append(
            ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.{digits}f}",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=font_size,
            color="white" if inside else "#111827",
            fontweight="bold" if inside else "normal",
            clip_on=False,
            )
        )
    return labels


def panel_title(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    y: float = 1.08,
) -> None:
    ax.text(
        0.0,
        y,
        rf"$\mathbf{{({label})}}$ {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
    )


def style_axes(axes) -> None:
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e7ec", linewidth=0.6)
        ax.set_axisbelow(True)


def assert_no_annotation_overlap(
    fig: plt.Figure,
    labels: list[Text],
    *,
    context: str,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes: list[tuple[str, Bbox]] = [
        (label.get_text(), label.get_window_extent(renderer).expanded(1.04, 1.08))
        for label in labels
    ]
    overlaps = []
    for index, (left_text, left_box) in enumerate(boxes):
        for right_text, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                overlaps.append(f"{left_text!r} with {right_text!r}")
    if overlaps:
        raise RuntimeError(
            f"{context} has overlapping value labels: {', '.join(overlaps)}"
        )


def assert_box_text_contained(
    fig: plt.Figure,
    pairs: list[tuple[FancyBboxPatch, Text]],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for patch, label in pairs:
        box_bounds = patch.get_window_extent(renderer)
        text_bounds = label.get_window_extent(renderer)
        inset = 2.0
        contained = (
            text_bounds.x0 >= box_bounds.x0 + inset
            and text_bounds.x1 <= box_bounds.x1 - inset
            and text_bounds.y0 >= box_bounds.y0 + inset
            and text_bounds.y1 <= box_bounds.y1 - inset
        )
        if not contained:
            failures.append(label.get_text().replace("\n", " / "))
    if failures:
        raise RuntimeError(
            "Workflow box text exceeds its bounds: " + ", ".join(failures)
        )


def wilson_interval(
    proportion: float,
    denominator: int = 125,
    z_score: float = 1.96,
) -> tuple[float, float]:
    z_squared = z_score**2
    scale = 1.0 + z_squared / denominator
    center = (proportion + z_squared / (2.0 * denominator)) / scale
    radius = (
        z_score
        * np.sqrt(
            proportion * (1.0 - proportion) / denominator
            + z_squared / (4.0 * denominator**2)
        )
        / scale
    )
    return center - radius, center + radius


def quality_range_panel(
    ax: plt.Axes,
    method_labels: tuple[str, ...],
    quality: np.ndarray,
    *,
    panel_label: str,
    title: str = "Retrieval quality",
    include_mrr: bool = True,
) -> list[Text]:
    metrics = (("hit@1", BLUE, "o"), ("hit@3", GREEN, "s"))
    rows = np.arange(len(method_labels))[::-1]
    labels: list[Text] = []
    for row, values in zip(rows, quality, strict=True):
        ax.plot(
            values[:2],
            (row + 0.10, row + 0.10),
            color="#cbd5e1",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
    for metric_index, (name, color, marker) in enumerate(metrics):
        values = quality[:, metric_index]
        y_values = rows + 0.10
        intervals = [wilson_interval(float(value)) for value in values]
        lower = values - np.asarray([item[0] for item in intervals])
        upper = np.asarray([item[1] for item in intervals]) - values
        x_error = np.vstack((lower, upper))
        ax.errorbar(
            values,
            y_values,
            xerr=x_error,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=name,
            zorder=3,
        )
    if include_mrr:
        mrr_values = quality[:, 2]
        ax.scatter(
            mrr_values,
            rows - 0.10,
            marker="D",
            s=27,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.45,
            label="MRR",
            zorder=3,
        )
    ax.axvline(1.025, color="#d0d5dd", linewidth=0.8)
    for row, values in zip(rows, quality, strict=True):
        exact = (
            f"{values[0]:.2f} / {values[1]:.2f} / {values[2]:.2f}"
            if include_mrr
            else f"{values[0]:.2f} / {values[1]:.2f}"
        )
        labels.append(
            ax.text(
                1.17,
                row,
                exact,
                ha="center",
                va="center",
                fontsize=8.4,
                color="#111827",
            )
        )
    ax.set_xlim(0.25, 1.34)
    ax.set_ylim(-0.45, len(method_labels) - 0.55)
    ax.set_yticks(rows, method_labels)
    ax.set_xticks((0.25, 0.50, 0.75, 1.00))
    ax.set_xlabel("Score on 125 labeled queries; whiskers show 95% intervals")
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_title(ax, panel_label, title)
    return labels


def categorical_dot_panel(
    ax: plt.Axes,
    method_labels: tuple[str, ...],
    values: list[float] | np.ndarray,
    colors: tuple[str, ...],
    *,
    panel_label: str,
    title: str,
    formatter,
    log_scale: bool = False,
    y_label: str | None = None,
) -> list[Text]:
    x_values = np.arange(len(method_labels))
    labels: list[Text] = []
    ax.plot(x_values, values, color="#d0d5dd", linewidth=1.0, zorder=1)
    for x_value, value, color in zip(x_values, values, colors, strict=True):
        ax.scatter(
            x_value,
            value,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        labels.append(
            ax.annotate(
                formatter(value),
                (x_value, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#111827",
            )
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x_values, method_labels)
    ax.set_xlim(-0.45, len(method_labels) - 0.55)
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#e4e7ec", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    panel_title(ax, panel_label, title)
    return labels


def instruction_figure(agent: dict, controls: dict, output: Path) -> None:
    qwen = agent["methods"]["graph_instruction"]
    methods = [
        ("Lexical\ngraph", controls["methods"]["graph_lexical"]),
        ("BGE\ngraph", controls["methods"]["graph_semantic_embedding"]),
        ("Qwen\nagent", qwen),
    ]
    quality = []
    views = []
    tokens = []
    latency = []
    for label, value in methods:
        if "quality" in value:
            quality.append(
                [
                    value["quality"]["hit_at_1"],
                    value["quality"]["hit_at_3"],
                    value["quality"]["mrr"],
                ]
            )
            efficiency = value["efficiency"]
            views.append(efficiency["views_checked"])
            tokens.append(efficiency["input_tokens"])
            latency.append(efficiency["elapsed_ms"])
        else:
            quality.append([value["hit_at_1"], value["hit_at_3"], value["mrr"]])
            views.append(value["mean_views_checked"])
            tokens.append(value["mean_input_tokens"])
            latency.append(value["mean_latency_ms"])

    quality_matrix = np.asarray(quality)
    fig = plt.figure(figsize=(5.2, 4.15))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.55, 1.0))
    quality_ax = fig.add_subplot(grid[0, :])
    cost_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    method_labels = tuple(value[0].replace("\n", " ") for value in methods)
    value_labels = quality_range_panel(
        quality_ax,
        method_labels,
        quality_matrix,
        panel_label="a",
        title="Quality with 95% hit-rate intervals",
    )
    quality_ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", color=BLUE, linestyle="none",
                markeredgecolor="white", label="hit@1",
            ),
            plt.Line2D(
                [], [], marker="s", color=GREEN, linestyle="none",
                markeredgecolor="white", label="hit@3",
            ),
            plt.Line2D(
                [], [], marker="D", color=ORANGE, linestyle="none",
                markeredgecolor="white", label="MRR",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        ncols=3,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    short_labels = ("Lexical", "BGE", "Qwen")
    value_labels.extend(
        categorical_dot_panel(
            cost_axes[0],
            short_labels,
            views,
            PALETTE,
            panel_label="b",
            title=r"Views / query $\downarrow$",
            formatter=lambda value: f"{value:.1f}",
        )
    )
    value_labels.extend(
        categorical_dot_panel(
            cost_axes[1],
            short_labels[1:],
            tokens[1:],
            PALETTE[1:],
            panel_label="c",
            title=r"Native tokens / query $\downarrow$",
            formatter=lambda value: f"{value:,.0f}",
        )
    )
    value_labels.extend(
        categorical_dot_panel(
            cost_axes[2],
            short_labels,
            latency,
            PALETTE,
            panel_label="d",
            title=r"CPU ms / query $\downarrow$",
            formatter=lambda value: f"{value:,.0f}",
            log_scale=True,
            y_label="log scale",
        )
    )
    cost_axes[0].set_ylim(0, max(views) * 1.28)
    cost_axes[1].set_ylim(0, max(tokens[1:]) * 1.28)
    cost_axes[2].set_ylim(min(latency) / 2.2, max(latency) * 2.2)
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.12,
        top=0.88,
        hspace=0.66,
        wspace=0.54,
    )
    assert_no_annotation_overlap(
        fig,
        value_labels,
        context="Instruction-model stress-test figure",
    )
    finish(fig, output)


def hierarchy_figure(raw: dict, output: Path) -> None:
    keys = ("flat_clip", "manual_hierarchy_clip", "raw_rgbd_hierarchy_clip")
    labels = ("Flat\nCLIP", "Manual\nzones", "Raw RGB-D\nclusters")
    values = [raw["methods"][key] for key in keys]
    quality = np.asarray(
        [[value["hit_at_1"], value["hit_at_3"], value["mrr"]] for value in values]
    )
    views = [value["mean_views_checked"] for value in values]
    structure = raw["structure"]

    fig = plt.figure(figsize=(5.2, 3.35))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1.0), width_ratios=(1.1, 1.0))
    quality_ax = fig.add_subplot(grid[0, :])
    views_ax = fig.add_subplot(grid[1, 0])
    structure_ax = fig.add_subplot(grid[1, 1])
    method_labels = tuple(label.replace("\n", " ") for label in labels)
    value_labels = quality_range_panel(
        quality_ax,
        method_labels,
        quality,
        panel_label="a",
        title="Retrieval quality with 95% hit-rate intervals",
    )
    quality_ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", color=BLUE, linestyle="none",
                markeredgecolor="white", label="hit@1",
            ),
            plt.Line2D(
                [], [], marker="s", color=GREEN, linestyle="none",
                markeredgecolor="white", label="hit@3",
            ),
            plt.Line2D(
                [], [], marker="D", color=ORANGE, linestyle="none",
                markeredgecolor="white", label="MRR",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        ncols=3,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    value_labels.extend(
        categorical_dot_panel(
            views_ax,
            ("Flat", "Manual", "Raw RGB-D"),
            views,
            PALETTE,
            panel_label="b",
            title=r"Views checked / query $\downarrow$",
            formatter=lambda value: f"{value:.1f}",
        )
    )
    views_ax.set_ylim(0, max(views) * 1.25)

    agreement_names = ("Pairwise F1", "Rand index")
    agreement_values = (structure["macro_pairwise_f1"], structure["macro_rand_index"])
    agreement_colors = (GREEN, BLUE)
    agreement_rows = np.arange(2)[::-1]
    for row, name, value, color in zip(
        agreement_rows,
        agreement_names,
        agreement_values,
        agreement_colors,
        strict=True,
    ):
        structure_ax.hlines(row, 0, 1, color="#e4e7ec", linewidth=3.2, zorder=1)
        structure_ax.scatter(
            value,
            row,
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        value_labels.append(
            structure_ax.annotate(
                f"{value:.3f}",
                (value, row),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
            )
        )
    structure_ax.set_xlim(0, 1.0)
    structure_ax.set_ylim(-0.5, 1.5)
    structure_ax.set_yticks(agreement_rows, agreement_names)
    structure_ax.set_xlabel("Agreement with manual zones")
    structure_ax.grid(axis="x", color="#e4e7ec", linewidth=0.65)
    structure_ax.set_axisbelow(True)
    structure_ax.spines[["top", "right", "left"]].set_visible(False)
    structure_ax.tick_params(axis="y", length=0)
    panel_title(structure_ax, "c", "Frozen structure agreement")
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.13,
        top=0.87,
        hspace=0.76,
        wspace=0.62,
    )
    assert_no_annotation_overlap(
        fig,
        value_labels,
        context="Raw RGB-D hierarchy figure",
    )
    finish(fig, output)


def construction_variants_figure(hierarchy: dict, output: Path) -> None:
    variant_keys = (
        "manual_reference",
        "pose_only",
        "pose_semantic_merge",
        "cursor_agent_mcp",
    )
    labels = ("Manual reference", "Pose only", "Pose + semantic", "Cursor + MCP")
    colors = (GRAY, BLUE, GREEN, ORANGE)
    variants = [hierarchy["variants"][key] for key in variant_keys]

    structural = np.asarray(
        [
            [
                value["assignment_agreement"]["pairwise_f1"],
                value["assignment_agreement"]["weighted_best_zone_jaccard"],
            ]
            for value in variants
        ]
    )
    quality = np.asarray(
        [
            [
                value["query_metrics"]["hit_at_1"],
                value["query_metrics"]["hit_at_3"],
            ]
            for value in variants
        ]
    )
    fig = plt.figure(figsize=(5.2, 4.25))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.55, 1.0),
        width_ratios=(1.55, 0.85),
    )
    quality_ax = fig.add_subplot(grid[0, :])
    structure_ax = fig.add_subplot(grid[1, 0])
    views_ax = fig.add_subplot(grid[1, 1])
    value_labels = quality_range_panel(
        quality_ax,
        labels,
        quality,
        panel_label="a",
        title="Retrieval quality with 95% hit-rate intervals",
        include_mrr=False,
    )
    quality_ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", color=BLUE, linestyle="none",
                markeredgecolor="white", label="hit@1",
            ),
            plt.Line2D(
                [], [], marker="s", color=GREEN, linestyle="none",
                markeredgecolor="white", label="hit@3",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    rows = np.arange(len(labels))[::-1]
    for row, pairwise, jaccard in zip(rows, structural[:, 0], structural[:, 1], strict=True):
        structure_ax.plot(
            (pairwise, jaccard),
            (row, row),
            color="#cbd5e1",
            linewidth=2.0,
            solid_capstyle="round",
            zorder=1,
        )
        structure_ax.scatter(
            pairwise,
            row + 0.10,
            s=58,
            marker="o",
            color=BLUE,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        structure_ax.scatter(
            jaccard,
            row - 0.10,
            s=58,
            marker="s",
            color=GREEN,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        value_labels.append(
            structure_ax.text(
                1.035,
                row,
                f"F1 {pairwise:.3f}  |  J {jaccard:.3f}",
                transform=structure_ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8.5,
                clip_on=False,
            )
        )
    structure_ax.set_xlim(0.45, 1.02)
    structure_ax.set_xticks((0.5, 0.7, 0.9, 1.0))
    structure_ax.set_ylim(-0.45, len(labels) - 0.25)
    structure_ax.set_yticks(rows, labels)
    structure_ax.set_xlabel("Agreement with manual zones")
    structure_ax.grid(axis="x", color="#e4e7ec", linewidth=0.65)
    structure_ax.set_axisbelow(True)
    structure_ax.spines[["top", "right", "left"]].set_visible(False)
    structure_ax.tick_params(axis="y", length=0)
    structure_ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", color="none", markerfacecolor=BLUE,
                markeredgecolor="white", label="Pairwise F1",
            ),
            plt.Line2D(
                [], [], marker="s", color="none", markerfacecolor=GREEN,
                markeredgecolor="white", label="Best-zone Jaccard",
            ),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    panel_title(structure_ax, "b", "Structural agreement", y=1.38)

    views = [value["query_metrics"]["mean_views_checked"] for value in variants]
    value_labels.extend(
        categorical_dot_panel(
            views_ax,
            ("Ref.", "Pose", "Pose+S", "Cursor"),
            views,
            colors,
            panel_label="c",
            title=r"Views / query $\downarrow$",
            formatter=lambda value: f"{value:.2f}",
        )
    )
    views_ax.set_ylim(0, max(views) * 1.27)
    views_ax.tick_params(axis="x", labelsize=7.2)
    plt.setp(
        views_ax.get_xticklabels(),
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.12,
        top=0.88,
        hspace=0.74,
        wspace=0.82,
    )
    assert_no_annotation_overlap(
        fig,
        value_labels,
        context="Hierarchy-construction figure",
    )
    finish(fig, output)


def workflow_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    box_text_pairs: list[tuple[FancyBboxPatch, Text]] = []

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        *,
        face: str = "#f7f8fa",
        edge: str = GRAY,
        dashed: bool = False,
        font_size: float = 7.2,
    ) -> None:
        patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
                linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        label = ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#111827",
        )
        box_text_pairs.append((patch, label))

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = GRAY,
        dashed: bool = False,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color=color,
                linewidth=1.1,
                linestyle="--" if dashed else "-",
            )
        )

    ax.text(
        0.2,
        7.65,
        "ONE-TIME HIERARCHY CONSTRUCTION",
        fontsize=7.3,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.2,
        3.05,
        "PER-QUERY RETRIEVAL",
        fontsize=7.3,
        fontweight="bold",
        color=BLUE,
    )

    box(
        1.35,
        6.55,
        2.5,
        1.05,
        "ViewJSON summaries\n+ camera poses",
        face="#eef6ff",
        edge=BLUE,
        font_size=6.5,
    )
    box(4.15, 6.55, 2.2, 0.95, "MCP proximity\nclusters", face="#eef6ff", edge=BLUE)
    box(7.0, 6.55, 2.25, 0.95, "Cursor semantic\npartition + labels", face="#fff5eb", edge=ORANGE)
    box(10.45, 5.55, 2.35, 1.05, "Frozen scene\nhierarchy", face="#eaf7ef", edge=GREEN)
    arrow((2.62, 6.55), (3.02, 6.55), color=BLUE)
    arrow((5.28, 6.55), (5.85, 6.55), color=ORANGE)
    arrow((8.15, 6.55), (9.28, 5.78), color=GREEN)

    box(1.35, 4.55, 2.2, 0.95, "Raw RGB-D\n+ camera poses", face="#eef6ff", edge=BLUE)
    box(4.15, 4.55, 2.2, 0.95, "CLIP + depth\n+ pose features", face="#eef6ff", edge=BLUE)
    box(7.0, 4.55, 2.25, 0.95, "Predicted clusters\n+ CLIP labels", face="#eaf7ef", edge=GREEN)
    arrow((2.47, 4.55), (3.02, 4.55), color=BLUE)
    arrow((5.28, 4.55), (5.85, 4.55), color=GREEN)
    arrow((8.15, 4.55), (9.28, 5.32), color=GREEN)

    box(
        10.45,
        7.15,
        2.35,
        0.82,
        "Manual zones\n(evaluation only)",
        face="#ffffff",
        edge=GRAY,
        dashed=True,
        font_size=6.4,
    )
    arrow((10.45, 6.78), (10.45, 6.12), dashed=True)

    box(1.35, 1.85, 2.2, 0.95, "Natural-language\nquery", face="#eef6ff", edge=BLUE)
    box(
        4.15,
        1.85,
        2.55,
        0.95,
        "Branch gate\nQwen / BGE / lexical",
        face="#fff5eb",
        edge=ORANGE,
        font_size=6.5,
    )
    box(7.0, 1.85, 2.35, 0.95, "Shared leaf scorer\non retained records", face="#f7f8fa", edge=GRAY)
    box(10.45, 1.85, 2.35, 0.95, "Grounded view\nor object evidence", face="#eaf7ef", edge=GREEN)
    arrow((2.47, 1.85), (3.02, 1.85), color=BLUE)
    arrow((5.45, 1.85), (5.82, 1.85), color=ORANGE)
    arrow((8.15, 1.85), (9.25, 1.85), color=GREEN)
    arrow((10.45, 4.98), (5.15, 2.36), color=GREEN, dashed=True)

    ax.text(
        6.0,
        0.45,
        "Construction and query costs are reported separately; Cursor UI tokens are unavailable, never imputed.",
        ha="center",
        fontsize=6.8,
        color=GRAY,
    )
    assert_box_text_contained(fig, box_text_pairs)
    finish(fig, output)


def macros(agent: dict, raw: dict, hierarchy: dict, bootstrap: dict) -> str:
    a = agent["methods"]["graph_instruction"]
    c = raw["construction"]
    s = raw["structure"]
    r = raw["methods"]["raw_rgbd_hierarchy_clip"]
    m = raw["methods"]["manual_hierarchy_clip"]
    f = raw["methods"]["flat_clip"]
    cursor = hierarchy["variants"]["cursor_agent_mcp"]
    pose = hierarchy["variants"]["pose_only"]
    semantic = hierarchy["variants"]["pose_semantic_merge"]
    intervals = bootstrap["metrics"]
    values = {
        "AgentHitOne": f"{a['hit_at_1']:.3f}",
        "AgentHitThree": f"{a['hit_at_3']:.3f}",
        "AgentMRR": f"{a['mrr']:.3f}",
        "AgentViews": f"{a['mean_views_checked']:.2f}",
        "AgentCalls": f"{a['mean_model_calls']:.2f}",
        "AgentInputTokens": f"{a['mean_input_tokens']:.1f}",
        "AgentOutputTokens": f"{a['mean_output_tokens']:.1f}",
        "AgentLatencySeconds": f"{a['mean_latency_ms'] / 1000:.2f}",
        "AgentTotalCalls": str(a["total_model_calls"]),
        "AgentTotalTokens": str(a["total_tokens"]),
        "RawConstructionCalls": str(c["model_call_count"]),
        "RawConstructionTokens": str(c["text_tokens"]),
        "RawConstructionSeconds": f"{c['construction_latency_ms'] / 1000:.2f}",
        "RawPairwiseFOne": f"{s['macro_pairwise_f1']:.3f}",
        "RawRandIndex": f"{s['macro_rand_index']:.3f}",
        "RawHitOne": f"{r['hit_at_1']:.3f}",
        "RawHitThree": f"{r['hit_at_3']:.3f}",
        "RawMRR": f"{r['mrr']:.3f}",
        "RawViews": f"{r['mean_views_checked']:.2f}",
        "ManualHitOne": f"{m['hit_at_1']:.3f}",
        "ManualHitThree": f"{m['hit_at_3']:.3f}",
        "ManualViews": f"{m['mean_views_checked']:.2f}",
        "FlatClipHitOne": f"{f['hit_at_1']:.3f}",
        "FlatClipHitThree": f"{f['hit_at_3']:.3f}",
        "FlatClipViews": f"{f['mean_views_checked']:.2f}",
        "CursorMcpCalls": str(hierarchy["agent_evidence"]["mcp_tool_call_count"]),
        "CursorZones": str(cursor["structure"]["zone_count"]),
        "CursorPairwiseFOne": f"{cursor['assignment_agreement']['pairwise_f1']:.3f}",
        "CursorJaccard": f"{cursor['assignment_agreement']['weighted_best_zone_jaccard']:.3f}",
        "CursorHitOne": f"{cursor['query_metrics']['hit_at_1']:.3f}",
        "CursorHitThree": f"{cursor['query_metrics']['hit_at_3']:.3f}",
        "CursorMRR": f"{cursor['query_metrics']['mrr']:.3f}",
        "CursorViews": f"{cursor['query_metrics']['mean_views_checked']:.2f}",
        "CursorQueryTokens": f"{cursor['query_metrics']['mean_input_tokens']:.0f}",
        "PosePairwiseFOne": f"{pose['assignment_agreement']['pairwise_f1']:.3f}",
        "PoseHitThree": f"{pose['query_metrics']['hit_at_3']:.3f}",
        "SemanticPairwiseFOne": f"{semantic['assignment_agreement']['pairwise_f1']:.3f}",
        "SemanticHitOne": f"{semantic['query_metrics']['hit_at_1']:.3f}",
        "SemanticHitThree": f"{semantic['query_metrics']['hit_at_3']:.3f}",
        "ViewSavingsPerQuery": f"{intervals['view_savings_pct']['estimate']:.1f}",
        "ViewSavingsLow": f"{intervals['view_savings_pct']['ci_95']['low']:.1f}",
        "ViewSavingsHigh": f"{intervals['view_savings_pct']['ci_95']['high']:.1f}",
        "TokenSavingsPerQuery": f"{intervals['token_savings_pct']['estimate']:.1f}",
        "TokenSavingsLow": f"{intervals['token_savings_pct']['ci_95']['low']:.1f}",
        "TokenSavingsHigh": f"{intervals['token_savings_pct']['ci_95']['high']:.1f}",
        "HitOneDelta": f"{intervals['hit_at_1_graph_minus_flat']['estimate']:.3f}",
        "HitOneLow": f"{intervals['hit_at_1_graph_minus_flat']['ci_95']['low']:.3f}",
        "HitOneHigh": f"{intervals['hit_at_1_graph_minus_flat']['ci_95']['high']:.3f}",
        "HitThreeDelta": f"{intervals['hit_at_3_graph_minus_flat']['estimate']:.3f}",
        "HitThreeLow": f"{intervals['hit_at_3_graph_minus_flat']['ci_95']['low']:.3f}",
        "HitThreeHigh": f"{intervals['hit_at_3_graph_minus_flat']['ci_95']['high']:.3f}",
    }
    lines = ["% Generated by scripts/export_twinworld_agent_figures.py."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        type=Path,
        default=ROOT
        / "outputs"
        / "instruction_agent"
        / "five_scene_qwen25_05b_graph_v1"
        / "metrics_summary.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT
        / "outputs"
        / "raw_rgbd_hierarchy"
        / "five_scene_clip_v1"
        / "metrics_summary.json",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=ROOT
        / "outputs"
        / "agent_semantic"
        / "five_scene_four_variant_v1"
        / "metrics_summary.json",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=ROOT
        / "docs"
        / "experiments"
        / "hierarchy_construction"
        / "four_variant_v1"
        / "hierarchy_evaluation.json",
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=ROOT
        / "docs"
        / "reports"
        / "final"
        / "twinworld_bootstrap_ci.json",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=ROOT / "papers" / "twinworld",
    )
    args = parser.parse_args()
    agent = read_json(args.agent.resolve())
    raw = read_json(args.raw.resolve())
    controls = read_json(args.controls.resolve())
    hierarchy = read_json(args.hierarchy.resolve())
    bootstrap = read_json(args.bootstrap.resolve())
    figures = args.paper.resolve() / "figures"
    instruction_figure(agent, controls, figures / "fig_instruction_agent")
    hierarchy_figure(raw, figures / "fig_raw_rgbd_hierarchy")
    construction_variants_figure(
        hierarchy,
        figures / "fig_hierarchy_construction_variants",
    )
    workflow_figure(figures / "fig_agent_mcp_workflow")
    tables = args.paper.resolve() / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "new_experiment_macros.tex").write_text(
        macros(agent, raw, hierarchy, bootstrap),
        encoding="utf-8",
    )
    print(f"Wrote figures and macros under {args.paper.resolve()}")


if __name__ == "__main__":
    main()
