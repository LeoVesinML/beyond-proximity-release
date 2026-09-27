#!/usr/bin/env python3
"""Export journal figures from the frozen IEEE revision result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#1769aa"
GREEN = "#218354"
ORANGE = "#c54a0b"
INK = "#20262d"
GREY = "#6b7280"
GRID = "#d9dee5"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def frontier(source: dict[str, Any], out: Path) -> None:
    rows = source["frontier"]
    x = np.array([row["mean_records_checked"] for row in rows], dtype=float)
    hit1 = np.array([row["hit_at_1"] for row in rows], dtype=float)
    hit3 = np.array([row["hit_at_3"] for row in rows], dtype=float)
    order = np.argsort(x)
    flat = source["flat_test"]
    selected = source["operating_point"]

    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    ax.plot(x[order], hit1[order], color=BLUE, marker="o", markersize=4.5, label="hierarchical hit@1")
    ax.plot(x[order], hit3[order], color=GREEN, marker="s", markersize=4.2, linestyle="--", label="hierarchical hit@3")
    ax.axhline(flat["hit_at_1"], color=BLUE, linestyle=":", linewidth=1.2, label="flat hit@1")
    ax.axhline(flat["hit_at_3"], color=GREEN, linestyle="-.", linewidth=1.2, label="flat hit@3")
    ax.scatter(
        [selected["mean_records_checked"]], [selected["hit_at_3"]],
        s=75, facecolor="white", edgecolor=ORANGE, linewidth=1.8, zorder=5,
        label="selected M7 point",
    )
    ax.annotate(
        f"M7: {selected['mean_records_checked']:.2f} records\n"
        f"hit@3={selected['hit_at_3']:.3f}",
        (selected["mean_records_checked"], selected["hit_at_3"]),
        xytext=(14, -42), textcoords="offset points", color=INK,
        arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.8},
    )
    ax.set_title("Held-out efficiency--quality frontier")
    ax.set_xlabel("Mean serialized scene records checked per query (lower is better)")
    ax.set_ylabel("Retrieval rate on 75 held-out positives")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.25)
    fig.text(
        0.01, 0.035,
        "Same lexical leaf scorer; 10,000-resample paired bootstrap. Proxy comparisons are accounted separately.",
        color=GREY, fontsize=7.5,
    )
    _save(fig, out, "fig_recall_frontier")


def scorer_frontiers(
    lexical: dict[str, Any], bge: dict[str, Any], out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)
    for ax, source, title in zip(
        axes, (lexical, bge), ("(a) Lexical scorer", "(b) BGE-small scorer")
    ):
        rows = sorted(source["frontier"], key=lambda row: row["mean_records_checked"])
        x = [row["mean_records_checked"] for row in rows]
        ax.plot(x, [row["hit_at_1"] for row in rows], color=BLUE, marker="o", label="hit@1")
        ax.plot(x, [row["hit_at_3"] for row in rows], color=GREEN, marker="s", linestyle="--", label="hit@3")
        ax.axhline(source["flat_test"]["hit_at_1"], color=BLUE, linestyle=":", linewidth=1.1)
        ax.axhline(source["flat_test"]["hit_at_3"], color=GREEN, linestyle="-.", linewidth=1.1)
        point = source["operating_point"]
        ax.scatter(
            point["mean_records_checked"], point["hit_at_3"], s=68,
            facecolor="white", edgecolor=ORANGE, linewidth=1.7, zorder=5,
        )
        ax.annotate(
            f"M7: {point['mean_records_checked']:.2f}",
            (point["mean_records_checked"], point["hit_at_3"]),
            xytext=(8, -22), textcoords="offset points", fontsize=7.3,
            arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.7},
        )
        ax.set_title(title)
        ax.set_xlabel("Mean retained records/query")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Retrieval rate (75 held-out positives)")
    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    fig.subplots_adjust(bottom=0.23, wspace=0.18)
    fig.text(
        0.01, 0.035,
        "Horizontal lines are exhaustive same-scorer controls; rings mark the smallest qualifying M7 budget.",
        color=GREY, fontsize=7.3,
    )
    _save(fig, out, "fig_scorer_frontiers")


def ablations(source: dict[str, Any], out: Path) -> None:
    rows = source["ablations"]
    labels = [row["label"].replace("_", " ") for row in rows]
    y = np.arange(len(rows))[::-1]
    hit3 = np.array([row["hit_at_3"] for row in rows])
    records = np.array([row["mean_records_checked"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), gridspec_kw={"wspace": 0.45})
    axes[0].hlines(y, 0, hit3, color=GRID, linewidth=2)
    axes[0].scatter(hit3, y, color=GREEN, marker="s", s=35, zorder=3)
    for value, ypos in zip(hit3, y):
        axes[0].text(value + 0.012, ypos, f"{value:.3f}", va="center", fontsize=7.5)
    axes[0].axvline(source["flat_test"]["hit_at_3"], color=INK, linestyle=":", linewidth=1.1)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0, 1.02)
    axes[0].set_title("(a) Hit@3 by mitigation stage")
    axes[0].set_xlabel("Hit@3 (n=75)")

    axes[1].hlines(y, 0, records, color=GRID, linewidth=2)
    axes[1].scatter(records, y, color=BLUE, marker="o", s=38, zorder=3)
    for value, ypos in zip(records, y):
        axes[1].text(value + 0.08, ypos, f"{value:.2f}", va="center", fontsize=7.5)
    axes[1].axvline(source["flat_test"]["mean_records_checked"], color=INK, linestyle=":", linewidth=1.1)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xlim(0, source["flat_test"]["mean_records_checked"] * 1.08)
    axes[1].set_title("(b) Retained-record cost")
    axes[1].set_xlabel("Mean records checked (lower is better)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sequential mitigation ablation at the selected budget", y=1.02, fontsize=10)
    _save(fig, out, "fig_mitigation_ablation")


def failure_taxonomy(source: dict[str, Any], out: Path) -> None:
    counts = source["failure_taxonomy"]["category_counts"]
    if len(counts) < 2:
        return
    labels = [key.replace("_", " ") for key in counts]
    values = np.array([counts[key] for key in counts], dtype=float)
    order = np.argsort(values)
    labels = [labels[index] for index in order]
    values = values[order]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.hlines(y, 0, values, color=GRID, linewidth=3)
    ax.scatter(values, y, color=ORANGE, marker="D", s=38, zorder=3)
    for value, ypos in zip(values, y):
        ax.text(value + 0.12, ypos, f"{int(value)}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Held-out hit@3 misses")
    ax.set_title("Branch-gate failure taxonomy")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_failure_taxonomy")


def public_scale(source: dict[str, Any], out: Path) -> None:
    rows = source["summaries"]
    methods = {
        "flat_dense_exact": ("Flat exact", BLUE, "o", "-"),
        "scene_scoped_dense_exact": ("Scene-scoped exact", GREEN, "s", "--"),
        "hnsw_dense": ("HNSW", ORANGE, "D", "-."),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), gridspec_kw={"wspace": 0.43})
    for method, (label, color, marker, line) in methods.items():
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["collection_size"])
        x = [row["collection_size"] for row in selected]
        axes[0].plot(x, [row["hit_at_3"] for row in selected], label=label, color=color, marker=marker, linestyle=line)
        axes[1].plot(x, [row["median_latency_ms"] for row in selected], label=label, color=color, marker=marker, linestyle=line)
        axes[2].plot(x, [row["mean_scored_records"] for row in selected], label=label, color=color, marker=marker, linestyle=line)
    axes[0].set_title("(a) Retrieval quality")
    axes[0].set_xlabel("Indexed ReferIt3D records")
    axes[0].set_ylabel("Hit@3")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title("(b) Query latency")
    axes[1].set_xlabel("Indexed ReferIt3D records")
    axes[1].set_ylabel("Median CPU latency (ms)")
    axes[1].set_yscale("log")
    axes[2].set_title("(c) Scoring work")
    axes[2].set_xlabel("Indexed ReferIt3D records")
    axes[2].set_ylabel("Records scored/query")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.subplots_adjust(bottom=0.30, wspace=0.43)
    fig.text(
        0.01, 0.025,
        "Query record excluded; relevant hits are different expressions for the same ScanNet scene/object. Scene-scoped search assumes known scan ID.",
        color=GREY, fontsize=7.3,
    )
    _save(fig, out, "fig_public_scale_ann")


def predicted_maps(source: dict[str, Any], out: Path) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source["summaries"]:
        dataset = "ScanNet" if row["query_count"] == 48 else "Replica"
        grouped.setdefault(dataset, {})[row["method"]] = row
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.38})
    y = np.arange(len(grouped))
    names = list(grouped)
    flat_quality = [grouped[name]["flat_bge_predicted_map"]["acc_at_0_25"] for name in names]
    graph_quality = [grouped[name]["graph_bge_predicted_map"]["acc_at_0_25"] for name in names]
    axes[0].scatter(flat_quality, y + 0.08, color=GREY, marker="o", s=42, label="flat")
    axes[0].scatter(graph_quality, y - 0.08, facecolor="white", edgecolor=GREEN, marker="s", s=48, linewidth=1.5, label="graph")
    for index, value in enumerate(graph_quality):
        axes[0].text(value + 0.008, index, f"{value:.3f}", va="center", fontsize=7.5)
    axes[0].set_yticks(y, [f"{name} (n={grouped[name]['flat_bge_predicted_map']['query_count']})" for name in names])
    axes[0].set_xlim(0, max(0.20, max(graph_quality) + 0.04))
    axes[0].set_xlabel("3D Acc@0.25")
    axes[0].set_title("(a) Same predicted maps and scorer")
    axes[0].legend(frameon=False, loc="upper right")

    flat_checked = [grouped[name]["flat_bge_predicted_map"]["mean_checked_objects"] for name in names]
    graph_checked = [grouped[name]["graph_bge_predicted_map"]["mean_checked_objects"] for name in names]
    for index in range(len(names)):
        axes[1].plot([graph_checked[index], flat_checked[index]], [index, index], color=GRID, linewidth=2.5)
    axes[1].scatter(flat_checked, y, color=GREY, marker="o", s=42, label="flat")
    axes[1].scatter(graph_checked, y, color=GREEN, marker="s", s=42, label="graph")
    axes[1].set_yticks(y, [""] * len(names))
    axes[1].set_xlabel("Downstream objects retained/query")
    axes[1].set_title("(b) Candidate reduction")
    axes[1].legend(frameon=False, loc="center right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.01, 0.035,
        "Both methods compare every cached BGE object proxy; the graph reduces retained downstream evidence, not proxy scoring.",
        color=GREY, fontsize=7.3,
    )
    _save(fig, out, "fig_predicted_map_pair")


def sensitivity(source: dict[str, Any], out: Path) -> None:
    titles = {
        "branch_mass": "Branch mass threshold",
        "temperature": "Softmax temperature",
        "branch_cap": "Maximum branches",
        "safety_shortlist": "Safety shortlist size",
        "candidate_budget": "Candidate budget",
        "boundary_overlap": "Boundary overlap",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.0), gridspec_kw={"hspace": 0.48, "wspace": 0.33})
    for ax, (group, title) in zip(axes.flat, titles.items()):
        rows = [row for row in source["sweeps"] if row["group"] == group]
        x = np.arange(len(rows))
        y = [row["hit_at_3"] for row in rows]
        ax.plot(x, y, color=GREY, linewidth=1.0)
        ax.scatter(x, y, color=BLUE, s=27, zorder=3)
        ax.set_xticks(x, [str(row["value"]) for row in rows])
        ax.set_title(title)
        ax.set_xlabel("Swept value")
        ax.set_ylabel("Dev hit@3")
        ax.set_ylim(0.68, 1.01)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Development-set sensitivity around the selected operating point (n=50)", fontsize=10, y=0.995)
    _save(fig, out, "fig_sensitivity")


def graphical_abstract(
    lexical: dict[str, Any], bge: dict[str, Any], scale: dict[str, Any], out: Path
) -> None:
    fig = plt.figure(figsize=(7.15, 3.45))
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.80])
    ax.set_axis_off()

    def box(x: float, y: float, width: float, text: str, color: str) -> None:
        ax.text(
            x + width / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.3,
            color=INK,
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": color, "lw": 1.5},
        )

    y = 0.67
    boxes = [
        (0.02, 0.20, "Scene records\nRGB-D, objects, text", BLUE),
        (0.29, 0.20, "Adaptive branch mass\n+ max-leaf proxy", ORANGE),
        (0.56, 0.17, "Safety shortlist\n+ budget 3", GREEN),
        (0.80, 0.18, "Shared leaf scorer\n+ trace", BLUE),
    ]
    for x, width, label, color in boxes:
        box(x, y, width, label, color)
    for start, end in ((0.22, 0.29), (0.49, 0.56), (0.73, 0.80)):
        ax.annotate("", xy=(end, y), xytext=(start, y), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": GREY})

    lex = lexical["operating_point"]
    bg = bge["operating_point"]
    rows = [
        ("Lexical", lex["hit_at_3"], 100 * lex["record_savings_vs_flat"], 100 * lex["context_token_savings_vs_flat"]),
        ("BGE-small", bg["hit_at_3"], 100 * bg["record_savings_vs_flat"], 100 * bg["context_token_savings_vs_flat"]),
    ]
    ax.text(0.03, 0.36, "Held-out same-scorer result (n=75)", weight="bold", fontsize=9.2, color=INK)
    ax.text(0.03, 0.25, "Scorer", weight="bold", fontsize=8.1)
    ax.text(0.20, 0.25, "hit@3 = flat", weight="bold", fontsize=8.1)
    ax.text(0.39, 0.25, "records saved", weight="bold", fontsize=8.1)
    ax.text(0.58, 0.25, "context saved", weight="bold", fontsize=8.1)
    for index, (name, hit3, record, context) in enumerate(rows):
        yy = 0.16 - 0.09 * index
        ax.text(0.03, yy, name, fontsize=8.3)
        ax.text(0.20, yy, f"{hit3:.3f}", fontsize=8.3, color=GREEN, weight="bold")
        ax.text(0.39, yy, f"{record:.1f}%", fontsize=8.3, color=BLUE, weight="bold")
        ax.text(0.58, yy, f"{context:.1f}%", fontsize=8.3, color=ORANGE, weight="bold")

    largest = int(scale["collection_sizes"][-1])
    scene_count = int(scale["data_manifest"]["scene_count"])
    ax.text(0.77, 0.30, "Measured public scale", weight="bold", fontsize=9.0, color=INK)
    ax.text(0.77, 0.20, f"{largest:,} records", fontsize=8.5)
    ax.text(0.77, 0.12, f"{scene_count:,} ScanNet scenes", fontsize=8.5)
    ax.text(0.77, 0.04, "Dense exact + HNSW controls", fontsize=8.0, color=GREY)
    fig.suptitle("Beyond Proximity: recall-controlled context reduction with explicit quality controls", fontsize=10.5, y=0.99)
    _save(fig, out, "graphical_abstract")


def instruction_pair(source: dict[str, Any], out: Path) -> None:
    flat = source["methods"]["flat_instruction"]
    graph = source["methods"]["graph_instruction"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.48})

    quality_labels = ["hit@3", "hit@1", "MRR"]
    quality_keys = ["hit_at_3", "hit_at_1", "mrr"]
    y = np.arange(len(quality_labels))[::-1]
    for position, key in zip(y, quality_keys):
        axes[0].plot([graph[key], flat[key]], [position, position], color=GRID, linewidth=3, zorder=1)
    axes[0].scatter([graph[key] for key in quality_keys], y, marker="s", s=48, color=GREEN, label="Graph", zorder=3)
    axes[0].scatter([flat[key] for key in quality_keys], y, marker="o", s=48, color=BLUE, label="Flat", zorder=3)
    for position, key in zip(y, quality_keys):
        axes[0].text(graph[key] - 0.007, position + 0.14, f"{graph[key]:.3f}", ha="right", fontsize=7.2, color=GREEN)
        axes[0].text(flat[key] + 0.007, position + 0.14, f"{flat[key]:.3f}", ha="left", fontsize=7.2, color=BLUE)
    axes[0].set_yticks(y, quality_labels)
    axes[0].set_xlim(0.64, 0.94)
    axes[0].set_xlabel("Score on 125 labeled queries")
    axes[0].set_title("(a) Retrieval quality")

    resource_labels = ["Views", "Input tokens", "CPU latency"]
    resource_keys = ["mean_views_checked", "mean_input_tokens", "mean_latency_ms"]
    ratios = [100.0 * graph[key] / flat[key] for key in resource_keys]
    y2 = np.arange(len(resource_labels))[::-1]
    for position, ratio in zip(y2, ratios):
        axes[1].plot([ratio, 100.0], [position, position], color=GRID, linewidth=3, zorder=1)
    axes[1].scatter(ratios, y2, marker="s", s=48, color=GREEN, label="Graph", zorder=3)
    axes[1].scatter([100.0] * len(y2), y2, marker="o", s=48, color=BLUE, label="Flat", zorder=3)
    for position, ratio in zip(y2, ratios):
        axes[1].text(ratio - 2.0, position + 0.13, f"{ratio:.1f}%", ha="right", fontsize=7.2, color=GREEN)
    axes[1].set_yticks(y2, resource_labels)
    axes[1].set_xlim(45, 108)
    axes[1].set_xlabel("Graph cost relative to flat (flat = 100%)")
    axes[1].set_title("(b) Measured query cost")

    for ax in axes:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], color=GREEN, marker="s", linestyle="None", label="Graph"),
        plt.Line2D([], [], color=BLUE, marker="o", linestyle="None", label="Flat"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(top=0.75, bottom=0.24)
    _save(fig, out, "fig_instruction_pair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, default=ROOT / "outputs/ieee_revision/recall_frontier_v1/frontier_summary.json")
    parser.add_argument("--bge", type=Path, default=ROOT / "outputs/ieee_revision/bge_frontier_v1/bge_frontier_summary.json")
    parser.add_argument("--scale", type=Path, default=ROOT / "outputs/ieee_revision/public_scale_ann_v1/scale_ann_summary.json")
    parser.add_argument("--predicted", type=Path, default=ROOT / "outputs/ieee_revision/predicted_map_pair_v1/predicted_map_pair_summary.json")
    parser.add_argument("--sensitivity", type=Path, default=ROOT / "outputs/ieee_revision/sensitivity_v1/sensitivity_summary.json")
    parser.add_argument("--instruction", type=Path, default=ROOT / "outputs/ieee_revision/qwen25_15b_pair_v1/metrics_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "papers/ieee-access/figures")
    args = parser.parse_args()
    _style()
    frontier_source = _read(args.frontier)
    frontier(frontier_source, args.out)
    if args.bge.is_file():
        scorer_frontiers(frontier_source, _read(args.bge), args.out)
    ablations(frontier_source, args.out)
    failure_taxonomy(frontier_source, args.out)
    if args.scale.is_file():
        scale_source = _read(args.scale)
        public_scale(scale_source, args.out)
    if args.predicted.is_file():
        predicted_maps(_read(args.predicted), args.out)
    if args.sensitivity.is_file():
        sensitivity(_read(args.sensitivity), args.out)
    if args.instruction.is_file():
        instruction_pair(_read(args.instruction), args.out)
    if args.bge.is_file() and args.scale.is_file():
        graphical_abstract(frontier_source, _read(args.bge), scale_source, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
