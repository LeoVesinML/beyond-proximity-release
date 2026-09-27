#!/usr/bin/env python3
"""Export the revision-2 journal figures from the frozen result files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from export_ieee_revision_figures import BLUE, GREEN, GREY, INK, ORANGE, _save, _style

ROOT = Path(__file__).resolve().parents[1]
REV2 = ROOT / "outputs/ieee_revision2"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scorer_frontiers(out: Path) -> None:
    summary = _read(REV2 / "frontier/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), sharey=True)
    for ax, (name, title) in zip(axes, (("lexical", "(a) Lexical scorer"), ("bge", "(b) BGE-small scorer"))):
        test = summary[name]["test"]
        ks = sorted(test["budgets"], key=int)
        algo = [test["budgets"][k]["algorithm1"] for k in ks]
        topk = [test["budgets"][k]["flat_top_k"] for k in ks]
        x_algo = [row["retained"] for row in algo]
        x_topk = [row["retained"] for row in topk]
        for metric, color, label in (("hit_at_1", BLUE, "hit@1"), ("hit_at_3", GREEN, "hit@3")):
            ax.plot(x_algo, [row[metric] for row in algo], color=color, marker="o", ms=3.5, lw=1.3,
                    label=f"Algorithm 1, {label}")
            ax.plot(x_topk, [row[metric] for row in topk], color=color, marker="x", ms=4.5, lw=0.9,
                    ls="--", label=f"flat top-$k$, {label}")
            ax.axhline(test["flat"][metric], color=color, lw=0.8, ls=":")
        sel = ks.index("3")
        ax.scatter([x_algo[sel]], [algo[sel]["hit_at_3"]], s=120, facecolors="none", edgecolors=ORANGE,
                   linewidths=1.4, zorder=5)
        ax.annotate("$k=3$", (x_algo[sel], algo[sel]["hit_at_3"]), xytext=(8, -16),
                    textcoords="offset points", color=ORANGE, fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel("Mean retained records/query")
        ax.set_ylim(0.55, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out retrieval rate ($n=75$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.17, "Dotted lines: exhaustive scoring of all 19.4 records. Ring: operating point $k=3$.",
             ha="center", fontsize=7.5, color=GREY)
    _save(fig, out, "fig_scorer_frontiers")


def ablation(out: Path) -> None:
    summary = _read(REV2 / "frontier/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.4), sharey=True)
    for ax, (name, title) in zip(axes, (("lexical", "(a) Lexical scorer"), ("bge", "(b) BGE-small scorer"))):
        dev = summary[name]["ablation_dev"]
        test = summary[name]["ablation_test"]
        labels = [row["stage"].replace(" (Algorithm 1)", "\n(Algorithm 1)") for row in dev]
        y = np.arange(len(labels))[::-1]
        ax.barh(y + 0.19, [row["hit_at_3"] for row in dev], height=0.36, color=BLUE, label="development ($n=50$)")
        ax.barh(y - 0.19, [row["hit_at_3"] for row in test], height=0.36, color=GREEN, label="held-out test ($n=75$)")
        for yy, row in zip(y, dev):
            ax.text(row["hit_at_3"] + 0.01, yy + 0.19, f"{row['hit_at_3']:.3f}", va="center", fontsize=7)
        for yy, row in zip(y, test):
            ax.text(row["hit_at_3"] + 0.01, yy - 0.19, f"{row['hit_at_3']:.3f}", va="center", fontsize=7)
        ax.set_yticks(y, labels)
        ax.set_xlim(0.5, 1.07)
        ax.set_xlabel("hit@3 at $k=3$")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.28), ncol=2, frameon=False)
    _save(fig, out, "fig_mitigation_ablation")


def sensitivity(out: Path) -> None:
    rows = list(csv.DictReader((REV2 / "sensitivity/sensitivity.csv").open()))
    selected = {"branch_mass": "0.7", "temperature": "0.25", "branch_cap": "5", "safety_shortlist": "1",
                "candidate_budget": "3", "boundary_overlap": "0.0"}
    titles = {"branch_mass": r"Mass target $\tau$", "temperature": "Temperature $T$",
              "branch_cap": "Branch cap $B$", "safety_shortlist": "Shortlist size $m$",
              "candidate_budget": "Budget $k$", "boundary_overlap": r"Boundary width $\epsilon$"}
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.3), gridspec_kw={"hspace": 0.6, "wspace": 0.45})
    for ax, (group, title) in zip(axes.flat, titles.items()):
        g = [row for row in rows if row["group"] == group]
        x = np.arange(len(g))
        y = [float(row["hit_at_3"]) for row in g]
        cost = [float(row["mean_records_checked"]) for row in g]
        ax.plot(x, y, color=GREY, lw=1.0)
        ax.scatter(x, y, color=BLUE, s=22, zorder=3)
        for i, row in enumerate(g):
            if float(row["value"]) == float(selected[group]):
                ax.scatter([i], [y[i]], s=110, facecolors="none", edgecolors=ORANGE, linewidths=1.4, zorder=4)
        ax.set_xticks(x, [row["value"] for row in g])
        ax.set_ylim(0.6, 0.92)
        ax.set_title(title)
        if ax in axes[:, 0]:
            ax.set_ylabel("dev hit@3", color=BLUE)
        ax2 = ax.twinx()
        ax2.plot(x, cost, color=GREEN, lw=0.9, ls="--", marker="s", ms=2.5)
        ax2.set_ylim(0, 10)
        if ax in axes[:, -1]:
            ax2.set_ylabel("retained records", color=GREEN)
        ax.spines[["top"]].set_visible(False)
        ax2.spines[["top"]].set_visible(False)
        ax2.grid(False)
    _save(fig, out, "fig_sensitivity")


def scene_gallery(out: Path) -> None:
    scenes = [("ConferenceHall-capture-pilot", "Conference hall"), ("Museume-capture", "Museum"),
              ("Theater-capture", "Theater"), ("outdoor-street-capture", "Outdoor street"),
              ("outdoor-drone-capture", "Outdoor drone flight")]
    fig, axes = plt.subplots(1, 5, figsize=(7.15, 1.55))
    for ax, (scene, title) in zip(axes, scenes):
        images = sorted((ROOT / "backend/data/scenes" / scene / "images").glob("*.png"))
        image = Image.open(images[len(images) // 2]).convert("RGB")
        image.thumbnail((900, 450))
        ax.imshow(image)
        ax.set_title(f"{title}\n{len(images)} views", fontsize=7)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04)
    _save(fig, out, "fig_scene_gallery")


def predicted_maps(out: Path) -> None:
    path = REV2 / "predicted_maps/figure_data.json"
    if not path.is_file():
        print("skip predicted maps: no figure_data.json")
        return
    data = _read(path)
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    rows = data["rows"]
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [row["value"] for row in rows], color=[row.get("color", BLUE) for row in rows])
    for yy, row in zip(y, rows):
        ax.text(row["value"] + 0.01, yy, row["text"], va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(data["xlabel"])
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_predicted_map_pair")


def structure(out: Path) -> None:
    rows = {row["hierarchy"]: row for row in csv.DictReader((REV2 / "hierarchy/structural_agreement.csv").open())}
    names = [("pose_only", "Pose only"), ("pose_semantic_merge", "Pose+semantic"), ("raw_rgbd", "Raw RGB-D")]
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    y = np.arange(len(names))[::-1]
    for yy, (key, label) in zip(y, names):
        row = rows[key]
        ax.plot([float(row["random_size_matched_p2_5"]), float(row["random_size_matched_p97_5"])], [yy, yy],
                color=GREY, lw=4, alpha=0.5, solid_capstyle="butt", label="size-matched random (95%)" if key == "pose_only" else None)
        ax.scatter([float(row["macro_pairwise_f1"])], [yy], color=BLUE, s=30, zorder=3, label="observed" if key == "pose_only" else None)
        ax.text(float(row["macro_pairwise_f1"]) + 0.02, yy + 0.18, f"{float(row['macro_pairwise_f1']):.3f}", fontsize=7)
    ref = float(rows["disjoint_manual_projection"]["macro_pairwise_f1"])
    ax.axvline(ref, color=ORANGE, ls="--", lw=1.0, label=f"best disjoint reference ({ref:.3f})")
    ax.set_yticks(y, [label for _, label in names])
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Macro pairwise F1 against manual zones")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.42), ncol=2, frameon=False, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_structure_controls")


def public_scale(out: Path) -> None:
    rows = list(csv.DictReader((REV2 / "scale_hierarchy/scale_hierarchy_curves.csv").open()))
    ann = list(csv.DictReader((REV2 / "scale_hierarchy/reproduction_public_scale_ann_v1/scale_ann_summary.csv").open()))
    series = [
        ("flat exact", [r for r in rows if r["method"] == "flat_dense_exact"], "hit_at_3", "mean_scored_records", INK, "o", "-"),
        ("scene hierarchy (Alg. 1, summary gate)", [r for r in rows if r["method"] == "hier_summary_budget_1pp"], "hit_at_3", "mean_scored_records", GREEN, "s", "-"),
        ("scene hierarchy, no shortlist", [r for r in rows if r["method"] == "hier_summary_gate_only"], "hit_at_3", "mean_scored_records", GREEN, "v", ":"),
        ("HNSW", [r for r in ann if r["method"] == "hnsw_dense"], "hit_at_3", None, BLUE, "^", "--"),
        ("known scene (oracle)", [r for r in ann if r["method"] == "scene_scoped_dense_exact"], "hit_at_3", "mean_scored_records", ORANGE, "D", "-."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5))
    for label, rs, q, cost, color, marker, ls in series:
        rs = sorted(rs, key=lambda r: int(r["collection_size"]))
        x = [int(r["collection_size"]) for r in rs]
        axes[0].plot(x, [float(r[q]) for r in rs], color=color, marker=marker, ms=3.5, ls=ls, lw=1.1, label=label)
        if cost:
            axes[1].plot(x, [float(r[cost]) for r in rs], color=color, marker=marker, ms=3.5, ls=ls, lw=1.1)
    axes[0].set_ylabel("hit@3")
    axes[1].set_ylabel("records scored per query")
    axes[1].set_yscale("log")
    for ax, title in zip(axes, ("(a) Retrieval quality", "(b) Similarity computations")):
        ax.set_xscale("log")
        ax.set_xlabel("Indexed ReferIt3D expressions")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, out, "fig_public_scale_ann")


def llm_cascade(out: Path) -> None:
    path = REV2 / "llm_cascade/qwen25_7b_bf16/summary_a2.json"
    res = _read(path)["results"]["test75"]
    conds = [("exhaustive__full", "no selection (all records)", INK, "o"),
             ("flat_bge_topN_matched__full", "flat BGE top-$N_q$", BLUE, "^"),
             ("alg1_bge__full", "Algorithm 1 (summary gate)", GREEN, "s"),
             ("alg1_bge_maxleaf__full", "Algorithm 1 (max-leaf gate)", GREEN, "D"),
             ("flat_bge_top3__full", "flat BGE top-3", BLUE, "x"),
             ("alg1_bge_budget3__full", "Algorithm 1, $k=3$", GREEN, "P"),
             ("gate_llm__full", "LLM two-branch gate", ORANGE, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5))
    for key, label, color, marker in conds:
        r = res[key]
        axes[0].scatter(r["mean_input_tokens"], r["hit_at_3"], color=color, marker=marker, s=34, label=label, zorder=3)
        axes[1].scatter(r["median_end_to_end_ms"] / 1000, r["hit_at_1"], color=color, marker=marker, s=34, zorder=3)
    axes[0].set_xlabel("Qwen input tokens per query")
    axes[0].set_ylabel("hit@3")
    axes[1].set_xlabel("median latency per query (s, MPS)")
    axes[1].set_ylabel("hit@1")
    axes[0].set_title("(a) Recall versus context")
    axes[1].set_title("(b) Top-1 accuracy versus latency")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=6.8)
    _save(fig, out, "fig_llm_cascade")

def graphical_abstract(out: Path) -> None:
    res = _read(REV2 / "llm_cascade/qwen25_7b_bf16/summary_a2.json")["results"]["test75"]
    fig = plt.figure(figsize=(7.0, 3.3))
    fig.text(0.5, 0.95, "How much of a 3D scene should a language model read?", ha="center", fontsize=12, weight="bold")
    boxes = [(0.02, "Scene records\n(views, objects,\nexpressions)", BLUE), (0.27, "Cheap proxy\n(BGE ranking)", GREEN),
             (0.52, "Optional hierarchy gate\n+ safety shortlist", GREY), (0.77, "LLM reads only\nretained records", ORANGE)]
    for x, text, color in boxes:
        fig.text(x + 0.1, 0.74, text, ha="center", va="center", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1.6))
    for x in (0.235, 0.49, 0.752):
        fig.text(x, 0.74, "$\\rightarrow$", ha="center", va="center", fontsize=14, color=GREY)
    ax = fig.add_axes([0.08, 0.1, 0.38, 0.42])
    keys = [("exhaustive__full", "all records"), ("flat_bge_topN_matched__full", "flat cascade"),
            ("alg1_bge__full", "hierarchy"), ("flat_bge_top3__full", "flat top-3")]
    toks = [res[k]["mean_input_tokens"] for k, _ in keys]
    ax.bar(range(4), toks, color=[INK, BLUE, GREEN, BLUE])
    for i, (k, _) in enumerate(keys):
        ax.text(i, toks[i] + 60, f"hit@3 {res[k]['hit_at_3']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4), [l for _, l in keys], fontsize=7)
    ax.set_ylabel("LLM input tokens", fontsize=8)
    ax.set_ylim(0, 2700)
    ax.set_title("Qwen2.5-7B: half the context, hit@3 within 0.03", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.55, 0.47, "Findings", fontsize=9.5, weight="bold")
    lines = ["Answer accuracy 0.61 -> 0.73 with 3 records",
             "Hierarchy = flat cascade at matched budget",
             "Safety shortlist makes the hierarchy interchangeable",
             "At 155k records, 96.6% of flat misses are other-scene",
             "Replica GT orientation error found and corrected"]
    for i, line in enumerate(lines):
        fig.text(0.55, 0.39 - 0.07 * i, "- " + line, fontsize=8)
    _save(fig, out, "graphical_abstract")

def two_branch_gate(out: Path) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from backend.query.recall_controlled import wilson_interval
    d = _read(REV2 / "hierarchy/e3_two_branch_ci.json")
    g = d["recomputed_gate_means"]
    rows = [("flat_clip", "Flat (all views)", INK), ("pose_only_clip", "Pose only", BLUE),
            ("pose_semantic_merge_clip", "Pose+semantic", BLUE), ("manual_hierarchy_clip", "Manual zones", ORANGE),
            ("raw_rgbd_hierarchy_clip", "Raw RGB-D", GREEN)]
    n = d["n"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.3), gridspec_kw={"width_ratios": [2.2, 1]})
    y = np.arange(len(rows))[::-1]
    for yy, (key, label, color) in zip(y, rows):
        for metric, off, marker in (("hit_at_1", 0.15, "o"), ("hit_at_3", -0.15, "s")):
            v = g[key][metric]
            lo, hi = wilson_interval(round(v * n), n)
            axes[0].errorbar([v], [yy + off], xerr=[[v - lo], [hi - v]], fmt=marker, color=color, ms=4.5, capsize=2, lw=1)
            axes[0].text(hi + 0.01, yy + off, f"{v:.3f}", va="center", fontsize=6.8)
        axes[1].barh(yy, g[key]["mean_views_checked"], color=color, height=0.6)
        axes[1].text(g[key]["mean_views_checked"] + 0.3, yy, f"{g[key]['mean_views_checked']:.1f}", va="center", fontsize=6.8)
    axes[0].set_yticks(y, [label for _, label, _ in rows])
    axes[0].set_xlim(0.2, 0.85)
    axes[0].set_xlabel("hit@1 (circles) and hit@3 (squares), Wilson 95%")
    axes[0].set_title("(a) Retrieval quality")
    axes[1].set_yticks(y, [""] * len(rows))
    axes[1].set_xlim(0, 23)
    axes[1].set_xlabel("views scored per query")
    axes[1].set_title("(b) Cost")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_raw_rgbd_hierarchy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "papers/ieee-access/figures")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    _style()
    jobs = {"frontier": scorer_frontiers, "ablation": ablation, "sensitivity": sensitivity,
            "gallery": scene_gallery, "structure": structure, "scale": public_scale, "llm": llm_cascade, "abstract": graphical_abstract, "twobranch": two_branch_gate}
    for name, job in jobs.items():
        if not args.only or name in args.only:
            job(args.out)


if __name__ == "__main__":
    main()
