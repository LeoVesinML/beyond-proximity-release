#!/usr/bin/env python3
"""Render graph-vs-flat paper figures from metrics_summary.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/graph_vs_flat/five_scene_graph_vs_flat_v2",
        help="Directory with metrics_summary.json",
    )
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    summary_path = run_dir / "metrics_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    import matplotlib.pyplot as plt

    data = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
    averages = data.get("averages", {})
    quality = data.get("quality", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Views", "Tokens", "Runtime (ms)"]
    flat_vals = [
        averages.get("flat_views_checked", 0),
        averages.get("flat_input_tokens", 0),
        averages.get("flat_elapsed_ms", 0),
    ]
    graph_vals = [
        averages.get("graph_views_checked", 0),
        averages.get("graph_input_tokens", 0),
        averages.get("graph_elapsed_ms", 0),
    ]
    x = range(len(labels))
    w = 0.35
    axes[0].bar([i - w / 2 for i in x], flat_vals, width=w, label="Flat")
    axes[0].bar([i + w / 2 for i in x], graph_vals, width=w, label="Graph")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_title("Average cost (lower is better)")
    axes[0].legend()

    if quality.get("hit_at_1_graph") is not None:
        q_labels = ["hit@1", "hit@3"]
        g_vals = [quality.get("hit_at_1_graph", 0), quality.get("hit_at_3_graph", 0)]
        f_vals = [quality.get("hit_at_1_flat", 0), quality.get("hit_at_3_flat", 0)]
        xq = range(len(q_labels))
        axes[1].bar([i - w / 2 for i in xq], g_vals, width=w, label="Graph")
        axes[1].bar([i + w / 2 for i in xq], f_vals, width=w, label="Flat")
        axes[1].set_xticks(list(xq), q_labels)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title(f"Quality (n={data.get('gt_query_count', 0)} GT queries)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No GT metrics", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    out = run_dir / "graph_vs_flat_summary.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
