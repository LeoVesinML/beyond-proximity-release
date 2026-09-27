#!/usr/bin/env python3
"""Compute deterministic paired bootstrap intervals for the TwinWorld paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {"low": round(float(low), 4), "high": round(float(high), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/per_query_results.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/reports/final/twinworld_bootstrap_ci.json",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_715)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("queries") or payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Per-query input has no rows")
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(rows), size=(args.samples, len(rows)))
    flat_views = np.asarray([row["flat"]["views_checked"] for row in rows], dtype=float)
    graph_views = np.asarray([row["graph"]["views_checked"] for row in rows], dtype=float)
    flat_tokens = np.asarray([row["flat"]["input_tokens"] for row in rows], dtype=float)
    graph_tokens = np.asarray([row["graph"]["input_tokens"] for row in rows], dtype=float)
    view_savings_per_query = np.asarray(
        [row["savings_graph_vs_flat"]["views_pct"] for row in rows], dtype=float
    )
    token_savings_per_query = np.asarray(
        [row["savings_graph_vs_flat"]["tokens_pct"] for row in rows], dtype=float
    )
    view_savings = view_savings_per_query[indices].mean(axis=1)
    token_savings = token_savings_per_query[indices].mean(axis=1)

    eligible = [row for row in rows if row["quality_graph"]["gt_available"]]
    quality_indices = rng.integers(0, len(eligible), size=(args.samples, len(eligible)))
    graph_hit1 = np.asarray([row["quality_graph"]["hit_at_1"] for row in eligible], dtype=float)
    flat_hit1 = np.asarray([row["quality_flat"]["hit_at_1"] for row in eligible], dtype=float)
    graph_hit3 = np.asarray([row["quality_graph"]["hit_at_3"] for row in eligible], dtype=float)
    flat_hit3 = np.asarray([row["quality_flat"]["hit_at_3"] for row in eligible], dtype=float)
    hit1_delta = (graph_hit1[quality_indices] - flat_hit1[quality_indices]).mean(axis=1)
    hit3_delta = (graph_hit3[quality_indices] - flat_hit3[quality_indices]).mean(axis=1)

    result = {
        "schema_version": "semanticsplat.bootstrap_ci.v1",
        "source": str(args.input),
        "seed": args.seed,
        "bootstrap_samples": args.samples,
        "query_count": len(rows),
        "quality_query_count": len(eligible),
        "method": "paired nonparametric query bootstrap; percentile 95% interval",
        "metrics": {
            "view_savings_pct": {
                "estimate": round(float(view_savings_per_query.mean()), 4),
                "ci_95": interval(view_savings),
            },
            "token_savings_pct": {
                "estimate": round(float(token_savings_per_query.mean()), 4),
                "ci_95": interval(token_savings),
            },
            "hit_at_1_graph_minus_flat": {
                "estimate": round(float((graph_hit1 - flat_hit1).mean()), 4),
                "ci_95": interval(hit1_delta),
            },
            "hit_at_3_graph_minus_flat": {
                "estimate": round(float((graph_hit3 - flat_hit3).mean()), 4),
                "ci_95": interval(hit3_delta),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
