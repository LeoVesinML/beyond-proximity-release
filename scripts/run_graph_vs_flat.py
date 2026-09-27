#!/usr/bin/env python3
"""Run same-input graph-vs-flat benchmark on five captured semantic indexes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (  # noqa: E402
    DEFAULT_SCENE_MAP,
    run_five_scene_benchmark,
    write_run_outputs,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/graph_vs_flat_five_scenes.yaml",
        help="Run config path",
    )
    parser.add_argument("--benchmark", help="Override benchmark JSON path")
    parser.add_argument("--scene-root", help="Override scene root directory")
    parser.add_argument("--out", help="Override output directory")
    parser.add_argument("--run-id", help="Optional run id subdirectory name")
    parser.add_argument("--no-affordance", action="store_true", help="Skip graph_affordance mode")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(config_path) if config_path.exists() else {}

    benchmark_path = ROOT / (args.benchmark or config.get("benchmark_path", "docs/benchmarks/benchmark_queries_v1.json"))
    scene_root = ROOT / (args.scene_root or config.get("scene_root", "backend/data/scenes"))
    output_root = ROOT / (args.out or config.get("output_root", "outputs/graph_vs_flat"))
    run_id = args.run_id or config.get("run_id") or f"five_scene_{utc_stamp()}"
    scene_map = config.get("scene_map") or DEFAULT_SCENE_MAP
    include_affordance = not args.no_affordance and config.get("include_affordance", True)

    out_dir = output_root / run_id
    run = run_five_scene_benchmark(
        benchmark_path=benchmark_path,
        scene_root=scene_root,
        scene_map=scene_map,
        include_affordance=include_affordance,
    )
    run["run_id"] = run_id
    run["git_revision"] = git_revision()
    run["config_path"] = str(config_path)
    write_run_outputs(run, out_dir)

    summary = run["summary"]
    print(f"Wrote graph-vs-flat run to {out_dir}")
    print(f"Queries: {summary.get('query_count', 0)}")
    print(f"Avg view savings: {summary.get('averages', {}).get('savings_views_pct')}%")
    print(f"Avg token savings: {summary.get('averages', {}).get('savings_tokens_pct')}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
