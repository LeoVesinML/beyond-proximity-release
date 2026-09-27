#!/usr/bin/env python3
"""Graph vs flat benchmark — tokens, views, calls, estimated seconds.

Usage:
    PYTHONPATH=. python scripts/run_full_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fix_windows_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

# Writable matplotlib cache (avoids font-cache errors in CI/sandbox)
_mpl_dir = ROOT / ".matplotlib"
_mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

OUT_DIR = ROOT / "docs" / "benchmark_results"
DOCS_DIR = ROOT / "docs"
MD_PATH = DOCS_DIR / "benchmarks" / "benchmark_graph_vs_flat.md"


def _short_label(query: str, max_len: int = 27) -> str:
    q = query.strip()
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


def _fmt_times(ratio: Optional[float], label: str = "less") -> str:
    if ratio is None:
        return "—"
    return f"{ratio}× {label}"


def print_table(report: dict) -> None:
    s = report["summary"]
    tv = report["total_views"]
    gs = report.get("graph_stats") or {}
    tm = report.get("timing_model") or {}

    print()
    print("=" * 115)
    print(f"  BENCHMARK (resources & speed) — scene '{report['scene_id']}' ({tv} views)")
    print("=" * 115)

    if gs.get("has_tree"):
        print()
        print("  GRAPH SIZE:")
        print(
            f"    {gs['node_count']} nodes  |  depth {gs['max_depth']}  |  "
            f"{gs.get('leaf_count', 0)} leaves  |  {gs.get('zone_count', 0)} zones  |  "
            f"{gs.get('views_in_tree', tv)} views in tree"
        )

    print()
    print("  TIMING MODEL (fast LLM, no thinking):")
    print(f"    {tm.get('formula', 'total_sec = infra + llm')}")
    print(f"    {tm.get('sec_per_call', 0.35)} s/call + {tm.get('tokens_per_sec', 140)} tok/s")
    print()
    print("  AVERAGE — graph vs flat:")
    print(f"    Tokens:      {s['avg_graph_tokens']:,} vs {s['avg_flat_tokens']:,}  →  {_fmt_times(s['avg_tokens_times_less'])}")
    print(f"    Views:       {s['avg_graph_views_checked']} vs {tv}  →  {_fmt_times(s['avg_views_times_faster'], 'faster')}")
    avg_g_calls = round(sum(r["graph"]["llm_calls"] for r in report["results"]) / len(report["results"]), 1)
    print(f"    LLM calls:   {avg_g_calls} vs 20  →  {_fmt_times(s['avg_calls_times_less'])}")
    print(
        f"    Time (est):  {s['avg_graph_total_sec']} s vs {s['avg_flat_total_sec']} s  →  "
        f"{_fmt_times(s.get('avg_sec_times_faster'), 'faster')}"
    )
    print()
    print("  AVERAGE — graph oracle (real LLM path) vs flat:")
    print(f"    Tokens:      — vs {s['avg_flat_tokens']:,}  →  {_fmt_times(s['oracle_avg_tokens_times_less'])}")
    print(f"    Views:       — vs {tv}  →  {_fmt_times(s['oracle_avg_views_times_faster'], 'faster')}")
    print(f"    LLM calls:   — vs 20  →  {_fmt_times(s['oracle_avg_calls_times_less'])}")
    print()
    print(
        f"{'Query':<28} {'Graph tok':>10} {'Flat tok':>10} {'× less':>7} "
        f"{'Views':>8} {'× faster':>9} {'Graph s':>8} {'Flat s':>7} {'× faster':>9}"
    )
    print("-" * 115)
    for r in report["results"]:
        g, f = r["graph"], r["flat"]
        tl = r["savings"].get("tokens_times_less")
        vf = r["savings"].get("views_times_faster")
        g_sec = g.get("timing", {}).get("total_sec", 0)
        f_sec = f.get("timing", {}).get("total_sec", 0)
        time_ratio = round(f_sec / g_sec, 1) if g_sec > 0 else None
        print(
            f"{_short_label(r['query']):<28} "
            f"{g['input_tokens']:>10,} {f['input_tokens']:>10,} {f'{tl}×':>7} "
            f"{g['views_checked']}/{f['views_checked']:>5} {f'{vf}×' if vf else '—':>9} "
            f"{g_sec:>7.1f} {f_sec:>7.1f} "
            f"{f'{time_ratio}×' if time_ratio else '—':>9}"
        )
    print("-" * 115)

    scaling = report.get("scaling") or {}
    rows = scaling.get("rows") or []
    interp = scaling.get("interpretation") or {}
    if rows:
        print()
        print(f"  SCALING SWEEP — query \"{scaling.get('query', '')}\" (vary view count):")
        print(
            f"    {'Views':>6} {'Flat s':>8} {'Graph s':>8} {'× faster':>9} "
            f"{'Flat views chk':>14} {'Graph views':>11} {'Graph nodes':>12}"
        )
        print("    " + "-" * 72)
        for row in rows:
            print(
                f"    {row['views_available']:>6} "
                f"{row['flat']['total_sec']:>8.1f} "
                f"{row['graph']['total_sec']:>8.1f} "
                f"{row.get('speedup_sec', '—'):>8}× "
                f"{row['flat']['views_checked']:>14} "
                f"{row['graph']['views_checked']:>11} "
                f"{row['graph']['nodes_visited']:>12}"
            )
        print()
        print(f"    Flat:  +{interp.get('flat_sec_per_extra_view', 0)} s per extra view  →  speed depends on **view count**")
        print(
            f"    Graph: +{interp.get('graph_sec_per_extra_view', 0)} s per extra view  →  "
            f"speed depends on **graph depth / branches**, not scene size"
        )
        if interp.get("note"):
            print(f"    {interp['note']}")

    print()
    print("  total_sec = local infra + estimated LLM time (fast mode, no extended thinking).")
    print()


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def print_output_links(
    *,
    hub_path: Path,
    run_dash_path: Path,
    md_path: Path,
    log_path: Path,
    json_path: Path,
    video_path: Optional[Path],
    reasoning_report: dict,
    charts: dict,
    scene_id: str,
) -> None:
    """Print clickable file:// links and open commands for Terminal (macOS)."""
    bundle = DOCS_DIR / (reasoning_report.get("bundle_path") or f"benchmark_results/reasoning_{scene_id}.json")

    print()
    print("=" * 72)
    print("  START HERE — click file:// links (Cmd+click in Terminal)")
    print("=" * 72)
    print()
    print("  🏠 DOCS HUB (one page — everything linked)")
    print(f"    {_file_url(hub_path)}")
    print(f"    open \"{hub_path}\"")
    print()
    print("  📋 RUN DASHBOARD (this run — reasoning + charts)")
    print(f"    {_file_url(run_dash_path)}")
    print(f"    open \"{run_dash_path}\"")
    print()
    print("  📊 BENCHMARK REPORT")
    print(f"    {_file_url(md_path)}")
    print(f"    open \"{md_path}\"")
    print()
    print("  📝 PROJECT LOG")
    print(f"    {_file_url(log_path)}")
    print(f"    open \"{log_path}\"")
    print()
    print("  🧠 REASONING (pick a case from run dashboard, or direct links):")
    for case in reasoning_report.get("cases", [])[:5]:
        md_rel = case.get("reasoning_md")
        if md_rel:
            p = DOCS_DIR / md_rel
            print(f"    [{case.get('case_index', '?')}] {case.get('query', '')[:50]}")
            print(f"        {_file_url(p)}")
    print()
    if video_path and video_path.exists():
        print("  🎬 DEMO VIDEO")
        print(f"    {_file_url(video_path)}")
        print(f"    open \"{video_path}\"")
        print()
    if bundle.exists():
        print(f"  📦 Bundle JSON: {_file_url(bundle)}")
        print()
    print("  🌐 LIVE UI (if ./start_all.sh running): http://localhost:5173 → Query Flow")
    print()
    print("=" * 72)
    print()


def main() -> None:
    _fix_windows_console()
    parser = argparse.ArgumentParser(description="Graph vs flat — resources & speed only")
    parser.add_argument("--scene", default="default")
    args = parser.parse_args()

    from backend.query.benchmark import run_scene_benchmark  # noqa: E402
    from backend.query.failure_cases import run_failure_suite  # noqa: E402
    from backend.query.benchmark_report import write_report  # noqa: E402
    from backend.query.project_log import write_project_log  # noqa: E402
    from backend.query.benchmark_reasoning import write_benchmark_reasoning  # noqa: E402

    report = run_scene_benchmark(args.scene)
    failure_report = run_failure_suite(args.scene)
    report["failure_cases"] = failure_report
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    run_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    report["run_slug"] = run_slug

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"benchmark_{args.scene}.json"

    video_path = OUT_DIR / "failure_case_demo.mp4"
    e2e_path = OUT_DIR / "e2e_product_demo.mp4"
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "record_failure_demo.py"), "--output", str(video_path)],
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT), "MPLCONFIGDIR": str(_mpl_dir)},
        )
        report["demo_video"] = "benchmark_results/failure_case_demo.mp4"
    except Exception as exc:
        print(f"  video: skipped ({exc})")
        video_path = None

    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "render_e2e_product_demo.py"), "--output", str(e2e_path)],
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT), "MPLCONFIGDIR": str(_mpl_dir)},
        )
        report["e2e_demo_video"] = "benchmark_results/e2e_product_demo.mp4"
    except Exception as exc:
        print(f"  e2e video: skipped ({exc})")

    reasoning_report = write_benchmark_reasoning(
        args.scene, failure_report, docs_dir=DOCS_DIR, run_slug=run_slug,
    )
    report["reasoning"] = reasoning_report

    md_path, charts = write_report(report, DOCS_DIR, failure_report=failure_report)

    log_path = write_project_log(report, failure_report, DOCS_DIR, reasoning_report=reasoning_report)
    report["charts"] = charts

    from backend.query.docs_hub import write_docs_hub, write_run_dashboard  # noqa: E402

    hub_path = write_docs_hub(
        DOCS_DIR, report, failure_report, reasoning_report,
        run_slug=run_slug, log_path=log_path,
    )
    run_dash_path = write_run_dashboard(
        DOCS_DIR, run_slug, report, failure_report, reasoning_report,
        log_rel=str(
            log_path.relative_to(DOCS_DIR / "experiments" / "stub" / "demo_runs")
        ),
    )
    report["docs_hub"] = str(hub_path.relative_to(ROOT))
    report["run_dashboard"] = str(run_dash_path.relative_to(ROOT))

    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print_table(report)
    fs = failure_report.get("summary", {})
    print(
        f"Failure cases: flat wrong-room {fs.get('flat_wrong_room_count', 0)}/"
        f"{failure_report.get('cases_run', 0)} · graph wins {fs.get('graph_wins_room_count', 0)}"
    )

    print_output_links(
        hub_path=hub_path,
        run_dash_path=run_dash_path,
        md_path=md_path,
        log_path=log_path,
        json_path=json_path,
        video_path=video_path if video_path and video_path.exists() else None,
        reasoning_report=reasoning_report,
        charts=charts,
        scene_id=args.scene,
    )


if __name__ == "__main__":
    main()
