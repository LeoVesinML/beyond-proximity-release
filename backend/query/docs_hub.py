"""Generate docs navigation hub and per-run dashboards."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEMO_RUNS_REL = Path("experiments/stub/demo_runs")


def _verdict_icon(verdict: str) -> str:
    return {
        "correct_room": "✅",
        "correct_zone": "✅",
        "wrong_room": "❌",
        "miss": "⚠️",
        "other": "❓",
    }.get(verdict, "·")


def write_run_dashboard(
    docs_dir: Path,
    run_slug: str,
    report: dict,
    failure_report: dict,
    reasoning_report: dict,
    *,
    log_rel: str,
    report_rel: str = "benchmarks/benchmark_graph_vs_flat.md",
    video_rel: Optional[str] = "benchmark_results/failure_case_demo.mp4",
) -> Path:
    """Write a dashboard below docs/experiments/stub/demo_runs/."""
    run_dir = docs_dir / _DEMO_RUNS_REL / "runs" / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "README.md"

    s = report.get("summary", {})
    fs = failure_report.get("summary", {})
    scene = report.get("scene_id", "default")
    cases = reasoning_report.get("cases", [])

    lines: List[str] = [
        f"# Run `{run_slug}`",
        "",
        f"**Scene:** `{scene}` · **Queries:** {report.get('queries_run', 0)} efficiency · "
        f"{failure_report.get('cases_run', 0)} failure cases",
        "",
        "## Navigation",
        "",
        "| | Link |",
        "|--|------|",
        f"| 🏠 Docs hub | [../../../../../README.md](../../../../../README.md) |",
        f"| 📊 Benchmark report | [{report_rel}](../../../../../{report_rel}) |",
        f"| 📝 Project log | [{log_rel}](../../{log_rel}) |",
    ]
    if video_rel:
        lines.append(f"| 🎬 Demo video | [{video_rel}](../../../../../{video_rel}) |")
    lines += [
        f"| 📦 Raw JSON | [benchmark_results/benchmark_{scene}.json](../../../../../benchmark_results/benchmark_{scene}.json) |",
        "",
        "## Results summary",
        "",
        "| Metric | Graph | Flat |",
        "|--------|------:|-----:|",
        f"| Time (est.) | {s.get('avg_graph_total_sec', '—')} s | {s.get('avg_flat_total_sec', '—')} s |",
        f"| Tokens | {s.get('avg_graph_tokens', '—'):,} | {s.get('avg_flat_tokens', '—'):,} |",
        f"| Flat wrong-room | — | **{fs.get('flat_wrong_room_count', 0)}/{failure_report.get('cases_run', 0)}** |",
        "",
        "## Reasoning traces",
        "",
        "Click a case to see step-by-step graph vs flat reasoning:",
        "",
        "| # | Query | Graph | Flat | Trace |",
        "|--:|-------|-------|------|-------|",
    ]

    for i, case in enumerate(cases, 1):
        md = case.get("reasoning_md", "")
        # reasoning md is relative to docs/ — convert to link from run dashboard
        fname = Path(md).name if md else ""
        rel = f"reasoning/{fname}" if fname else "—"
        g = case.get("graph_verdict", "?")
        f = case.get("flat_verdict", "?")
        q = case.get("query", "")
        short = q if len(q) <= 42 else q[:41] + "…"
        lines.append(
            f"| {i} | {short} | {_verdict_icon(g)} `{g}` | {_verdict_icon(f)} `{f}` | "
            f"[open]({rel}) |"
        )

    lines += [
        "",
        "## Charts",
        "",
        "| Chart | |",
        "|-------|---|",
        "| Tokens | [tokens_comparison.png](../../../../../benchmark_results/tokens_comparison.png) |",
        "| Time | [time_seconds.png](../../../../../benchmark_results/time_seconds.png) |",
        "| Failure cases | [failure_room_accuracy.png](../../../../../benchmark_results/failure_room_accuracy.png) |",
        "| Scaling | [scaling_curve.png](../../../../../benchmark_results/scaling_curve.png) |",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_docs_hub(
    docs_dir: Path,
    report: dict,
    failure_report: dict,
    reasoning_report: dict,
    *,
    run_slug: str,
    log_path: Path,
) -> Path:
    """Write the latest simulated-demo run summary without replacing docs/README.md."""
    path = docs_dir / _DEMO_RUNS_REL / "latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report.get("summary", {})
    fs = failure_report.get("summary", {})
    scene = report.get("scene_id", "default")
    gen_at = report.get("generated_at", "")
    try:
        ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        date_str = "—"

    log_rel = log_path.relative_to(docs_dir / _DEMO_RUNS_REL)
    run_dash = f"runs/{run_slug}/README.md"

    lines = [
        "# SemanticSplat — Documentation Hub",
        "",
        f"> **Latest run:** `{run_slug}` · {date_str} · scene `{scene}`",
        "",
        "---",
        "",
        "## Start here",
        "",
        "| What you need | Open |",
        "|---------------|------|",
        f"| **Latest full report** (metrics, charts, failure cases) | [benchmark_graph_vs_flat.md](../../../benchmarks/benchmark_graph_vs_flat.md) |",
        f"| **Latest run dashboard** (one page, all links) | [{run_dash}]({run_dash}) |",
        f"| **Latest project log** (tokens, decisions, next steps) | [{log_rel}]({log_rel}) |",
        f"| **Demo video** (conference room screen failure) | [failure_case_demo.mp4](../../../benchmark_results/failure_case_demo.mp4) |",
        f"| **All past runs** | [README.md](README.md) |",
        "",
        "---",
        "",
        "## Latest results",
        "",
        "| | Graph | Flat | Advantage |",
        "|--|------:|-----:|----------:|",
        f"| Time | {s.get('avg_graph_total_sec', '—')} s | {s.get('avg_flat_total_sec', '—')} s | "
        f"{s.get('avg_sec_times_faster', '—')}× faster |",
        f"| Tokens | {s.get('avg_graph_tokens', 0):,} | {s.get('avg_flat_tokens', 0):,} | "
        f"{s.get('avg_tokens_times_less', '—')}× less |",
        f"| Flat wrong-room cases | — | {fs.get('flat_wrong_room_count', 0)}/{failure_report.get('cases_run', 0)} | graph wins {fs.get('graph_wins_room_count', 0)} |",
        "",
        "---",
        "",
        "## Reasoning traces (latest run)",
        "",
        "| # | Case | Graph | Flat | |",
        "|--:|------|-------|------|---|",
    ]

    for i, case in enumerate(reasoning_report.get("cases", []), 1):
        md = case.get("reasoning_md", "")
        q = case.get("query", "")
        short = q if len(q) <= 40 else q[:39] + "…"
        g, f = case.get("graph_verdict", "?"), case.get("flat_verdict", "?")
        lines.append(
            f"| {i} | {short} | {_verdict_icon(g)} | {_verdict_icon(f)} | "
            f"[trace](../../../{md}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Folder structure",
        "",
        "```",
        "docs/",
        "├── README.md                    ← you are here",
        "├── benchmarks/benchmark_graph_vs_flat.md",
        "├── benchmark_results/           ← JSON, PNG, MP4",
        "└── experiments/stub/demo_runs/",
        "    ├── README.md                ← history of all runs",
        "    └── runs/{timestamp}/",
        "        ├── README.md            ← run dashboard",
        "        └── reasoning/*.md       ← step-by-step traces",
        "```",
        "",
        "---",
        "",
        "## Regenerate everything",
        "",
        "```bash",
        "./run_all_docs.sh",
        "```",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def update_project_log_index(
    log_dir: Path,
    *,
    run_slug: str,
    log_filename: str,
    ts: datetime,
    report: dict,
    failure_report: dict,
) -> None:
    """Rewrite the simulated-demo run index with dashboard links."""
    index_path = log_dir / "README.md"
    fs = failure_report.get("summary", {})
    s = report.get("summary", {})

    entry = (
        f"- **{ts.strftime('%Y-%m-%d %H:%M')}** — "
        f"[Run dashboard](runs/{run_slug}/README.md) · "
        f"[Log]({log_filename}) · "
        f"graph {s.get('avg_graph_total_sec', '?')}s vs flat {s.get('avg_flat_total_sec', '?')}s · "
        f"wrong-room {fs.get('flat_wrong_room_count', 0)}/{failure_report.get('cases_run', 0)}"
    )

    header = [
        "# Project log",
        "",
        "History of benchmark runs. For the latest overview, start at "
        "[**docs/README.md**](../../../README.md).",
        "",
        "## Runs (newest first)",
        "",
    ]

    existing: List[str] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- **20"):
                existing.append(line)

    if entry not in existing:
        existing.insert(0, entry)

    index_path.write_text("\n".join(header + existing) + "\n", encoding="utf-8")
