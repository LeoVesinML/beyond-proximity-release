"""Append structured session logs so any model can resume project context."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt_ratio(val: Optional[float]) -> str:
    return f"{val}×" if val is not None else "—"


def write_project_log(
    report: dict,
    failure_report: dict,
    docs_dir: Path,
    *,
    run_label: str = "benchmark",
    reasoning_report: Optional[dict] = None,
) -> Path:
    """Write timestamped simulated-demo logs under docs/experiments/stub/."""
    log_dir = docs_dir / "experiments" / "stub" / "demo_runs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gen_at = report.get("generated_at", "")
    try:
        ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
    except Exception:
        ts = datetime.now(timezone.utc)

    slug = ts.strftime("%Y-%m-%d_%H%M")
    path = log_dir / f"{slug}_{run_label}.md"

    s = report.get("summary", {})
    gs = report.get("graph_stats", {})
    tm = report.get("timing_model", {})
    fs = failure_report.get("summary", {})

    lines = [
        f"# Project log — {run_label}",
        "",
        f"**When:** {ts.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Scene:** `{report.get('scene_id', 'default')}`  ",
        f"**Run:** `{report.get('run_slug', '—')}`  ",
        "",
        "**Navigation:** "
        "[Docs hub](../../../README.md) · "
        f"[Run dashboard](runs/{report.get('run_slug', '')}/README.md) · "
        "[Benchmark report](../../../benchmarks/benchmark_graph_vs_flat.md)",
        "",
        "---",
        "",
        "## What we did",
        "",
        "Ran the full graph-vs-flat benchmark pipeline:",
        "",
        "1. Simulated **graph** search (semantic tree traversal + pruned leaf checks)",
        "2. Simulated **flat** search (leaf check on every view)",
        "3. Estimated **end-to-end time in seconds** (fast LLM, no thinking mode)",
        "4. Ran **room-scoped failure cases** to find wrong-room predictions",
        "5. Generated charts + updated [`benchmark_graph_vs_flat.md`](../benchmark_graph_vs_flat.md)",
        "",
        "---",
        "",
        "## Efficiency results (tokens & time)",
        "",
        "| Metric | Graph (avg) | Flat (avg) | Advantage |",
        "|--------|------------:|-----------:|----------:|",
        f"| Input tokens | {s.get('avg_graph_tokens', 0):,} | {s.get('avg_flat_tokens', 0):,} | {_fmt_ratio(s.get('avg_tokens_times_less'))} less |",
        f"| Views checked | {s.get('avg_graph_views_checked', 0)} | {report.get('total_views', 0)} | {_fmt_ratio(s.get('avg_views_times_faster'))} fewer |",
        f"| Time (est.) | {s.get('avg_graph_total_sec', 0)} s | {s.get('avg_flat_total_sec', 0)} s | {_fmt_ratio(s.get('avg_sec_times_faster'))} faster |",
        "",
        "**Timing model:**",
        "",
        f"```",
        tm.get("formula", "total_sec = infra + llm_calls×0.35 + tokens/140"),
        f"```",
        "",
        "### Per-query token spend",
        "",
        "| Query | Graph tokens | Flat tokens | Graph s | Flat s |",
        "|-------|-------------:|------------:|--------:|-------:|",
    ]

    for r in report.get("results", []):
        g, f = r["graph"], r["flat"]
        lines.append(
            f"| {r['query']} "
            f"| {g['input_tokens']:,} "
            f"| {f['input_tokens']:,} "
            f"| {g.get('timing', {}).get('total_sec', '—')} "
            f"| {f.get('timing', {}).get('total_sec', '—')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Failure / room-disambiguation cases",
        "",
        f"| | Graph room-correct | Flat room-correct | Flat wrong room | Graph wins |",
        f"|--|-------------------:|------------------:|----------------:|-----------:|",
        f"| Count | {fs.get('graph_room_correct', 0)}/{failure_report.get('cases_run', 0)} "
        f"| {fs.get('flat_room_correct', 0)}/{failure_report.get('cases_run', 0)} "
        f"| {fs.get('flat_wrong_room_count', 0)} "
        f"| {fs.get('graph_wins_room_count', 0)} |",
        "",
    ]

    if fs.get("highlight_case"):
        lines += [
            f"**Key demo query:** *\"{fs['highlight_case']}\"* — graph finds the screen in the "
            f"correct room; flat often returns a screen from a different zone.",
            "",
        ]

    lines += [
        "### Case-by-case",
        "",
        "| Query | Graph view | Graph verdict | Flat view | Flat verdict | Graph wins? |",
        "|-------|------------|---------------|-----------|--------------|-------------|",
    ]

    for r in failure_report.get("results", []):
        g, f = r["graph"], r["flat"]
        lines.append(
            f"| {r['case']['query']} "
            f"| {g.get('view_id') or '—'} "
            f"| {g['classification']} "
            f"| {f.get('view_id') or '—'} "
            f"| {f['classification']} "
            f"| {'✓' if r['graph_wins_room'] else '—'} |"
        )

    lines += [
        "",
        "### Reasoning — why graph helps",
        "",
    ]
    for r in failure_report.get("results", []):
        if r.get("demo_highlight") or r.get("flat_wrong_room"):
            lines += [
                f"**{r['case']['query']}**",
                "",
                f"- Room target: {r['case']['room_label']}",
                f"- Expected views: `{', '.join(r['case']['positive_view_ids'])}`",
                f"- Common flat mistake: `{', '.join(r['case']['wrong_room_view_ids'])}`",
                f"- {r['case']['why_it_matters']}",
                "",
            ]

    if reasoning_report and reasoning_report.get("cases"):
        lines += [
            "---",
            "",
            "## Reasoning traces",
            "",
            f"Full step-by-step traces: [run dashboard](runs/{report.get('run_slug', '')}/README.md)",
            "",
        ]
        for case in reasoning_report.get("cases", []):
            md = Path(case.get("reasoning_md", ""))
            trace = f"runs/{report.get('run_slug', '')}/reasoning/{md.name}"
            lines.append(
                f"- Case {case.get('case_index', '?')}: "
                f"[{case.get('query', '?')}]({trace})"
            )
        lines.append("")

    if reasoning_report and reasoning_report.get("reasoning_md_paths"):
        pass  # legacy block removed — links above

    lines += [
        "---",
        "",
        "## Graph topology (this scene)",
        "",
        f"- Nodes: {gs.get('node_count', '—')} · Depth: {gs.get('max_depth', '—')} · "
        f"Leaves: {gs.get('leaf_count', '—')} · Views: {report.get('total_views', '—')}",
        "",
        "---",
        "",
        "## Known issues / next steps",
        "",
        "- [ ] Add real LLM runs for failure cases (current sim uses keyword oracle on ViewJSON)",
        "- [ ] Second scene (Theater / outdoor) for cross-scene failure matrix",
        "- [x] Demo video: conference-room screen query (`docs/benchmark_results/failure_case_demo.mp4`)",
        "- [ ] Real LLM runs for failure cases (auto logs use simulation + tree summaries)",
        "",
        "---",
        "",
        "## Files produced this run",
        "",
        f"- [`benchmark_results/benchmark_{report.get('scene_id', 'default')}.json`](../../../benchmark_results/benchmark_{report.get('scene_id', 'default')}.json)",
        "- [`benchmark_graph_vs_flat.md`](../../../benchmarks/benchmark_graph_vs_flat.md)",
        "- [`benchmark_results/failure_case_demo.mp4`](../../../benchmark_results/failure_case_demo.mp4)",
        f"- This log: `experiments/stub/demo_runs/{path.name}`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")

    # Keep a rolling index for easy sharing
    from backend.query.docs_hub import update_project_log_index

    update_project_log_index(
        log_dir,
        run_slug=report.get("run_slug", path.stem.split("_")[0] + "_" + path.stem.split("_")[1]),
        log_filename=path.name,
        ts=ts,
        report=report,
        failure_report=failure_report,
    )
    return path
