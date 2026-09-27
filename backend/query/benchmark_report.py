"""Generate benchmark charts and auto-written markdown report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Non-interactive backend for headless runs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palette
C_GRAPH = "#2563eb"
C_FLAT = "#64748b"
C_ORACLE = "#7c3aed"
C_SPEEDUP = "#059669"
C_BG = "#f8fafc"
C_GRID = "#e2e8f0"


def _fmt_ratio(val: Optional[float], label: str = "less") -> str:
    if val is None:
        return "—"
    return f"**{val}×** {label}"


def _fmt_num(n: float, decimals: int = 1) -> str:
    if decimals == 0:
        return f"{int(round(n)):,}"
    return f"{n:,.{decimals}f}"


def _short_query(q: str, n: int = 32) -> str:
    q = q.strip()
    return q if len(q) <= n else q[: n - 1] + "…"


def _oracle_averages(report: dict) -> Dict[str, Optional[float]]:
    rows = [r for r in report.get("results", []) if r.get("oracle")]
    if not rows:
        return {}
    n = len(rows)

    def _avg(getter):
        vals = [getter(r) for r in rows if getter(r) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "tokens": round(sum(r["oracle"]["graph_oracle"]["input_tokens"] for r in rows) / n),
        "views": _avg(lambda r: r["oracle"]["graph_oracle"]["views_checked"]),
        "calls": _avg(lambda r: r["oracle"]["graph_oracle"]["llm_calls"]),
        "tokens_times_less": _avg(lambda r: r["oracle"]["savings_vs_flat"]["tokens_times_less"]),
        "views_times_faster": _avg(lambda r: r["oracle"]["savings_vs_flat"]["views_times_faster"]),
        "calls_times_less": _avg(lambda r: r["oracle"]["savings_vs_flat"]["calls_times_less"]),
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, linestyle="-", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)


def generate_charts(report: dict, out_dir: Path, failure_report: Optional[dict] = None) -> Dict[str, str]:
    """Render PNG charts; return relative paths (from docs/) for markdown embeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = report["results"]
    s = report["summary"]
    queries = [_short_query(r["query"], 22) for r in results]
    x = np.arange(len(queries))

    paths: Dict[str, str] = {}

    # ── 1. Tokens per query ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    w = 0.36
    g_tok = [r["graph"]["input_tokens"] for r in results]
    f_tok = [r["flat"]["input_tokens"] for r in results]
    ax.bar(x - w / 2, g_tok, w, label="Graph", color=C_GRAPH, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, f_tok, w, label="Flat", color=C_FLAT, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Input tokens")
    ax.set_title("Token cost per query — graph prunes most prompts", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    p = out_dir / "tokens_comparison.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths["tokens"] = "benchmark_results/tokens_comparison.png"

    # ── 2. Time (seconds) per query ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    g_sec = [r["graph"]["timing"]["total_sec"] for r in results]
    f_sec = [r["flat"]["timing"]["total_sec"] for r in results]
    ax.bar(x - w / 2, g_sec, w, label="Graph", color=C_GRAPH, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, f_sec, w, label="Flat", color=C_FLAT, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Estimated time (seconds)")
    ax.set_title("End-to-end time per query (fast LLM, no thinking)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    p = out_dir / "time_seconds.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths["time"] = "benchmark_results/time_seconds.png"

    # ── 3. Summary averages ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    labels = ["Tokens (÷1000)", "Views checked", "LLM calls", "Time (s)"]
    g_vals = [
        s["avg_graph_tokens"] / 1000,
        s["avg_graph_views_checked"],
        sum(r["graph"]["llm_calls"] for r in results) / len(results),
        s["avg_graph_total_sec"],
    ]
    f_vals = [
        s["avg_flat_tokens"] / 1000,
        report["total_views"],
        20,
        s["avg_flat_total_sec"],
    ]
    y = np.arange(len(labels))
    bh = 0.32
    ax.barh(y - bh / 2, g_vals, bh, label="Graph (avg)", color=C_GRAPH, edgecolor="white")
    ax.barh(y + bh / 2, f_vals, bh, label="Flat (avg)", color=C_FLAT, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Average resource use — graph vs flat", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    p = out_dir / "summary_averages.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths["summary"] = "benchmark_results/summary_averages.png"

    # ── 4. Scaling curve ─────────────────────────────────────────────────
    scaling = report.get("scaling") or {}
    rows = scaling.get("rows") or []
    if rows:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        views = [r["views_available"] for r in rows]
        flat_s = [r["flat"]["total_sec"] for r in rows]
        graph_s = [r["graph"]["total_sec"] for r in rows]
        ax.plot(views, flat_s, "o-", color=C_FLAT, linewidth=2.2, markersize=7, label="Flat")
        ax.plot(views, graph_s, "s-", color=C_GRAPH, linewidth=2.2, markersize=7, label="Graph")
        ax.fill_between(views, graph_s, flat_s, alpha=0.12, color=C_SPEEDUP)
        ax.set_xlabel("Views in scene")
        ax.set_ylabel("Estimated time (seconds)")
        q = scaling.get("query", "")
        ax.set_title(f'Scaling: how time grows with view count — "{_short_query(q, 28)}"',
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        p = out_dir / "scaling_curve.png"
        fig.savefig(p, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        paths["scaling"] = "benchmark_results/scaling_curve.png"

    # ── 5. Speedup multipliers per query ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    speedups = []
    for r in results:
        g = r["graph"]["timing"]["total_sec"]
        f = r["flat"]["timing"]["total_sec"]
        speedups.append(round(f / g, 1) if g > 0 else 0)
    colors = [C_SPEEDUP if sp >= 5 else C_GRAPH for sp in speedups]
    bars = ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(s.get("avg_sec_times_faster") or 0, color=C_FLAT, linestyle="--", linewidth=1.2,
               label=f"Avg {s.get('avg_sec_times_faster', '—')}×")
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("× faster (flat ÷ graph)")
    ax.set_title("Time speedup per query", fontsize=12, fontweight="bold", pad=12)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val}×", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    p = out_dir / "speedup_per_query.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths["speedup"] = "benchmark_results/speedup_per_query.png"

    # ── 6. Failure case room accuracy ────────────────────────────────────
    if failure_report and failure_report.get("results"):
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
        cases = failure_report["results"]
        labels = [_short_query(c["case"]["query"], 26) for c in cases]
        x2 = np.arange(len(labels))
        verdict_score = {"correct_room": 2, "correct_zone": 1.5, "other": 1, "miss": 0, "wrong_room": -1}
        g_scores = [verdict_score.get(c["graph"]["classification"], 0) for c in cases]
        f_scores = [verdict_score.get(c["flat"]["classification"], 0) for c in cases]
        ax.bar(x2 - w / 2, g_scores, w, label="Graph", color=C_GRAPH, edgecolor="white")
        ax.bar(x2 + w / 2, f_scores, w, label="Flat", color=C_FLAT, edgecolor="white")
        ax.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
        ax.set_yticks([-1, 0, 1, 1.5, 2])
        ax.set_yticklabels(["wrong room", "miss", "other", "correct zone", "correct room"])
        ax.set_title("Room-scoped failure cases — graph vs flat", fontsize=11, fontweight="bold", pad=12)
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        p = out_dir / "failure_room_accuracy.png"
        fig.savefig(p, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        paths["failure"] = "benchmark_results/failure_room_accuracy.png"

    return paths


def generate_markdown(
    report: dict,
    chart_paths: Dict[str, str],
    *,
    failure_report: Optional[dict] = None,
    md_path: Optional[Path] = None,
) -> str:
    """Build a polished markdown report from benchmark JSON."""
    s = report["summary"]
    gs = report.get("graph_stats") or {}
    tm = report.get("timing_model") or {}
    scaling = report.get("scaling") or {}
    interp = scaling.get("interpretation") or {}
    oracle = _oracle_averages(report)
    tv = report["total_views"]
    scene = report["scene_id"]
    nq = report["queries_run"]

    gen_at = report.get("generated_at", "")
    try:
        ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        date_str = gen_at or "—"

    avg_g_calls = round(
        sum(r["graph"]["llm_calls"] for r in report["results"]) / max(nq, 1), 1
    )

    lines: List[str] = [
        "# Graph vs Flat — Benchmark Report",
        "",
        "> Semantic tree traversal (graph) compared to exhaustive view search (flat).",
        f"> **Scene:** `{scene}` · **Generated:** {date_str}",
        "",
        "**Navigation:** [Docs hub](../README.md) · "
        f"[Run dashboard](../experiments/stub/demo_runs/runs/{report.get('run_slug', 'latest')}/README.md) · "
        "[Demo run history](../experiments/stub/demo_runs/README.md)",
        "",
        "## Contents",
        "",
        "- [At a glance](#at-a-glance)",
        "- [Scene & semantic graph](#scene--semantic-graph)",
        "- [Timing model](#timing-model)",
        "- [Per-query results](#per-query-results)",
        "- [What drives speed?](#what-drives-speed)",
        "- [Failure cases](#failure-cases--room-disambiguation)",
        "- [Methodology](#methodology)",
        "",
        "---",
        "",
        "## At a glance",
        "",
        "| | Graph | Flat | Advantage |",
        "|--|------:|-----:|----------:|",
        f"| Input tokens | {_fmt_num(s['avg_graph_tokens'], 0)} | {_fmt_num(s['avg_flat_tokens'], 0)} | {_fmt_ratio(s['avg_tokens_times_less'])} |",
        f"| Views checked | {s['avg_graph_views_checked']} | {tv} | {_fmt_ratio(s['avg_views_times_faster'], 'faster')} |",
        f"| LLM calls | {avg_g_calls} | 20 | {_fmt_ratio(s['avg_calls_times_less'])} |",
        f"| **Time (est.)** | **{s['avg_graph_total_sec']} s** | **{s['avg_flat_total_sec']} s** | {_fmt_ratio(s.get('avg_sec_times_faster'), 'faster')} |",
        "",
        "The graph search checks **fewer views and sends fewer tokens** to the LLM, so queries finish in a fraction of the time.",
        "",
    ]

    if chart_paths.get("summary"):
        lines += [
            f"![Average metrics]({chart_paths['summary']})",
            "",
        ]

    lines += [
        "---",
        "",
        "## Scene & semantic graph",
        "",
        f"| Property | Value |",
        "|----------|------:|",
        f"| Captured views | {tv} |",
    ]
    if gs.get("has_tree"):
        lines += [
            f"| Tree nodes | {gs['node_count']} |",
            f"| Tree depth | {gs['max_depth']} |",
            f"| Leaf nodes | {gs.get('leaf_count', 0)} |",
            f"| Zone nodes | {gs.get('zone_count', 0)} |",
            f"| Views mapped in tree | {gs.get('views_in_tree', tv)} |",
        ]
    lines += [
        "",
        "The semantic graph groups views into zones and regions. Traversal prunes irrelevant branches before any leaf-level view confirmation runs.",
        "",
        "---",
        "",
        "## Timing model",
        "",
        "Speed is reported in **seconds**. We measure local prompt assembly and estimate LLM latency for a **fast model without extended thinking**:",
        "",
        "```",
        tm.get("formula", "total_sec = infra_sec + llm_calls×0.35 + input_tokens/140"),
        "```",
        "",
        f"| Parameter | Value |",
        "|-----------|------:|",
        f"| Overhead per LLM call | {tm.get('sec_per_call', 0.35)} s |",
        f"| Throughput | {tm.get('tokens_per_sec', 140)} tok/s |",
        "",
    ]

    if oracle:
        lines += [
            "### Oracle path (real logged LLM traversal)",
            "",
            "When a saved query session exists, we replay the exact branch choices the LLM made:",
            "",
            "| | Oracle | Flat | Advantage |",
            "|--|-------:|-----:|----------:|",
            f"| Input tokens | ~{_fmt_num(oracle['tokens'], 0)} | {_fmt_num(s['avg_flat_tokens'], 0)} | {_fmt_ratio(oracle.get('tokens_times_less'))} |",
            f"| Views checked | ~{oracle.get('views', '—')} | {tv} | {_fmt_ratio(oracle.get('views_times_faster'), 'faster')} |",
            f"| LLM calls | ~{oracle.get('calls', '—')} | 20 | {_fmt_ratio(oracle.get('calls_times_less'))} |",
            "",
        ]

    lines += [
        "---",
        "",
        "## Per-query results",
        "",
    ]

    if chart_paths.get("tokens"):
        lines.append(f"![Token comparison]({chart_paths['tokens']})")
        lines.append("")
    if chart_paths.get("time"):
        lines.append(f"![Time comparison]({chart_paths['time']})")
        lines.append("")

    lines += [
        "| Query | Graph tok | Flat tok | × less | Views G/F | Graph s | Flat s | × faster |",
        "|-------|----------:|---------:|-------:|----------:|--------:|-------:|---------:|",
    ]
    for r in report["results"]:
        g, f = r["graph"], r["flat"]
        tl = r["savings"].get("tokens_times_less")
        g_sec = g["timing"]["total_sec"]
        f_sec = f["timing"]["total_sec"]
        tr = round(f_sec / g_sec, 1) if g_sec > 0 else None
        lines.append(
            f"| {r['query']} "
            f"| {_fmt_num(g['input_tokens'], 0)} "
            f"| {_fmt_num(f['input_tokens'], 0)} "
            f"| {tl}× "
            f"| {g['views_checked']}/{f['views_checked']} "
            f"| {g_sec} "
            f"| {f_sec} "
            f"| {tr}× |"
        )

    if chart_paths.get("speedup"):
        lines += [
            "",
            f"![Speedup per query]({chart_paths['speedup']})",
            "",
        ]

    lines += [
        "---",
        "",
        "## What drives speed?",
        "",
        "| Factor | Flat search | Graph search |",
        "|--------|-------------|--------------|",
        f"| **View count** | **Primary** — +{interp.get('flat_sec_per_extra_view', '?')} s per extra view | Weak — only relevant leaves |",
        f"| **Graph size** | No effect (tree unused) | Moderate — depth {gs.get('max_depth', '?')}, {gs.get('leaf_count', '?')} leaves |",
        "| **Scene / PLY size** | No effect | No effect |",
        "",
    ]

    scale_rows = scaling.get("rows") or []
    if scale_rows:
        if chart_paths.get("scaling"):
            lines += [
                f"![Scaling curve]({chart_paths['scaling']})",
                "",
            ]
        lines += [
            f'Scaling sweep on *"{scaling.get("query", "")}"* — vary how many views exist in the scene:',
            "",
            "| Views | Flat (s) | Graph (s) | × faster | Flat views chk | Graph views chk | Graph nodes |",
            "|------:|---------:|----------:|---------:|---------------:|----------------:|------------:|",
        ]
        for row in scale_rows:
            lines.append(
                f"| {row['views_available']} "
                f"| {row['flat']['total_sec']} "
                f"| {row['graph']['total_sec']} "
                f"| {row.get('speedup_sec', '—')}× "
                f"| {row['flat']['views_checked']} "
                f"| {row['graph']['views_checked']} "
                f"| {row['graph']['nodes_visited']} |"
            )
        if interp.get("note"):
            lines += [
                "",
                f"**Takeaway:** {interp['note']}",
                "",
            ]

    if failure_report and failure_report.get("results"):
        fs = failure_report["summary"]
        lines += [
            "---",
            "",
            "## Failure cases — room disambiguation",
            "",
            "Queries where the same object exists in **multiple rooms**. Flat search scans "
            "all views and may return the first keyword match in the **wrong room**. Graph "
            "traversal scopes to the correct zone first.",
            "",
            f"| | Graph room-correct | Flat room-correct | Flat **wrong room** | Graph wins |",
            f"|--|-------------------:|------------------:|--------------------:|-----------:|",
            f"| Count | {fs.get('graph_room_correct', 0)}/{failure_report.get('cases_run', 0)} "
            f"| {fs.get('flat_room_correct', 0)}/{failure_report.get('cases_run', 0)} "
            f"| **{fs.get('flat_wrong_room_count', 0)}** "
            f"| {fs.get('graph_wins_room_count', 0)} |",
            "",
        ]
        if fs.get("highlight_case"):
            lines += [
                f"> **Demo query:** *\"{fs['highlight_case']}\"* — flat returns a ballroom-floor "
                f"screen (v012); graph returns the conference-room stage screen (v017).",
                "",
            ]
        if chart_paths.get("failure"):
            lines += [
                f"![Room accuracy]({chart_paths['failure']})",
                "",
            ]
        lines += [
            "| Query | Graph view | Verdict | Flat view | Verdict | Tokens G/F | Time G/F |",
            "|-------|------------|---------|-----------|---------|------------|----------|",
        ]
        for r in failure_report["results"]:
            g, f = r["graph"], r["flat"]
            lines.append(
                f"| {r['case']['query']} "
                f"| {g.get('view_id') or '—'} "
                f"| {g['classification']} "
                f"| {f.get('view_id') or '—'} "
                f"| **{f['classification']}** "
                f"| {g['input_tokens']:,}/{f['input_tokens']:,} "
                f"| {g.get('timing', {}).get('total_sec', '—')}/{f.get('timing', {}).get('total_sec', '—')} s |"
            )
        lines += [
            "",
            "### Why this matters for the product",
            "",
        ]
        for r in failure_report["results"]:
            if r.get("flat_wrong_room") or r.get("demo_highlight"):
                lines += [
                    f"**{r['case']['query']}**",
                    "",
                    f"- Target room: {r['case']['room_label']}",
                    f"- {r['case']['why_it_matters']}",
                    "",
                ]
        demo_video = report.get("demo_video")
        reasoning = report.get("reasoning") or {}
        if demo_video:
            lines += [
                "### Demo video",
                "",
                f"[Download / play `failure_case_demo.mp4`]({demo_video}) — flat vs graph side-by-side.",
                "",
            ]
        if reasoning.get("run_dashboard"):
            lines += [
                "### Reasoning traces",
                "",
                f"[Open run dashboard]({reasoning['run_dashboard']}) — all cases with links.",
                "",
            ]
        if reasoning.get("reasoning_md_paths"):
            lines += [
                "### Reasoning traces",
                "",
                "Auto-generated traversal reasoning (graph + flat) for each failure case:",
                "",
            ]
            for case in reasoning.get("cases", []):
                md = case.get("reasoning_md", "")
                lines.append(f"- [{case.get('query', '?')}]({md})")
            lines += [
                "",
                f"Session JSON: `backend/data/scenes/{scene}/queries/auto_*_graph.json`",
                "",
            ]
        lines.append("")

    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "- **Graph mode** — query decomposition → semantic tree traversal with branch pruning → leaf confirmation only on visited leaves.",
        "- **Flat mode** — same decomposition, then leaf confirmation on **every** captured view (no tree).",
        "- **Oracle mode** — graph traversal replayed from saved session logs (real LLM branch choices).",
        f"- **Queries:** {nq} (defaults + saved session logs).",
        "",
        "---",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. .venv/bin/python scripts/run_full_benchmark.py",
        "# or one command from project root:",
        "./run_all_docs.sh",
        "```",
        "",
        f"Raw data: [`benchmark_results/benchmark_{scene}.json`](../benchmark_results/benchmark_{scene}.json)",
        "",
        "Session log: [`demo_runs/README.md`](../experiments/stub/demo_runs/README.md) "
        "(auto-updated each run)",
        "",
    ]

    text = "\n".join(lines)
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
    return text


def write_report(
    report: dict,
    docs_dir: Path,
    *,
    failure_report: Optional[dict] = None,
    md_filename: str = "benchmarks/benchmark_graph_vs_flat.md",
) -> Tuple[Path, Dict[str, str]]:
    """Generate charts + markdown in one call."""
    results_dir = docs_dir / "benchmark_results"
    chart_paths = generate_charts(report, results_dir, failure_report=failure_report)
    md_path = docs_dir / md_filename
    markdown_chart_paths = {key: f"../{value}" for key, value in chart_paths.items()}
    generate_markdown(
        report,
        markdown_chart_paths,
        failure_report=failure_report,
        md_path=md_path,
    )
    return md_path, chart_paths
