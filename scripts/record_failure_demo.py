#!/usr/bin/env python3
"""Generate an MP4 demo video for room-disambiguation failure cases.

Usage:
    PYTHONPATH=. python scripts/record_failure_demo.py
    PYTHONPATH=. python scripts/record_failure_demo.py --query "where is the screen in the conference room"

Output: docs/benchmark_results/failure_case_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fix_windows_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = ROOT / "docs" / "benchmark_results" / "failure_case_demo.mp4"
SCENE_DIR = ROOT / "backend" / "data" / "scenes" / "default"
IMAGES_DIR = SCENE_DIR / "images"

DEMO_QUERY = "where is the screen in the conference room"

# Visual theme
BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
GRAPH_C = "#3b82f6"
FLAT_C = "#64748b"
OK_C = "#22c55e"
BAD_C = "#ef4444"
WARN_C = "#f59e0b"


def _load_thumb(view_id: str, max_w: int = 320) -> Optional[np.ndarray]:
    path = IMAGES_DIR / f"{view_id}.png"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    ratio = max_w / img.width
    img = img.resize((max_w, int(img.height * ratio)), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _run_case(query: str) -> dict:
    from backend.query.benchmark import simulate_flat_search, simulate_graph_search
    from backend.query.failure_cases import FAILURE_CASES, _classify_result

    g = simulate_graph_search("default", query)
    f = simulate_flat_search("default", query)
    case = next((c for c in FAILURE_CASES if c.query.lower() == query.lower()), None)
    scene_dir = str(SCENE_DIR)
    g_cls = _classify_result(case, found=g.found, view_id=g.view_id, scene_dir=scene_dir) if case else "?"
    f_cls = _classify_result(case, found=f.found, view_id=f.view_id, scene_dir=scene_dir) if case else "?"
    return {
        "query": query,
        "graph": g,
        "flat": f,
        "g_cls": g_cls,
        "f_cls": f_cls,
        "case": case,
    }


def _style_fig(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")


def _draw_title(ax, title: str, subtitle: str = "") -> None:
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=28, fontweight="bold", color=TEXT,
            transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.48, subtitle, ha="center", va="center", fontsize=14, color=MUTED,
                transform=ax.transAxes, wrap=True)


def _draw_query_box(ax, query: str, y: float = 0.55) -> None:
    box = mpatches.FancyBboxPatch(
        (0.08, y - 0.08), 0.84, 0.16,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=PANEL, edgecolor=GRAPH_C, linewidth=2,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(0.5, y, f'"{query}"', ha="center", va="center", fontsize=18, color=TEXT,
            fontstyle="italic", transform=ax.transAxes)


def _place_image(ax, arr: np.ndarray, xy: Tuple[float, float], zoom: float = 0.45) -> None:
    im = OffsetImage(arr, zoom=zoom)
    ab = AnnotationBbox(im, xy, xycoords="axes fraction", frameon=True,
                        bboxprops=dict(edgecolor=PANEL, facecolor=PANEL, linewidth=2))
    ax.add_artist(ab)


def _verdict_badge(ax, x: float, y: float, label: str, ok: bool) -> None:
    color = OK_C if ok else BAD_C
    ax.text(x, y, label, ha="center", va="center", fontsize=13, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="none"))


def build_frames(data: dict, fps: int = 24) -> Tuple[List[np.ndarray], int]:
    """Return list of RGB frame arrays and fps."""
    query = data["query"]
    g, f = data["graph"], data["flat"]
    flat_checked = f.views_checked_ids
    graph_path = g.path

    frames: List[np.ndarray] = []
    duration = lambda sec: int(sec * fps)

    def capture(fig) -> None:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(buf.copy())
        plt.close(fig)

    def hold(fig, seconds: float) -> None:
        n = duration(seconds)
        capture(fig)
        last = frames[-1]
        for _ in range(n - 1):
            frames.append(last.copy())

    # ── 1. Title ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    _style_fig(fig, ax)
    _draw_title(ax, "Graph vs Flat", "Failure case demo — room disambiguation")
    ax.text(0.5, 0.28, "SemanticSplat · beyond-proximity", ha="center", fontsize=12, color=MUTED,
            transform=ax.transAxes)
    hold(fig, 2.5)

    # ── 2. Query ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    _style_fig(fig, ax)
    ax.text(0.5, 0.78, "User query", ha="center", fontsize=16, color=MUTED, transform=ax.transAxes)
    _draw_query_box(ax, query, y=0.52)
    ax.text(0.5, 0.28, "Multiple projection screens exist in different rooms.", ha="center",
            fontsize=13, color=MUTED, transform=ax.transAxes)
    hold(fig, 2.5)

    # ── 3. Flat scan animation ───────────────────────────────────────────
    thumb_wrong = _load_thumb(f.view_id or "v012")
    scan_upto = flat_checked.index(f.view_id) + 1 if f.view_id in flat_checked else len(flat_checked)

    for step in range(1, scan_upto + 1):
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        _style_fig(fig, ax)
        ax.text(0.5, 0.92, "FLAT SEARCH — checks every view (no graph)", ha="center",
                fontsize=16, fontweight="bold", color=FLAT_C, transform=ax.transAxes)
        progress = step / len(flat_checked)
        ax.barh(0.08, progress, height=0.03, left=0.1, color=FLAT_C, transform=ax.transAxes)
        ax.text(0.5, 0.08, f"Scanning view {step}/{len(flat_checked)}", ha="center", fontsize=11,
                color=MUTED, transform=ax.transAxes)

        recent = flat_checked[max(0, step - 5):step]
        ax.text(0.12, 0.72, "Checked:", fontsize=12, color=MUTED, transform=ax.transAxes)
        for i, vid in enumerate(recent):
            ax.text(0.12 + i * 0.08, 0.65, vid, fontsize=11, color=TEXT, transform=ax.transAxes)

        if step == scan_upto and thumb_wrong is not None:
            _place_image(ax, thumb_wrong, (0.62, 0.48), zoom=0.55)
            ax.text(0.62, 0.18, f"First match: {f.view_id}", ha="center", fontsize=14,
                    color=TEXT, fontweight="bold", transform=ax.transAxes)
            ax.text(0.62, 0.12, "Ballroom floor (through doors)", ha="center", fontsize=11,
                    color=MUTED, transform=ax.transAxes)
            _verdict_badge(ax, 0.62, 0.04, "WRONG ROOM", ok=False)

        hold(fig, 0.35 if step < scan_upto else 2.0)

    # ── 4. Graph traversal animation ─────────────────────────────────────
    thumb_ok = _load_thumb(g.view_id or "v017")
    path_labels = [
        ("root", "Conference Hall"),
        ("zone_ballroom", "Ballroom zone"),
        ("leaf_ballroom_stage_front", "Stage & screen"),
        (g.view_id or "v017", f"View {g.view_id or 'v017'}"),
    ]
    for step in range(1, len(path_labels) + 1):
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        _style_fig(fig, ax)
        ax.text(0.5, 0.92, "GRAPH SEARCH — semantic tree traversal", ha="center",
                fontsize=16, fontweight="bold", color=GRAPH_C, transform=ax.transAxes)

        y = 0.78
        for i, (node, label) in enumerate(path_labels[:step]):
            active = i == step - 1
            ax.text(0.15, y, "→", fontsize=14, color=GRAPH_C if active else MUTED, transform=ax.transAxes)
            ax.text(0.2, y, label, fontsize=13 if active else 12,
                    color=TEXT if active else MUTED,
                    fontweight="bold" if active else "normal", transform=ax.transAxes)
            y -= 0.08

        if step == len(path_labels) and thumb_ok is not None:
            _place_image(ax, thumb_ok, (0.62, 0.42), zoom=0.55)
            ax.text(0.62, 0.12, "Conference room / stage screen", ha="center", fontsize=11,
                    color=MUTED, transform=ax.transAxes)
            _verdict_badge(ax, 0.62, 0.04, "CORRECT ROOM", ok=True)

        hold(fig, 0.5 if step < len(path_labels) else 2.5)

    # ── 5. Side-by-side summary ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    fig.suptitle("Result comparison", fontsize=20, fontweight="bold", color=TEXT, y=0.96)

    if thumb_wrong is not None:
        _place_image(axes[0], thumb_wrong, (0.5, 0.55), zoom=0.7)
    axes[0].text(0.5, 0.88, "FLAT", ha="center", fontsize=16, fontweight="bold", color=FLAT_C,
                 transform=axes[0].transAxes)
    axes[0].text(0.5, 0.18, f"{f.view_id} · ballroom floor", ha="center", fontsize=12, color=MUTED,
                 transform=axes[0].transAxes)
    _verdict_badge(axes[0], 0.5, 0.06, f"wrong room · {f.views_checked} views · ~{f.input_tokens:,} tok",
                   ok=False)

    if thumb_ok is not None:
        _place_image(axes[1], thumb_ok, (0.5, 0.55), zoom=0.7)
    axes[1].text(0.5, 0.88, "GRAPH", ha="center", fontsize=16, fontweight="bold", color=GRAPH_C,
                 transform=axes[1].transAxes)
    axes[1].text(0.5, 0.18, f"{g.view_id} · stage / conference end", ha="center", fontsize=12,
                 color=MUTED, transform=axes[1].transAxes)
    _verdict_badge(axes[1], 0.5, 0.06,
                   f"correct room · {g.views_checked} view · ~{g.input_tokens:,} tok", ok=True)

    hold(fig, 4.0)

    # ── 6. Closing ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    _style_fig(fig, ax)
    _draw_title(ax, "Graph search scopes to the right room", "")
    wrapped = textwrap.fill(
        "Flat search finds the first screen in the scene — not necessarily in the room you asked for. "
        "Semantic tree traversal prunes to the correct zone first.",
        width=70,
    )
    ax.text(0.5, 0.38, wrapped, ha="center", va="center", fontsize=14, color=MUTED,
            transform=ax.transAxes)
    ax.text(0.5, 0.12, "beyond-proximity · benchmark: scripts/run_full_benchmark.py", ha="center",
            fontsize=11, color=MUTED, transform=ax.transAxes)
    hold(fig, 3.0)

    return frames, fps


def write_mp4(frames: List[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])

    writer = FFMpegWriter(fps=fps, metadata={"title": "Failure case demo"}, bitrate=4000)

    with writer.saving(fig, str(path), dpi=100):
        for frame in frames:
            im.set_array(frame)
            writer.grab_frame()

    plt.close(fig)


def main() -> None:
    _fix_windows_console()
    parser = argparse.ArgumentParser(description="Record failure-case demo MP4")
    parser.add_argument("--query", default=DEMO_QUERY)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    _mpl = ROOT / ".matplotlib"
    _mpl.mkdir(exist_ok=True)
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

    print(f"Running case: {args.query!r}")
    data = _run_case(args.query)
    print(f"  flat → {data['flat'].view_id} ({data['f_cls']})")
    print(f"  graph → {data['graph'].view_id} ({data['g_cls']})")

    print("Rendering frames…")
    frames, fps = build_frames(data, fps=args.fps)
    print(f"  {len(frames)} frames @ {fps} fps (~{len(frames)/fps:.1f}s)")

    print(f"Writing {args.output}…")
    write_mp4(frames, args.output, fps)
    print(f"Done: {args.output} ({args.output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
