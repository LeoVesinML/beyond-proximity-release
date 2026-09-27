#!/usr/bin/env python3
"""Render a polished end-to-end product demo MP4 (no browser required).

Uses real pipeline session data + view images to tell the full story:
Navigate → Capture → Query → Pipeline → Found location.

Usage:
    PYTHONPATH=. python scripts/render_e2e_product_demo.py
    PYTHONPATH=. python scripts/render_e2e_product_demo.py --query "Find the sofa"

Output: docs/benchmark_results/e2e_product_demo.mp4
"""
from __future__ import annotations

import argparse
import os
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
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "benchmark_results" / "e2e_product_demo.mp4"
SCENE = "default"
SCENE_DIR = ROOT / "backend" / "data" / "scenes" / SCENE
IMAGES = SCENE_DIR / "images"

BG, PANEL, TEXT, MUTED = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8"
ACCENT, OK, BAD = "#4ade80", "#22c55e", "#ef4444"


def _thumb(view_id: str, w: int = 480) -> Optional[np.ndarray]:
    p = IMAGES / f"{view_id}.png"
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    r = w / img.width
    img = img.resize((w, int(img.height * r)), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _draw_bbox(arr: np.ndarray, bbox: Optional[list], color: str = ACCENT) -> np.ndarray:
    if not bbox or len(bbox) != 4:
        return arr
    img = Image.fromarray(arr.copy())
    d = ImageDraw.Draw(img)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = [int(b * (w if i % 2 == 0 else h)) for i, b in enumerate(bbox)]
    d.rectangle([x1, y1, x2, y2], outline=color, width=max(3, w // 120))
    return np.asarray(img)


def _style(ax):
    ax.set_facecolor(BG)
    ax.axis("off")


def _title(ax, main: str, sub: str = ""):
    ax.text(0.5, 0.62, main, ha="center", va="center", fontsize=30, fontweight="bold", color=TEXT, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.48, sub, ha="center", va="center", fontsize=14, color=MUTED, transform=ax.transAxes)


def _ui_bar(ax, labels: List[str], active: int):
    xs = np.linspace(0.12, 0.88, len(labels))
    for i, (x, lab) in enumerate(zip(xs, labels)):
        col = ACCENT if i == active else PANEL
        tc = BG if i == active else MUTED
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.08, 0.91), 0.16, 0.06, boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="#334155", transform=ax.transAxes))
        ax.text(x, 0.94, lab, ha="center", va="center", fontsize=9, color=tc, fontweight="bold", transform=ax.transAxes)


def _place_img(ax, arr: np.ndarray, xy=(0.5, 0.45), zoom=0.55):
    ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), xy, xycoords="axes fraction", frameon=True,
                          bboxprops=dict(edgecolor=PANEL, facecolor=PANEL, linewidth=2))
    ax.add_artist(ab)


def _run_pipeline(query: str) -> dict:
    from backend.query.live_session import run_query_session
    from backend.query.log import QuerySession

    sess = QuerySession(SCENE, query)
    sid = sess.session_id
    run_query_session(SCENE, sid, query, step_delay_sec=0)
    import json
    data = json.loads((SCENE_DIR / "queries" / f"{sid}.json").read_text())
    data["session_id"] = sid
    return data


def build_frames(data: dict, query: str, fps: int = 24) -> Tuple[List[np.ndarray], int]:
    frames: List[np.ndarray] = []
    steps = data.get("steps", [])
    result = data.get("result") or {}
    view_id = result.get("view_id")
    found = result.get("found", False)

    def hold(fig, sec: float):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        n = max(1, int(sec * fps))
        for _ in range(n):
            frames.append(buf.copy())
        plt.close(fig)

    # 1 Title
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _title(ax, "SemanticSplat", "End-to-end product demo")
    ax.text(0.5, 0.28, "Navigate · Capture · Query · Pipeline · Localize", ha="center", fontsize=13, color=MUTED, transform=ax.transAxes)
    hold(fig, 2.5)

    # 2 Navigator
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _ui_bar(ax, ["Navigator", "Semantic Tree", "Query Flow"], 0)
    nav_img = _thumb("v001")
    if nav_img is None:
        nav_img = _thumb("v006")
    if nav_img is not None:
        _place_img(ax, nav_img, (0.42, 0.48), 0.62)
    ax.add_patch(mpatches.FancyBboxPatch((0.72, 0.15), 0.22, 0.68, boxstyle="round,pad=0.02",
                 facecolor=PANEL, edgecolor="#334155", transform=ax.transAxes))
    ax.text(0.83, 0.78, "Captured Views", ha="center", fontsize=10, color=MUTED, fontweight="bold", transform=ax.transAxes)
    for i, vid in enumerate(["v001", "v004", "v006"][:3]):
        t = _thumb(vid, 120)
        if t is not None:
            _place_img(ax, t, (0.83, 0.62 - i * 0.18), 0.22)
    ax.text(0.42, 0.08, "WASD fly-through · ConferenceHall.ply · Press R to capture", ha="center", fontsize=12, color=MUTED, transform=ax.transAxes)
    hold(fig, 3.0)

    # 3 Capture flash
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _ui_bar(ax, ["Navigator", "Semantic Tree", "Query Flow"], 0)
    ax.text(0.5, 0.55, "CAPTURE  (R)", ha="center", fontsize=36, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.38, "RGB + depth + camera pose saved to scene", ha="center", fontsize=14, color=MUTED, transform=ax.transAxes)
    hold(fig, 1.5)

    # 4 Query
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _ui_bar(ax, ["Navigator", "Semantic Tree", "Query Flow"], 0)
    ax.add_patch(mpatches.FancyBboxPatch((0.15, 0.82), 0.55, 0.08, boxstyle="round,pad=0.02",
                 facecolor=PANEL, edgecolor="#334155", transform=ax.transAxes))
    ax.text(0.17, 0.86, query, ha="left", va="center", fontsize=16, color=TEXT, fontstyle="italic", transform=ax.transAxes)
    ax.add_patch(mpatches.FancyBboxPatch((0.72, 0.82), 0.1, 0.08, boxstyle="round,pad=0.02",
                 facecolor="#1d4ed8", edgecolor="none", transform=ax.transAxes))
    ax.text(0.77, 0.86, "Ask", ha="center", va="center", fontsize=14, color="white", fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.35, "Pipeline starts automatically — no Cursor session ID", ha="center", fontsize=13, color=MUTED, transform=ax.transAxes)
    hold(fig, 2.5)

    # 5 Pipeline steps
    _ui_bar_labels = ["Navigator", "Semantic Tree", "Query Flow"]
    for step in steps:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        fig.patch.set_facecolor(BG)
        _style(ax)
        _ui_bar(ax, _ui_bar_labels, 2)
        st = step.get("step_type", "")
        badge = st.upper().replace("_", " ")
        color = {"decomposition": "#6366f1", "traversal": "#0ea5e9", "leaf_check": "#a855f7", "found": OK, "not_found": BAD}.get(st, ACCENT)
        ax.text(0.08, 0.82, badge, fontsize=14, fontweight="bold", color="white", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="none"))

        if st == "decomposition":
            plan = step.get("structured_plan", {})
            y = 0.68
            for k, v in plan.items():
                ax.text(0.1, y, f"{k}: {v}", fontsize=12, color=TEXT, transform=ax.transAxes)
                y -= 0.06
        elif st == "traversal":
            ax.text(0.1, 0.68, step.get("node_name", ""), fontsize=18, fontweight="bold", color=TEXT, transform=ax.transAxes)
            wrapped = textwrap.fill(step.get("reasoning", ""), 90)
            ax.text(0.1, 0.55, wrapped, fontsize=11, color=MUTED, transform=ax.transAxes, va="top")
            ax.text(0.1, 0.22, "Descend into: " + ", ".join(step.get("descend_into", [])), fontsize=11, color=ACCENT, transform=ax.transAxes)
        elif st == "leaf_check":
            vid = step.get("view_id", "")
            t = _thumb(vid, 400)
            if t is not None:
                if step.get("found") and step.get("bbox_2d"):
                    t = _draw_bbox(t, step["bbox_2d"])
                _place_img(ax, t, (0.62, 0.45), 0.55)
            ok = step.get("found", False)
            ax.text(0.1, 0.68, f"{step.get('node_name', '')} · {vid}", fontsize=14, fontweight="bold", color=TEXT, transform=ax.transAxes)
            ax.text(0.1, 0.58, step.get("explanation", "")[:200], fontsize=11, color=MUTED, transform=ax.transAxes)
            label = "MATCH" if ok else "skip"
            bc = OK if ok else "#475569"
            ax.text(0.1, 0.12, label, fontsize=12, fontweight="bold", color="white", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=bc, edgecolor="none"))
        elif st in ("found", "not_found"):
            ax.text(0.5, 0.55, "RESULT", ha="center", fontsize=28, fontweight="bold", color=OK if found else BAD, transform=ax.transAxes)
            ax.text(0.5, 0.42, result.get("explanation", "")[:160], ha="center", fontsize=12, color=MUTED, transform=ax.transAxes)

        hold(fig, 1.8 if st == "leaf_check" else 1.2)

    # 6 Final localization
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _title(ax, "Found" if found else "Not found", f"Query: {query}")
    if view_id:
        leaf = next((s for s in steps if s.get("step_type") == "leaf_check" and s.get("view_id") == view_id and s.get("found")), None)
        t = _thumb(view_id, 520)
        if t is not None and leaf and leaf.get("bbox_2d"):
            t = _draw_bbox(t, leaf["bbox_2d"])
        if t is not None:
            _place_img(ax, t, (0.5, 0.42), 0.72)
        ax.text(0.5, 0.08, f"View {view_id} · matched: {leaf.get('matched_object', 'object') if leaf else '?'}", ha="center", fontsize=13, color=ACCENT, transform=ax.transAxes)
    hold(fig, 4.0)

    # 7 Closing
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    _title(ax, "SemanticSplat", "Graph-pruned semantic 3DGS navigation")
    ax.text(0.5, 0.3, "./start_all.sh  ·  Ask  ·  Query Flow", ha="center", fontsize=14, color=MUTED, transform=ax.transAxes)
    hold(fig, 2.5)

    return frames, fps


def write_mp4(frames: List[np.ndarray], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(path), dpi=100):
        for f in frames:
            im.set_array(f)
            writer.grab_frame()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Find the sofa")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    (ROOT / ".matplotlib").mkdir(exist_ok=True)

    print(f"Running pipeline for: {args.query!r}")
    data = _run_pipeline(args.query)
    print(f"  session {data['session_id']} · found={data.get('result', {}).get('found')} · view={data.get('result', {}).get('view_id')}")

    print("Rendering frames…")
    frames, fps = build_frames(data, args.query, args.fps)
    print(f"  {len(frames)} frames (~{len(frames)/fps:.0f}s)")

    print(f"Writing {args.output}…")
    write_mp4(frames, args.output, fps)
    print(f"Done: {args.output} ({args.output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
