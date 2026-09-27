#!/usr/bin/env python3
"""Generate qualitative and overview figures for the workshop paper."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "twinworld" / "figures"

GRAPH_C = "#0b559f"
FLAT_C = "#c1440e"
OK_C = "#2a7f4f"
GRAY = "#6b7280"
SOFT = "#eef2f7"


def _setup():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        print(f"Wrote {p}")


def render_system_overview() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch

    scene = ROOT / "backend" / "data" / "scenes" / "ConferenceHall-capture-pilot"
    view = json.loads((scene / "views" / "v018.json").read_text(encoding="utf-8"))
    image = _open_rgb(scene / "images" / "v018.png")
    if image is None:
        raise SystemExit("ConferenceHall v018 is required for the method figure.")
    rgb = np.asarray(image)
    depth = np.load(scene / "depths" / "v018_depth.npy")
    result_file = ROOT / "outputs" / "graph_vs_flat" / "five_scene_graph_vs_flat_v2" / "per_query_results.json"
    result_data = json.loads(result_file.read_text(encoding="utf-8"))
    query_result = next(row for row in result_data["queries"] if row["query_id"] == "qv2_010")
    construction = json.loads(
        (ROOT / "docs" / "reports" / "final" / "graph_construction_cost.json").read_text(encoding="utf-8")
    )
    target = next(obj for obj in view["visible_objects"] if obj["label"] == "covered chairs")

    fig = plt.figure(figsize=(7.2, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        12,
        left=0.025,
        right=0.985,
        bottom=0.065,
        top=0.875,
        hspace=0.62,
        wspace=0.82,
    )
    ax_rgb = fig.add_subplot(grid[0, 0:3])
    ax_index = fig.add_subplot(grid[0, 3:7])
    ax_graph = fig.add_subplot(grid[0, 7:12])
    ax_query = fig.add_subplot(grid[1, 0:3])
    ax_search = fig.add_subplot(grid[1, 3:8])
    ax_output = fig.add_subplot(grid[1, 8:12])

    def panel_title(ax, letter: str, title: str) -> None:
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=7.0, fontweight="bold", pad=4)

    def connect(
        ax_a,
        xy_a,
        ax_b,
        xy_b,
        *,
        color="#64748b",
        arrowstyle="-|>",
        lw=1.0,
    ) -> None:
        """Connect panels using their transforms instead of figure pixel guesses."""
        fig.add_artist(ConnectionPatch(
            xyA=xy_a,
            xyB=xy_b,
            coordsA=ax_a.transAxes,
            coordsB=ax_b.transAxes,
            arrowstyle=arrowstyle,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=20,
            clip_on=False,
        ))

    # A: real RGB-D input. The inset is rendered directly from the captured depth array.
    ax_rgb.imshow(rgb)
    panel_title(ax_rgb, "A", "Posed RGB-D capture")
    ax_rgb.axis("off")
    inset = ax_rgb.inset_axes([0.57, 0.04, 0.4, 0.43])
    finite = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(finite, [2, 98])
    inset.imshow(np.clip(depth, lo, hi), cmap="cividis", vmin=lo, vmax=hi)
    inset.set_title("measured depth", fontsize=5.3, color="white", pad=1,
                    backgroundcolor="#111827")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)

    # B: actual ViewJSON annotation and coarse depth-projected box.
    ax_index.imshow(rgb)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = target["bbox_2d"]
    ax_index.add_patch(patches.Rectangle(
        (x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
        linewidth=1.7, edgecolor="#f5c542", facecolor="none",
    ))
    ax_index.text(
        x0 * w + 4,
        max(y0 * h - 6, 4),
        "covered chairs  |  white  |  v018",
        va="bottom",
        fontsize=5.2,
        color="#111827",
        bbox=dict(
            boxstyle="round,pad=0.22,rounding_size=0.08",
            facecolor="#fff7d6",
            edgecolor="#f5c542",
            linewidth=0.8,
        ),
        clip_on=True,
        zorder=4,
    )
    panel_title(ax_index, "B", "Semantic ViewJSON record")
    ax_index.axis("off")

    # C: a compact, data-backed hierarchy slice from the checked-in scene graph.
    ax_graph.set_xlim(0, 11)
    ax_graph.set_ylim(0, 7.2)
    ax_graph.axis("off")
    panel_title(ax_graph, "C", "Deterministic hierarchy assembly")
    nodes = {
        "root": (5.5, 6.15, "scene root\n206 nodes", True, 2.15, 0.86),
        "lounge": (1.75, 4.55, "lounge / corridor", False, 2.75, 0.74),
        "banquet": (5.5, 4.55, "banquet hall", True, 2.35, 0.74),
        "exit": (9.25, 4.55, "exit / wayfinding", False, 2.75, 0.74),
        "tables": (3.65, 2.72, "round tables", False, 2.25, 0.74),
        "chairs": (7.2, 2.72, "covered chairs", True, 2.55, 0.74),
        "view": (7.2, 1.08, "v018 evidence", True, 2.25, 0.74),
    }
    edges = [
        ("root", "lounge"), ("root", "banquet"), ("root", "exit"),
        ("banquet", "tables"), ("banquet", "chairs"), ("chairs", "view"),
    ]
    def node_edge(source: str, target_name: str):
        x_a, y_a, _, _, width_a, height_a = nodes[source]
        x_b, y_b, _, _, _, _ = nodes[target_name]
        dx, dy = x_b - x_a, y_b - y_a
        tx = (width_a / 2) / abs(dx) if dx else float("inf")
        ty = (height_a / 2) / abs(dy) if dy else float("inf")
        scale = min(tx, ty)
        return x_a + dx * scale, y_a + dy * scale

    for parent, child in edges:
        _, _, _, on_a, _, _ = nodes[parent]
        _, _, _, on_b, _, _ = nodes[child]
        active = on_a and on_b
        ax_graph.add_patch(FancyArrowPatch(
            node_edge(parent, child),
            node_edge(child, parent),
            arrowstyle="-|>",
            mutation_scale=8,
            lw=1.2 if active else 0.8,
            color=OK_C if active else "#cbd5e1",
            shrinkA=1,
            shrinkB=2,
            zorder=1,
        ))
    for _, (x, y, label, active, width, height) in nodes.items():
        ax_graph.add_patch(FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#e7f5ee" if active else "#f3f4f6",
            edgecolor=OK_C if active else "#9ca3af",
            lw=1.0,
            zorder=3,
        ))
        ax_graph.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            color="#111827" if active else GRAY,
            zorder=4,
        )
    totals = construction["totals"]
    ax_graph.text(
        5.5, 0.08,
        f"five scenes: {totals['view_count']} views  |  {totals['semantic_item_count']:,} items  |  "
        f"{totals['source_viewjson_estimated_tokens_chars_div_4']:,} input / "
        f"{totals['constructed_tree_estimated_tokens_chars_div_4']:,} tree token-equiv.",
        ha="center", va="bottom", fontsize=5.25, color=GRAY,
    )

    # D: the exact saved benchmark query and deterministic parse.
    ax_query.set_xlim(0, 1)
    ax_query.set_ylim(0, 1)
    ax_query.axis("off")
    panel_title(ax_query, "D", "Natural-language query")
    ax_query.add_patch(FancyBboxPatch(
        (0.05, 0.49), 0.90, 0.31, boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#eef6ff", edgecolor=GRAPH_C, lw=1.1,
    ))
    ax_query.text(0.5, 0.645, '"Find the white\ncovered chairs"', ha="center", va="center",
                  fontsize=6.9, fontweight="bold")
    ax_query.text(0.07, 0.29, "target", fontsize=5.25, color=GRAY)
    ax_query.text(0.39, 0.29, "covered chairs", fontsize=5.55, color="#111827")
    ax_query.text(0.07, 0.13, "attribute", fontsize=5.25, color=GRAY)
    ax_query.text(0.39, 0.13, "white", fontsize=5.55, color="#111827")

    # E: same-input graph/flat comparison from qv2_010.
    ax_search.set_xlim(0, 11.2)
    ax_search.set_ylim(0, 6)
    ax_search.axis("off")
    panel_title(ax_search, "E", "Graph-pruned vs. flat traversal")
    graph = query_result["graph"]
    flat = query_result["flat"]
    lanes = [
        (4.2, "Graph", OK_C, "root", "banquet zone", f"{graph['semantic_entries_scanned']} entries", graph["selected_view_id"]),
        (1.75, "Flat", FLAT_C, "all records", "no pruning", f"{flat['semantic_entries_scanned']} entries", flat["selected_view_id"]),
    ]
    split_x = 0.12
    label_left, label_width = 0.30, 1.05
    box_half_width, box_half_height = 0.76, 0.40
    positions_x = (2.25, 4.95, 7.65, 10.35)
    ax_search.plot([split_x, split_x], [lanes[1][0], lanes[0][0]], color="#94a3b8", lw=0.8, zorder=1)
    for y, label, color, a, b, c, d in lanes:
        ax_search.add_patch(FancyBboxPatch(
            (label_left, y - 0.32),
            label_width,
            0.64,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        ))
        ax_search.text(
            label_left + label_width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=5.4,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        ax_search.add_patch(FancyArrowPatch(
            (split_x, y),
            (label_left - 0.04, y),
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=0.9,
            zorder=2,
        ))
        positions = list(zip(positions_x, (a, b, c, d)))
        ax_search.add_patch(FancyArrowPatch(
            (label_left + label_width + 0.03, y),
            (positions_x[0] - box_half_width - 0.04, y),
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=0.9,
            zorder=2,
        ))
        for idx, (x, text_value) in enumerate(positions):
            ax_search.add_patch(FancyBboxPatch(
                (x - box_half_width, y - box_half_height),
                box_half_width * 2,
                box_half_height * 2,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#f3faf6" if color == OK_C else "#fff7f2",
                edgecolor=color,
                lw=0.9,
                zorder=3,
            ))
            ax_search.text(x, y, text_value, ha="center", va="center", fontsize=4.9, zorder=4)
            if idx < len(positions) - 1:
                ax_search.add_patch(FancyArrowPatch(
                    (x + box_half_width + 0.04, y),
                    (positions[idx + 1][0] - box_half_width - 0.04, y),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=color,
                    lw=0.9,
                    zorder=2,
                ))
    ax_search.text(
        5.75, 0.38,
        f"views: {graph['views_checked']} vs {flat['views_checked']}  |  "
        f"estimated context: {graph['input_tokens']:,} vs {flat['input_tokens']:,} tokens",
        ha="center", fontsize=5.15, color=GRAY,
    )

    # F: evidence view and result box, both sourced from the saved run/ViewJSON.
    ax_output.imshow(rgb)
    ax_output.add_patch(patches.Rectangle(
        (x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
        linewidth=1.8, edgecolor=OK_C, facecolor="none",
    ))
    ax_output.text(
        0.025, 0.96, "covered chairs  |  v018",
        transform=ax_output.transAxes, va="top", fontsize=5.35, color="white",
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.08", facecolor=OK_C, edgecolor="none"),
    )
    panel_title(ax_output, "F", "Grounded evidence")
    ax_output.axis("off")
    ax_output.text(
        0.5, -0.08,
        f"graph hit@1: yes  |  {flat['elapsed_ms'] / graph['elapsed_ms']:.2f}x speedup  |  "
        f"{query_result['savings_graph_vs_flat']['tokens_pct']:.1f}% fewer est. tokens",
        transform=ax_output.transAxes, ha="center", va="top", fontsize=5.0, color=GRAY,
    )

    # Panel-aware connectors remain aligned if the grid or image aspect ratio changes.
    connect(ax_rgb, (1.02, 0.50), ax_index, (-0.02, 0.50))
    connect(ax_index, (1.02, 0.50), ax_graph, (-0.02, 0.50))
    fig.add_artist(ConnectionPatch(
        xyA=(1.02, 0.645),
        xyB=(split_x, 2.975),
        coordsA=ax_query.transAxes,
        coordsB=ax_search.transData,
        arrowstyle="-",
        linewidth=1.0,
        color="#64748b",
        shrinkA=2,
        shrinkB=0,
        zorder=20,
        clip_on=False,
    ))
    fig.add_artist(ConnectionPatch(
        xyA=(positions_x[-1] + box_half_width, lanes[0][0]),
        xyB=(-0.02, 0.64),
        coordsA=ax_search.transData,
        coordsB=ax_output.transAxes,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.1,
        color=OK_C,
        shrinkA=2,
        shrinkB=2,
        zorder=20,
        clip_on=False,
    ))

    fig.text(0.015, 0.925, "ONE-TIME MAP CONSTRUCTION", fontsize=6.0, fontweight="bold", color=GRAPH_C)
    fig.text(0.015, 0.475, "PER-QUERY RETRIEVAL", fontsize=6.0, fontweight="bold", color=GRAPH_C)
    fig.text(
        0.5,
        0.985,
        "Agent-guided hierarchy: from captured evidence to grounded retrieval",
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )
    _save(fig, "fig_method_workflow")
    plt.close(fig)


def render_evaluation_protocol() -> None:
    """Render the controlled same-map graph-vs-flat evaluation protocol."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, face, edge, size=6.4):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=1.0,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)

    box(0.25, 1.6, 2.3, 1.8, "Frozen semantic map\n+ fixed query + GT", "#eef2f7", GRAPH_C)
    box(3.4, 3.0, 2.25, 1.25, "Graph traversal\npruned candidates", "#e7f5ee", OK_C)
    box(3.4, 0.75, 2.25, 1.25, "Flat search\nall candidates", "#fff1e8", FLAT_C)
    box(6.55, 3.0, 2.35, 1.25, "4.77 views/query\n1,014 est. tokens", "white", OK_C)
    box(6.55, 0.75, 2.35, 1.25, "19.4 views/query\n3,168 est. tokens", "white", FLAT_C)
    box(9.75, 1.6, 2.0, 1.8, "Paired quality + cost\nHit@k, scans, tokens,\nlatency, bootstrap CI", "#f8fafc", "#475569", 5.9)

    arrows = [
        ((2.55, 2.5), (3.35, 3.62), OK_C),
        ((2.55, 2.5), (3.35, 1.37), FLAT_C),
        ((5.65, 3.62), (6.5, 3.62), OK_C),
        ((5.65, 1.37), (6.5, 1.37), FLAT_C),
        ((8.9, 3.62), (9.7, 2.75), OK_C),
        ((8.9, 1.37), (9.7, 2.25), FLAT_C),
    ]
    for start, end, color in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0))

    ax.text(6, 4.75, "Controlled comparison: only the search structure changes",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text(6, 0.12,
            "Internal five-scene means; context tokens are deterministic characters/4 estimates, not API billing.",
            ha="center", fontsize=5.6, color=GRAY)
    _save(fig, "fig_evaluation_protocol")
    plt.close(fig)


def render_tree_traversal() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.35, 7)
    ax.axis("off")

    nodes = {
        "root": (5.0, 6.25, "Scene root"),
        "lobby": (1.4, 4.95, "Lobby\nzone"),
        "ball": (5.0, 4.95, "Ballroom\nzone"),
        "corr": (8.6, 4.95, "Corridor\nzone"),
        "stage": (3.9, 3.55, "Stage\nregion"),
        "floor": (6.1, 3.55, "Seating\nregion"),
        "screen": (3.9, 2.15, "Projection screen\nobject"),
        "chairs": (6.1, 2.15, "Chair group\nobjects"),
        "v018": (3.9, 0.75, "v018\nview evidence"),
        "v012": (6.1, 0.75, "v012\nview evidence"),
    }
    keep = {"root", "ball", "stage", "screen", "v018"}

    def draw_node(key):
        x, y, label = nodes[key]
        if key in keep:
            face, edge = "#e7f5ee", OK_C
        else:
            face, edge = "#f3f4f6", "#9ca3af"
        ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.43), 2.0, 0.86,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=face, edgecolor=edge, lw=1.2,
                                    linestyle="-" if key in keep else "--"))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.8,
                color="#111" if key in keep else "#6b7280")

    for k in nodes:
        draw_node(k)

    edges = [
        ("root", "lobby", False), ("root", "ball", True), ("root", "corr", False),
        ("ball", "stage", True), ("ball", "floor", False),
        ("stage", "screen", True), ("floor", "chairs", False),
        ("screen", "v018", True), ("chairs", "v012", False),
    ]
    for a, b, on in edges:
        x0, y0, _ = nodes[a]
        x1, y1, _ = nodes[b]
        ax.add_patch(FancyArrowPatch((x0, y0 - 0.43), (x1, y1 + 0.43),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=OK_C if on else "#cbd5e1",
                                     lw=1.6 if on else 0.9,
                                     linestyle="-" if on else "--"))

    ax.text(5, 6.88, 'Query: "where is the screen in the conference room?"',
            ha="center", fontsize=9.2, fontweight="bold")
    ax.text(0.08, -0.27, "Solid green = retained target path; dashed gray = pruned branch",
            fontsize=7.2, color=GRAY, va="bottom")
    _save(fig, "fig_tree_traversal")
    plt.close(fig)


def _open_rgb(path: Path):
    """Open image if it is a real raster file (skip Git LFS pointer stubs)."""
    from PIL import Image

    if not path.exists() or path.stat().st_size < 1024:
        return None
    head = path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs.github.com"):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def render_scene_gallery() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    picks = [
        ("ConferenceHall-capture-pilot", "v001.png", "Conference Hall"),
        ("Museume-capture", "v001.png", "Museum"),
        ("Theater-capture", "v001.png", "Theater"),
        ("outdoor-drone-capture", "v001.png", "Outdoor drone"),
        ("outdoor-street-capture", "v001.png", "Outdoor street"),
        ("default", "v018.png", "Demo Conference (default)"),
    ]
    panels = []
    for scene, fname, title in picks:
        p = ROOT / "backend" / "data" / "scenes" / scene / "images" / fname
        img = _open_rgb(p)
        if img is not None:
            panels.append((img, title))
    if len(panels) < 3:
        for name in ("v012.png", "v013.png", "v018.png"):
            p = ROOT / "backend" / "data" / "scenes" / "default" / "images" / name
            img = _open_rgb(p)
            if img is not None:
                panels.append((img, f"Demo {name[:-4]}"))
    if not panels:
        raise SystemExit("No readable scene images for gallery (git lfs pull may be required).")
    panels = panels[:6]
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.35 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle("Digital-twin pilot scenes (manual captures)", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "fig_scene_gallery")
    plt.close(fig)


def render_qualitative_bbox() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    scene = ROOT / "backend" / "data" / "scenes" / "default"
    view = json.loads((scene / "views" / "v018.json").read_text(encoding="utf-8"))
    pil = _open_rgb(scene / "images" / "v018.png")
    if pil is None:
        print("Skip qualitative: missing/unreadable v018.png (git lfs pull?)")
        return
    img = np.asarray(pil)
    h, w = img.shape[:2]

    # Find projection screen bbox
    target = None
    for obj in view.get("objects", []):
        if "screen" in str(obj.get("label", "")).lower():
            target = obj
            break
    if target is None and view.get("objects"):
        target = view["objects"][0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # Left: correct view with bbox
    axes[0].imshow(img)
    if target and target.get("bbox_2d"):
        x0, y0, x1, y1 = target["bbox_2d"]
        rect = patches.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                                 linewidth=2.0, edgecolor=OK_C, facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x0 * w, max(8, y0 * h - 8),
                     f"{target.get('label', 'object')} (graph path)",
                     color="white", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor=OK_C, edgecolor="none"))
    axes[0].set_title("Graph: v018 stage / conference end")
    axes[0].axis("off")

    # Right: alternative wrong-room view if exists (v012)
    wrong_pil = _open_rgb(scene / "images" / "v012.png")
    if wrong_pil is not None:
        wimg = np.asarray(wrong_pil)
        axes[1].imshow(wimg)
        axes[1].text(0.02, 0.97, "Flat full-scan distractor view\n(ballroom floor through doors)",
                     transform=axes[1].transAxes, va="top", fontsize=7.5, color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=FLAT_C, edgecolor="none"))
        axes[1].set_title("Flat risk: near-miss screen view")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "v012 unavailable", ha="center")
    axes[1].axis("off")

    fig.suptitle('Qualitative case: "screen in the conference room"', fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_qualitative_bbox")
    plt.close(fig)


def main() -> int:
    _setup()
    render_system_overview()
    render_evaluation_protocol()
    render_tree_traversal()
    render_scene_gallery()
    render_qualitative_bbox()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
