"""Week 1 headless pipeline entry point.

Stub mode (default) reuses existing view analyses and tree when available.
Live mode is reserved for future VLM/LLM integration without Cursor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "find", "near", "where",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def import_scene(project_root: Path, scene_path: Path, scene_id: str, copy_from: str | None) -> None:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "import_replica_scene.py"),
        "--input",
        str(scene_path),
        "--scene-id",
        scene_id,
        "--backend-root",
        str(project_root),
    ]
    if copy_from:
        cmd.extend(["--copy-annotations-from", copy_from])
    subprocess.run(cmd, check=True)


def run_stub_pipeline(project_root: Path, scene_id: str, query: str, out_dir: Path) -> dict[str, Any]:
    from backend.config import DATA_DIR
    from backend.io.view_store import list_view_summaries
    from backend.mcp.tools.tree_tools import get_tree_for_viz, search_summaries
    from backend.tree.storage import get_root_node_id, load_all_nodes

    logs: list[dict[str, Any]] = []
    scene_dir = DATA_DIR / scene_id
    if not (scene_dir / "transforms.json").exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    t0 = time.time()
    logs.append({"step": "load_scene", "scene_id": scene_id, "status": "ok", "at": utc_now()})

    metadata_path = scene_dir / "source_metadata.json"
    dataset_warnings: list[str] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            dataset_warnings = json.load(fh).get("warnings", [])
        if dataset_warnings:
            logs.append({"step": "dataset_warnings", "warnings": dataset_warnings, "at": utc_now()})

    views = list_view_summaries(str(scene_dir))
    if not views:
        raise RuntimeError(
            f"No analyzed views found for scene '{scene_id}'. "
            "Run import with --copy-annotations-from default in stub mode, "
            "or implement live VLM analysis."
        )
    logs.append({"step": "load_views", "count": len(views), "status": "ok", "at": utc_now()})

    tree_viz = get_tree_for_viz(scene_id)
    if "error" in tree_viz:
        raise RuntimeError(tree_viz["error"])
    nodes = load_all_nodes(str(scene_dir))
    root_id = get_root_node_id(str(scene_dir))
    logs.append({
        "step": "load_tree",
        "node_count": len(nodes),
        "root_id": root_id,
        "status": "ok",
        "at": utc_now(),
    })

    keywords = [
        w.strip(".,!?\"'")
        for w in query.lower().split()
        if len(w.strip(".,!?\"'")) > 2 and w.strip(".,!?\"'") not in STOPWORDS
    ]
    search_hits: list[dict[str, Any]] = []
    for kw in keywords:
        search_hits.extend(search_summaries(scene_id, kw))

    seen = set()
    unique_hits: list[dict[str, Any]] = []
    for hit in search_hits:
        key = (hit.get("type"), hit.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(hit)

    def hit_text(hit: dict[str, Any]) -> str:
        return " ".join(
            str(hit.get(field, "") or "")
            for field in ("name", "summary", "id")
        ).lower()

    ranked_hits = sorted(
        unique_hits,
        key=lambda hit: sum(1 for kw in keywords if kw in hit_text(hit)),
        reverse=True,
    )
    if not keywords:
        best = None
    else:
        best = ranked_hits[0] if ranked_hits and any(kw in hit_text(ranked_hits[0]) for kw in keywords) else None
    found = best is not None
    answer_text = best["summary"] if best and best.get("type") == "view" else (
        best["name"] if best else "No match found"
    )

    query_result = {
        "method": "semantic_splat_stub",
        "scene_id": scene_id,
        "query": query,
        "found": found,
        "answer_text": answer_text,
        "view_id": best.get("id") if best and best.get("type") == "view" else None,
        "node_id": best.get("id") if best and best.get("type") == "node" else None,
        "bbox_2d": None,
        "bbox_3d": None,
        "confidence": 0.5 if found else 0.0,
        "trace": [
            {"step": "keyword_search", "keywords": keywords, "hits": len(unique_hits)},
            {"step": "ranked_matches", "top": ranked_hits[:5]},
            {"step": "best_match", "match": best},
        ],
        "warnings": [
            "Stub mode uses keyword search over existing summaries.",
            "3D bbox is disabled until live VLM + valid depth are wired in.",
            *dataset_warnings,
        ],
    }

    elapsed = round(time.time() - t0, 3)
    logs.append({"step": "query", "found": found, "elapsed_seconds": elapsed, "at": utc_now()})

    write_json(out_dir / "views.json", {"scene_id": scene_id, "views": views})
    write_json(out_dir / "tree.json", {"scene_id": scene_id, "root_id": root_id, "tree": tree_viz})
    write_json(out_dir / "query_result.json", query_result)
    write_json(out_dir / "logs.json", {"scene_id": scene_id, "query": query, "events": logs})
    write_json(out_dir / "run_config.json", {
        "mode": "stub",
        "scene_id": scene_id,
        "query": query,
        "started_at": logs[0]["at"],
        "finished_at": logs[-1]["at"],
        "elapsed_seconds": elapsed,
    })

    return query_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SemanticSplat Week 1 headless pipeline.")
    parser.add_argument("--scene", required=True, help="Input scene folder in unified format.")
    parser.add_argument("--scene-id", default="replica_pilot_001", help="Backend scene id to create/use.")
    parser.add_argument("--query", required=True, help='Natural language query, e.g. "find the chair near the table".')
    parser.add_argument("--out", required=True, help="Output directory, e.g. outputs/week1_smoke_test.")
    parser.add_argument("--mode", choices=("stub", "live"), default="stub")
    parser.add_argument(
        "--copy-annotations-from",
        default="default",
        help="In stub mode, copy views/tree from this existing backend scene after import.",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Do not re-import scene; assume backend scene already exists.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    try:
        if not args.skip_import:
            import_scene(
                project_root,
                Path(args.scene).resolve(),
                args.scene_id,
                args.copy_annotations_from if args.mode == "stub" else None,
            )

        if args.mode == "stub":
            result = run_stub_pipeline(project_root, args.scene_id, args.query, out_dir)
        else:
            raise NotImplementedError(
                "Live mode is not implemented yet. Use --mode stub for Week 1 smoke test."
            )

        print(f"Pipeline finished. found={result['found']} answer={result['answer_text']!r}")
        print(f"Outputs written to {out_dir}")
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "errors.json", {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "at": utc_now(),
        })
        raise


if __name__ == "__main__":
    main()
