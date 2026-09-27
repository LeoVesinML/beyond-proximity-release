"""Auto-write LLM-style reasoning logs from benchmark / failure-case runs.

Outputs (all created by one `./run_all_docs.sh` call):
  docs/experiments/stub/demo_runs/runs/{run_slug}/reasoning/01_*.md
  docs/experiments/stub/demo_runs/runs/{run_slug}/README.md
  docs/benchmark_results/reasoning_{scene}.json        — bundled metadata
  backend/data/scenes/{scene}/queries/auto_*.json    — session-compatible JSON
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR
from backend.io.view_store import load_view_analysis
from backend.tree.storage import load_all_nodes


def _slug(text: str, n: int = 32) -> str:
    s = re.sub(r"[^\w]+", "_", text.strip().lower()).strip("_")
    return s[:n] or "query"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _view_summary(scene_dir: str, view_id: str) -> str:
    v = load_view_analysis(scene_dir, view_id)
    if not v:
        return "(no analysis)"
    return v.scene_summary or "(empty summary)"


def build_graph_reasoning_steps(
    scene_id: str,
    query: str,
    path: List[str],
    result_view: Optional[str],
    *,
    mode: str = "graph_auto",
) -> List[Dict[str, Any]]:
    """Build traversal + leaf_check steps with reasoning text."""
    scene_dir = str(DATA_DIR / scene_id)
    nodes = load_all_nodes(scene_dir)
    steps: List[Dict[str, Any]] = []

    steps.append({
        "step_type": "decomposition",
        "structured_plan": {
            "target": _extract_target(query),
            "location_constraints": _extract_room(query),
            "query_type": "object_finding",
            "output_type": "3d_bbox",
            "source": mode,
        },
        "prompt_excerpt": f"Decompose query for semantic tree navigation: {query}",
        "reasoning": (
            f"Parsed query '{query}'. "
            f"Target keywords and room constraints extracted; navigating semantic tree "
            f"instead of scanning all views."
        ),
        "ts": _now(),
    })

    if not path:
        return steps

    for i, node_id in enumerate(path):
        node = nodes.get(node_id)
        if not node:
            continue

        next_id = path[i + 1] if i + 1 < len(path) else None
        next_node = nodes.get(next_id) if next_id else None

        # Traversal: internal node or moving to next child
        if next_node is not None:
            siblings = [nodes[cid] for cid in node.children_ids if cid in nodes]
            sibling_names = [s.name for s in siblings]
            reasoning = (
                f"At '{node.name}' ({node.node_type.value}). "
                f"Zone summary: {node.summary} "
                f"Descending into '{next_node.name}' — "
                f"summary '{next_node.summary}' matches query room/object constraints. "
                f"Siblings not taken: {', '.join(n for n in sibling_names if n != next_node.name) or 'none'}."
            )
            steps.append({
                "step_type": "traversal",
                "node_id": node_id,
                "node_name": node.name,
                "children_count": len(siblings),
                "children_names": sibling_names,
                "descend_into": [next_id],
                "reasoning": reasoning,
                "ts": _now(),
            })

        # Leaf checks on terminal node
        if next_id is None and node.view_ids:
            for vid in node.view_ids:
                summary = _view_summary(scene_dir, vid)
                found = vid == result_view
                steps.append({
                    "step_type": "leaf_check",
                    "node_id": node_id,
                    "node_name": node.name,
                    "view_id": vid,
                    "found": found,
                    "confidence": "high",
                    "bbox_2d": None,
                    "matched_object": _extract_target(query) if found else None,
                    "explanation": summary,
                    "reasoning": (
                        f"Confirm object in view {vid} within leaf '{node.name}'. "
                        + ("Match — correct zone for query." if found else "No match.")
                    ),
                    "ts": _now(),
                })
                if found:
                    break

    return steps


def build_flat_reasoning_steps(
    scene_id: str,
    query: str,
    checked_ids: List[str],
    result_view: Optional[str],
    classification: str = "",
) -> List[Dict[str, Any]]:
    """Build flat-scan steps — shows why first keyword match may be wrong room."""
    scene_dir = str(DATA_DIR / scene_id)
    steps: List[Dict[str, Any]] = []

    steps.append({
        "step_type": "decomposition",
        "structured_plan": {"mode": "flat", "query_type": "object_finding"},
        "reasoning": (
            f"Flat mode: no semantic tree. Will check every view in order for '{query}'. "
            f"First keyword match wins — room constraints may be ignored."
        ),
        "ts": _now(),
    })

    for vid in checked_ids:
        summary = _view_summary(scene_dir, vid)
        found = vid == result_view
        wrong_room = classification == "wrong_room" and found
        steps.append({
            "step_type": "leaf_check",
            "node_id": "all_views",
            "node_name": "Flat scan (no graph)",
            "view_id": vid,
            "found": found,
            "confidence": "medium" if wrong_room else ("high" if found else "high"),
            "explanation": summary,
            "reasoning": (
                f"Scanned {vid}. "
                + (
                    "First keyword match — STOP. "
                    + ("⚠ WRONG ROOM: matched object is not in the requested zone." if wrong_room else "Accepted as answer.")
                    if found
                    else "No match, continue to next view."
                )
            ),
            "ts": _now(),
        })
        if found:
            break

    return steps


_TARGET_STOPWORDS = {
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "me", "my", "can", "could", "please",
    "show", "get", "locate", "look", "take", "go",
}

_TARGET_ALIASES = {
    "collumn": "column",
    "collumns": "columns",
    "pillar": "column",
    "pillars": "columns",
}


def _normalise_target_token(token: str) -> str:
    token = _TARGET_ALIASES.get(token.lower(), token.lower())
    if len(token) > 3 and token.endswith("s") and token not in {"stairs"}:
        token = token[:-1]
    return token


def _extract_target(query: str) -> str:
    q = query.lower()
    for phrase in (
        "exit sign",
        "projection screen",
        "screen",
        "projector",
        "red sofa",
        "sofa",
        "chair",
        "bar",
        "piano",
        "column",
        "columns",
        "pillar",
        "pillars",
        "exit",
    ):
        if phrase in q:
            return _normalise_target_token(phrase.split()[-1]) if phrase in {"columns", "pillar", "pillars"} else phrase

    tokens = [
        _normalise_target_token(token)
        for token in re.findall(r"[a-z0-9]+", q)
        if token not in _TARGET_STOPWORDS
    ]
    return tokens[0] if tokens else (query.split()[0] if query.split() else query)


def _extract_room(query: str) -> Optional[str]:
    q = query.lower()
    for room in ("conference room", "ballroom", "lobby", "stage", "reception"):
        if room in q:
            return room
    return None


def _session_json(
    scene_id: str,
    query: str,
    steps: List[Dict[str, Any]],
    result: Dict[str, Any],
    session_id: str,
    source: str,
) -> dict:
    return {
        "session_id": session_id,
        "scene_id": scene_id,
        "original_query": query,
        "source": source,
        "started_at": steps[0]["ts"] if steps else _now(),
        "finished_at": _now(),
        "steps": steps,
        "result": result,
    }


def _write_reasoning_md(
    path: Path,
    query: str,
    graph_steps: List[Dict[str, Any]],
    flat_steps: List[Dict[str, Any]],
    *,
    graph_view: Optional[str],
    flat_view: Optional[str],
    graph_verdict: str,
    flat_verdict: str,
    case_index: int = 1,
    run_slug: str = "",
) -> None:
    g_icon = _verdict_label(graph_verdict)
    f_icon = _verdict_label(flat_verdict)

    lines = [
        "---",
        f"title: \"Reasoning — {query}\"",
        f"run: {run_slug}",
        f"case: {case_index}",
        f"graph_verdict: {graph_verdict}",
        f"flat_verdict: {flat_verdict}",
        "---",
        "",
        f"[← Run dashboard](../README.md) · [← Docs hub](../../../README.md) · "
        f"[Benchmark report](../../../benchmark_graph_vs_flat.md)",
        "",
        f"# Case {case_index}: {query}",
        "",
        "## Summary",
        "",
        "| | Graph search | Flat search |",
        "|--|:-------------|:------------|",
        f"| **Answer** | `{graph_view or '—'}` | `{flat_view or '—'}` |",
        f"| **Verdict** | {g_icon} `{graph_verdict}` | {f_icon} `{flat_verdict}` |",
        "",
        "## Contents",
        "",
        "- [Graph traversal](#graph-traversal)",
        "- [Flat scan](#flat-scan)",
        "- [Takeaway](#takeaway)",
        "",
        "---",
        "",
        "## Graph traversal",
        "",
        "| Step | Node / view | Decision |",
        "|-----:|-------------|----------|",
    ]

    step_n = 0
    for step in graph_steps:
        st = step["step_type"]
        if st == "decomposition":
            step_n += 1
            txt = step.get("reasoning", "")
            cell = txt if len(txt) <= 120 else txt[:117] + "…"
            lines.append(f"| {step_n} | Decomposition | {cell} |")
        elif st == "traversal":
            step_n += 1
            name = step.get("node_name", step.get("node_id", ""))
            into = ", ".join(step.get("descend_into") or [])
            lines.append(f"| {step_n} | **{name}** | Descend → `{into}` |")

    lines += ["", "### Step details", ""]
    for step in graph_steps:
        st = step["step_type"]
        if st == "decomposition":
            lines += ["#### Decomposition", "", step.get("reasoning", ""), ""]
        elif st == "traversal":
            name = step.get("node_name", step.get("node_id"))
            lines += [
                f"#### Traversal — {name}",
                "",
                step.get("reasoning", ""),
                "",
                f"**Branch taken:** `{', '.join(step.get('descend_into') or [])}`",
                "",
            ]
        elif st == "leaf_check":
            mark = "✓ match" if step.get("found") else "✗ no match"
            lines += [
                f"#### Leaf — `{step.get('view_id')}` ({mark})",
                "",
                step.get("reasoning", step.get("explanation", "")),
                "",
            ]

    lines += [
        "---",
        "",
        "## Flat scan",
        "",
        "> Flat mode scans views in order. First keyword match wins — **room constraints are not enforced**.",
        "",
        "| View | Result | Reasoning |",
        "|------|--------|-------------|",
    ]

    for step in flat_steps:
        if step["step_type"] != "leaf_check":
            continue
        vid = step.get("view_id", "—")
        if step.get("found"):
            res = "✓ **STOP**"
        else:
            res = "skip"
        reason = (step.get("reasoning") or "")[:100]
        if len(step.get("reasoning") or "") > 100:
            reason += "…"
        lines.append(f"| `{vid}` | {res} | {reason} |")

    takeaway = _takeaway(graph_verdict, flat_verdict, query)
    lines += [
        "",
        "---",
        "",
        "## Takeaway",
        "",
        takeaway,
        "",
        "---",
        "",
        f"[← Back to run dashboard](../README.md)",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _verdict_label(verdict: str) -> str:
    return {
        "correct_room": "✅",
        "correct_zone": "✅",
        "wrong_room": "❌",
        "miss": "⚠️",
    }.get(verdict, "·")


def _takeaway(g_verdict: str, f_verdict: str, query: str) -> str:
    if f_verdict == "wrong_room" and g_verdict in ("correct_room", "correct_zone"):
        return (
            f"**Graph wins on room scope.** Flat found an object matching keywords but in the "
            f"wrong zone. Graph traversal scoped the search to the correct branch of the semantic "
            f"tree before confirming the view."
        )
    if g_verdict == "miss" and f_verdict == "miss":
        return (
            "Both modes missed — likely a simulation/oracle gap. Run with a real LLM session "
            "via Query Flow for this query."
        )
    if g_verdict == f_verdict == "correct_room":
        return "Both found the correct room; graph checked fewer views."
    return f"Query `{query}` — compare graph path vs flat scan above."


def write_benchmark_reasoning(
    scene_id: str,
    failure_report: dict,
    *,
    docs_dir: Path,
    run_slug: str,
) -> Dict[str, Any]:
    """Write reasoning MD + JSON session files for all failure cases."""
    reasoning_dir = (
        docs_dir / "experiments" / "stub" / "demo_runs" / "runs" / run_slug / "reasoning"
    )
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = DATA_DIR / scene_id / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    cases_out: List[Dict[str, Any]] = []
    md_paths: List[str] = []
    session_paths: List[str] = []

    for idx, row in enumerate(failure_report.get("results", []), 1):
        query = row["case"]["query"]
        g = row["graph"]
        f = row["flat"]
        qslug = _slug(query, 28)

        graph_steps = build_graph_reasoning_steps(
            scene_id, query, g.get("path") or [], g.get("view_id"), mode="graph_auto",
        )
        flat_steps = build_flat_reasoning_steps(
            scene_id, query, f.get("views_checked_ids") or [], f.get("view_id"),
            classification=f.get("classification", ""),
        )

        sid = hashlib.sha1(f"{run_slug}:{query}".encode()).hexdigest()[:8]
        session_id = f"auto_{sid}"

        graph_key = f"{session_id}_graph"
        flat_key = f"{session_id}_flat"
        graph_session = _session_json(
            scene_id, query, graph_steps,
            {"found": g.get("found"), "view_id": g.get("view_id"), "mode": "graph_auto",
             "explanation": graph_steps[-1].get("explanation", "") if graph_steps else ""},
            graph_key, "benchmark_auto_graph",
        )
        flat_session = _session_json(
            scene_id, query, flat_steps,
            {"found": f.get("found"), "view_id": f.get("view_id"), "mode": "flat",
             "explanation": flat_steps[-1].get("explanation", "") if flat_steps else ""},
            flat_key, "benchmark_auto_flat",
        )

        gpath = queries_dir / f"{graph_key}.json"
        fpath = queries_dir / f"{flat_key}.json"
        with open(gpath, "w", encoding="utf-8") as fh:
            json.dump(graph_session, fh, indent=2, ensure_ascii=False)
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(flat_session, fh, indent=2, ensure_ascii=False)
        session_paths.extend([
            f"backend/data/scenes/{scene_id}/queries/{gpath.name}",
            f"backend/data/scenes/{scene_id}/queries/{fpath.name}",
        ])

        md_name = f"{idx:02d}_{qslug}.md"
        md_path = reasoning_dir / md_name
        _write_reasoning_md(
            md_path, query, graph_steps, flat_steps,
            graph_view=g.get("view_id"),
            flat_view=f.get("view_id"),
            graph_verdict=g.get("classification", "?"),
            flat_verdict=f.get("classification", "?"),
            case_index=idx,
            run_slug=run_slug,
        )
        rel = str(md_path.relative_to(docs_dir))
        md_paths.append(rel)

        cases_out.append({
            "case_index": idx,
            "query": query,
            "graph_verdict": g.get("classification"),
            "flat_verdict": f.get("classification"),
            "reasoning_md": rel,
            "session_graph": str(gpath.name),
            "session_flat": str(fpath.name),
        })

    bundle = {
        "scene_id": scene_id,
        "generated_at": _now(),
        "run_slug": run_slug,
        "cases": cases_out,
        "reasoning_md_dir": str(reasoning_dir.relative_to(docs_dir)),
        "run_dashboard": f"experiments/stub/demo_runs/runs/{run_slug}/README.md",
    }
    bundle_path = docs_dir / "benchmark_results" / f"reasoning_{scene_id}.json"
    with open(bundle_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False)

    return {
        "bundle_path": str(bundle_path.relative_to(docs_dir)),
        "reasoning_md_paths": md_paths,
        "session_paths": session_paths,
        "cases": cases_out,
        "run_slug": run_slug,
        "run_dashboard": f"experiments/stub/demo_runs/runs/{run_slug}/README.md",
    }
