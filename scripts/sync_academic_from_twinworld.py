#!/usr/bin/env python3
"""Render the canonical TwinWorld manuscript as a named academic preprint."""
from __future__ import annotations

import re
import shutil
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "papers" / "twinworld"
TARGET = ROOT / "papers" / "beyond-proximity"

AUTHORS = (
    "Andrey Kuznetsov",
    "Mahmoud Mousatat",
    "Ahmad Taha",
    "Aleksandr Medvedev",
    "Leo Vesin",
    "Telman Nuruzov",
    "Nikita Shankin",
)

REVIEW_STYLE = r"""\documentclass[runningheads]{llncs}
\ifdefined\TwinWorldPreprint
  \usepackage{eccv}
\else
  % review_id.tex is private and ignored; OpenReview assigns this number.
  \input{review_id.tex}
  \usepackage[review,year=2026,ID=\TwinWorldPaperID]{eccv}
\fi"""

PREPRINT_STYLE = r"""\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage{cvpr}"""

REVIEW_HYPERREF = r"""\ifdefined\TwinWorldPreprint
  \usepackage[breaklinks,colorlinks,citecolor=eccvblue]{hyperref}
\else
  \usepackage[pagebackref,breaklinks,colorlinks,citecolor=eccvblue]{hyperref}
\fi"""

PREPRINT_HYPERREF = r"""\usepackage{balance}
\usepackage{dblfloatfix}
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue,
  filecolor=blue
}
\setlength{\columnsep}{18pt}"""

ANONYMOUS_AUTHORS = r"""\author{Anonymous Authors}
\authorrunning{Anonymous}
\institute{Anonymous Institution}"""

NAMED_AUTHORS = r"""\author{
Andrey Kuznetsov, Mahmoud Mousatat, Ahmad Taha, Aleksandr Medvedev,\\
Leo Vesin, Telman Nuruzov, Nikita Shankin
}
\date{Academic preprint, August 2026}"""

SUPPORT_FILES = (
    "cvpr.sty",
    "refs.bib",
    "tables/new_experiment_macros.tex",
    "tables/reviewer_revision_macros.tex",
)


def render_source(source: str) -> str:
    source = source.replace(
        "% papers/twinworld/main.tex\n% Anonymous ECCV 2026 review build. Review mode supplies line numbers.",
        "% Named academic preprint generated from the canonical review manuscript.\n"
        "% Edit the canonical review manuscript, then run the academic sync script.",
        1,
    )
    for old, new, label in (
        (REVIEW_STYLE, PREPRINT_STYLE, "review style preamble"),
        (REVIEW_HYPERREF, PREPRINT_HYPERREF, "review hyperlink preamble"),
        (ANONYMOUS_AUTHORS, NAMED_AUTHORS, "anonymous author block"),
    ):
        if old not in source:
            raise ValueError(f"canonical manuscript is missing the expected {label}")
        source = source.replace(old, new, 1)

    source = source.replace(
        "\\titlerunning{Beyond Proximity: Graph-Pruned Semantic Search}\n\n", "", 1
    )
    source = re.sub(
        r"\\keywords\{(.*?)\}\n\\end\{abstract\}",
        lambda match: (
            "\\end{abstract}\n\n"
            "\\noindent\\textbf{Keywords:} "
            + re.sub(r"\s*\\and\s*", "; ", match.group(1).replace("\n", " ")).strip()
            + "\\par\\medskip"
        ),
        source,
        count=1,
        flags=re.DOTALL,
    )

    source = re.sub(
        r"\\begin\{figure\}\[[tH]\]", r"\\begin{figure*}[!tbp]", source
    )
    source = source.replace(r"\end{figure}", r"\end{figure*}")

    def table_in_shared_float_queue(match: re.Match[str]) -> str:
        body = match.group(1).replace(r"\caption{", r"\captionof{table}{", 1)
        return "\\begin{figure*}[!tbp]\n" + body + "\\end{figure*}"

    source = re.sub(
        r"\\begin\{table\}\[[tH]\](.*?)\\end\{table\}",
        table_in_shared_float_queue,
        source,
        flags=re.DOTALL,
    )

    figure_widths = {
        "figures/fig_tree_traversal.pdf": "0.78\\textwidth",
        "figures/fig_agent_mcp_workflow.pdf": "0.98\\textwidth",
        "figures/fig_scene_gallery.pdf": "0.96\\textwidth",
        "figures/fig_instruction_agent.pdf": "0.96\\textwidth",
        "figures/fig_raw_rgbd_hierarchy.pdf": "0.90\\textwidth",
        "figures/fig_hierarchy_construction_variants.pdf": "0.94\\textwidth",
        "figures/fig_reviewer_controls.pdf": "0.94\\textwidth",
        "figures/fig_reviewer_scaling.pdf": "0.90\\textwidth",
        "figures/fig_five_scene_summary.pdf": "0.84\\textwidth",
    }
    for figure, width in figure_widths.items():
        source = re.sub(
            rf"\\includegraphics\[width=[^]]+\]\{{{re.escape(figure)}\}}",
            lambda _match, f=figure, w=width: f"\\includegraphics[width={w}]{{{f}}}",
            source,
            count=1,
        )

    # Figures and tables share one double-column float queue, preserving their
    # source order. Let prose flow under the first two result figures, then
    # flush the complete experiment sequence before the discussion.
    source = source.replace(
        "\\FloatBarrier\n\\subsection{Semantic-record construction variants}",
        "\\subsection{Semantic-record construction variants}",
        1,
    )
    source = source.replace(
        "\\FloatBarrier\n\\subsection{Deterministic same-input controls}",
        "\\subsection{Deterministic same-input controls}",
        1,
    )
    source = source.replace(
        "\\section{Discussion and Limitations}",
        "\\FloatBarrier\n\\section{Discussion and Limitations}",
        1,
    )

    source = source.replace(
        r"\bibliographystyle{splncs04}", "\\balance\n\\bibliographystyle{abbrv}"
    )
    return source


def referenced_figures(source: str) -> tuple[str, ...]:
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    return tuple(dict.fromkeys(figures))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".sync-tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except OSError:
            if attempt == 4:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))


def sync() -> dict:
    canonical = (SOURCE / "main.tex").read_text(encoding="utf-8")
    rendered = render_source(canonical)

    TARGET.mkdir(parents=True, exist_ok=True)
    write_text_atomic(TARGET / "main.tex", rendered)

    copied: list[str] = ["main.tex"]
    for relative in SUPPORT_FILES + referenced_figures(rendered):
        source = TARGET / relative if relative == "cvpr.sty" else SOURCE / relative
        if not source.is_file():
            raise FileNotFoundError(f"canonical preprint dependency is missing: {source}")
        target = TARGET / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "tables/new_experiment_macros.tex":
            macros = source.read_text(encoding="utf-8").replace(
                "% Generated by scripts/export_twinworld_agent_figures.py.",
                "% Frozen metric macros for the named academic preprint.",
                1,
            )
            write_text_atomic(target, macros)
        elif source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(relative)

    sync_manifest = {
        "schema_version": "beyond_proximity.academic_sync.v1",
        "canonical_source_sha256": sha256_bytes(canonical.encode("utf-8")),
        "rendered_source_sha256": sha256_bytes(rendered.encode("utf-8")),
        "authors": list(AUTHORS),
        "dependencies": {
            relative: sha256_file(TARGET / relative)
            for relative in SUPPORT_FILES + referenced_figures(rendered)
        },
    }
    manifest_path = TARGET / "canonical_sync.json"
    write_text_atomic(manifest_path, json.dumps(sync_manifest, indent=2) + "\n")
    copied.append("canonical_sync.json")

    return {
        "source": SOURCE.relative_to(ROOT).as_posix() + "/main.tex",
        "target": TARGET.relative_to(ROOT).as_posix() + "/main.tex",
        "authors": list(AUTHORS),
        "files": copied,
        "figures": list(referenced_figures(rendered)),
    }


def main() -> int:
    print(json.dumps(sync(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
