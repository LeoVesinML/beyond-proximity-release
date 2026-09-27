from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUT_DIR / "semanticsplat_current_status_and_roadmap_2026-07-15.pdf"
FIGURES = ROOT / "papers" / "twinworld" / "figures"

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5E6A7D")
NAVY = colors.HexColor("#163A5F")
BLUE = colors.HexColor("#1769AA")
TEAL = colors.HexColor("#16817A")
GREEN = colors.HexColor("#2E7D4F")
AMBER = colors.HexColor("#B26A16")
RED = colors.HexColor("#A43A3A")
LIGHT_BLUE = colors.HexColor("#EAF3FA")
LIGHT_TEAL = colors.HexColor("#E9F5F2")
LIGHT_AMBER = colors.HexColor("#FFF4E4")
LIGHT_RED = colors.HexColor("#FCECEC")
GRID = colors.HexColor("#C9D3DF")
ROW = colors.HexColor("#F6F8FB")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_text(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.1,
            textColor=INK,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.4,
            textColor=MUTED,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            alignment=TA_CENTER,
            textColor=NAVY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.2,
            alignment=TA_CENTER,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.2,
            textColor=colors.white,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.5,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=INK,
            backColor=LIGHT_BLUE,
            borderColor=BLUE,
            borderWidth=0.8,
            borderPadding=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Warning",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.1,
            leading=12.5,
            textColor=colors.HexColor("#6E3108"),
            backColor=LIGHT_AMBER,
            borderColor=AMBER,
            borderWidth=0.8,
            borderPadding=8,
            spaceAfter=8,
        )
    )
    return styles


def p(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=17,
        bulletFontName="Helvetica",
        bulletFontSize=6,
        bulletColor=TEAL,
        spaceAfter=4,
    )


def report_table(
    data: list[list[str]],
    widths: list[float],
    styles,
    *,
    header_color=NAVY,
    font_size: float | None = None,
) -> Table:
    body = []
    for row_index, row in enumerate(data):
        style_name = "TableHeader" if row_index == 0 else "TableCell"
        cells = []
        for value in row:
            cell_style = styles[style_name]
            if font_size is not None:
                cell_style = ParagraphStyle(
                    f"{style_name}{font_size}",
                    parent=cell_style,
                    fontSize=font_size,
                    leading=font_size + 2,
                )
            cells.append(Paragraph(str(value), cell_style))
        body.append(cells)
    result = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
            ]
        )
    )
    return result


def metric_strip(metrics: list[tuple[str, str]], styles) -> Table:
    cells = []
    for value, label in metrics:
        cells.append(
            Table(
                [[p(value, styles, "MetricValue")], [p(label, styles, "MetricLabel")]],
                colWidths=[3.75 * cm],
                rowHeights=[0.62 * cm, 0.72 * cm],
            )
        )
    result = Table([cells], colWidths=[3.9 * cm] * len(cells), hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return result


def image(path: Path, width_cm: float, max_height_cm: float | None = None) -> Image:
    result = Image(str(path))
    ratio = width_cm * cm / result.imageWidth
    result.drawWidth = width_cm * cm
    result.drawHeight = result.imageHeight * ratio
    if max_height_cm is not None and result.drawHeight > max_height_cm * cm:
        ratio = max_height_cm * cm / result.imageHeight
        result.drawHeight = max_height_cm * cm
        result.drawWidth = result.imageWidth * ratio
    result.hAlign = "CENTER"
    return result


def figure_with_caption(path: Path, caption: str, styles, width_cm: float, max_height_cm=None):
    return KeepTogether(
        [
            image(path, width_cm, max_height_cm),
            Spacer(1, 0.08 * cm),
            p(caption, styles, "Small"),
        ]
    )


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 1.12 * cm, width - doc.rightMargin, height - 1.12 * cm)
    canvas.setFont("Helvetica", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 0.8 * cm, "SemanticSplat")
    canvas.drawRightString(width - doc.rightMargin, height - 0.8 * cm, "Current Status and Research Roadmap")
    canvas.line(doc.leftMargin, 1.0 * cm, width - doc.rightMargin, 1.0 * cm)
    canvas.drawString(doc.leftMargin, 0.62 * cm, "Evidence status: 15 July 2026")
    canvas.drawRightString(width - doc.rightMargin, 0.62 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    internal = read_json(
        ROOT / "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/metrics_summary.json"
    )["summary"]
    replica = read_json(
        ROOT / "outputs/public_datasets/replica_pilot_v1/graph/grounding_summary.json"
    )["metrics"]
    replica_flat = read_json(
        ROOT / "outputs/public_datasets/replica_pilot_v1/flat_lexical/grounding_summary.json"
    )["metrics"]
    scannet = read_json(
        ROOT / "outputs/public_datasets/scannet_pilot_v1/graph/grounding_summary.json"
    )["metrics"]
    scannet_flat = read_json(
        ROOT / "outputs/public_datasets/scannet_pilot_v1/flat_lexical/grounding_summary.json"
    )["metrics"]
    concept = read_json(
        ROOT / "outputs/baselines/conceptgraphs_full_v1/metrics_summary.json"
    )
    langsplat = read_json(
        ROOT / "outputs/baselines/langsplat_smoke_v1/metrics_summary.json"
    )

    avg = internal["averages"]
    quality = internal["quality"]
    styles = make_styles()
    branch = git_text("branch", "--show-current")
    revision = git_text("rev-parse", "--short", "HEAD")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.4 * cm,
        title="SemanticSplat Current Status and Research Roadmap",
        author="SemanticSplat Team",
        subject="Technical progress, limitations, and research roadmap",
    )
    story = []

    # Page 1: answer-first summary.
    story.append(Spacer(1, 0.85 * cm))
    story.append(p("Beyond Proximity: Graph-Pruned Semantic Search over 3D Scene Records", styles, "ReportTitle"))
    story.append(p("Current Status and Research Roadmap", styles, "ReportTitle"))
    story.append(
        p(
            f"Technical report | 15 July 2026 | Branch: {branch} | Evidence base: {revision}",
            styles,
            "ReportSubtitle",
        )
    )
    story.append(
        p(
            "<b>Bottom line.</b> The team has completed a reproducible semantic-search prototype, "
            "official Replica and ScanNet oracle-map pilots, a five-scene ConceptGraphs run, "
            "baseline smoke evidence, and an anonymous ECCV-format workshop manuscript. The "
            "remaining work is concentrated in submission actions and stronger non-oracle, "
            "same-protocol science rather than basic project plumbing.",
            styles,
            "Callout",
        )
    )
    story.append(
        metric_strip(
            [
                ("97", "captured views"),
                ("150", "internal queries"),
                ("575", "Replica GT boxes"),
                ("392", "ScanNet GT boxes"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        metric_strip(
            [
                ("144", "passing tests"),
                ("42/42", "paper number checks"),
                ("11", "paper pages incl. refs"),
                ("13", "cited related works"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        figure_with_caption(
            FIGURES / "fig_scene_gallery.png",
            "Five manually captured pilot scenes plus the default qualitative view used for demos.",
            styles,
            16.2,
            7.6,
        )
    )
    story.append(PageBreak())

    # Page 2: completed system and evidence.
    story.append(p("What we completed: a reproducible evaluation prototype", styles, "Section"))
    story.append(
        p(
            "The current system converts captured or imported scene observations into a semantic "
            "index and hierarchy, runs graph-pruned or flat query variants, and saves canonical "
            "per-query results for evaluation. Construction cost is separated from query-time cost.",
            styles,
        )
    )
    story.append(
        figure_with_caption(
            FIGURES / "fig_system_overview.png",
            "System flow. Graph and flat search consume the same semantic items; only the search structure differs.",
            styles,
            16.0,
            5.1,
        )
    )
    story.append(p("Completed work and saved evidence", styles, "Subsection"))
    story.append(
        report_table(
            [
                ["Area", "Completed result", "Evidence"],
                ["Captured data", "5 scenes, 97 RGB-D views, 1,066 semantic items", "backend/data/scenes/"],
                ["Internal GT", "150 queries; 125 carry verified expected view IDs", "docs/benchmarks/benchmark_queries_v2.json"],
                ["Public data", "8 Replica + 8 ScanNet scenes with official 3D object boxes", "backend/data/scenes/replica_* and scannet_*"],
                ["Evaluation", "Graph, graph+fallback, flat lexical, flat embedding, and ablations", "outputs/public_datasets/"],
                ["External baselines", "ConceptGraphs five-scene run; LangSplat official-sofa smoke", "outputs/baselines/"],
                ["Article", "Anonymous ECCV 2026 review PDF, 10 content pages + references", "papers/twinworld/main.pdf"],
                ["Verification", "144 tests and a 42-check paper evidence gate", "tests/ and twinworld_number_check.md"],
            ],
            [2.7 * cm, 7.0 * cm, 6.5 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "<b>Claim boundary:</b> this is a semantic retrieval and reasoning prototype. Public "
            "pilots search official semantic maps; they do not measure automatic semantic perception.",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # Page 3: internal result.
    story.append(p("Hierarchy cuts context sharply, with a measured quality cost", styles, "Section"))
    story.append(
        p(
            "Across 150 same-input queries, graph traversal checks 75.5% fewer views and uses "
            "68.2% fewer estimated context tokens. On the 125 queries with verified view GT, "
            "however, graph hit@1 and hit@3 remain below flat lexical search. This is a cost-quality "
            "frontier, not a claim of free quality preservation.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Metric", "Flat", "Graph", "Difference / interpretation"],
                ["Views checked (mean)", f"{avg['flat_views_checked']:.2f}", f"{avg['graph_views_checked']:.2f}", "75.5% fewer"],
                ["Est. context tokens (mean)", f"{avg['flat_input_tokens']:.1f}", f"{avg['graph_input_tokens']:.1f}", "68.2% fewer"],
                ["Context characters (mean)", f"{avg['flat_context_chars']:.1f}", f"{avg['graph_context_chars']:.1f}", "Less serialized evidence"],
                ["hit@1 (n=125)", f"{quality['hit_at_1_flat']:.3f}", f"{quality['hit_at_1_graph']:.3f}", "Graph -0.088"],
                ["hit@3 (n=125)", f"{quality['hit_at_3_flat']:.3f}", f"{quality['hit_at_3_graph']:.3f}", "Graph -0.120"],
            ],
            [4.4 * cm, 2.1 * cm, 2.1 * cm, 7.6 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        figure_with_caption(
            FIGURES / "fig_five_scene_summary.png",
            "Normalized cost and verified-view hit@k. The graph saves context but currently trails flat lexical retrieval quality.",
            styles,
            15.5,
            6.5,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        figure_with_caption(
            FIGURES / "fig_cumulative.png",
            "Cumulative estimated context over 150 queries: about 475k flat versus 152k graph.",
            styles,
            8.7,
            6.4,
        )
    )
    story.append(PageBreak())

    # Page 4: public pilots.
    story.append(p("Official public pilots validate object retrieval over GT maps", styles, "Section"))
    story.append(
        p(
            "Replica and ScanNet now provide official boxes and language-grounding queries. Graph "
            "and flat variants are compared over identical oracle semantic candidates. The key "
            "result is reduced candidate inspection at matched flat-lexical quality in these pilots.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Track", "Scenes / queries", "GT boxes", "Graph Acc@0.25", "Objects: graph / flat", "Interpretation"],
                [
                    "Replica",
                    "8 / 56",
                    "575",
                    f"{replica['acc_at_0_25']['value']:.3f}",
                    f"{replica['efficiency']['checked_objects']['mean']:.1f} / {replica_flat['efficiency']['checked_objects']['mean']:.1f}",
                    "Quality ceiling on oracle candidates",
                ],
                [
                    "ScanNet",
                    "8 / 48",
                    "392",
                    f"{scannet['acc_at_0_25']['value']:.3f}",
                    f"{scannet['efficiency']['checked_objects']['mean']:.1f} / {scannet_flat['efficiency']['checked_objects']['mean']:.1f}",
                    "Matched flat lexical; about 77% fewer objects",
                ],
            ],
            [2.0 * cm, 2.5 * cm, 1.7 * cm, 2.5 * cm, 3.1 * cm, 4.4 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        figure_with_caption(
            FIGURES / "fig_scannet_pilot.png",
            "ScanNet pilot. Graph matches flat lexical Acc@0.25 while checking substantially fewer objects; fallback approaches flat cost.",
            styles,
            16.0,
            7.4,
        )
    )
    story.append(
        figure_with_caption(
            FIGURES / "fig_replica_pilot.png",
            "Replica pilot. Acc@k=1.0 is an oracle-candidate sanity ceiling, not independent perception accuracy.",
            styles,
            9.2,
            5.4,
        )
    )
    story.append(
        p(
            "<b>Metric definition:</b> Recall@1 requires the exact target object ID. Acc@0.25 "
            "requires predicted and GT 3D boxes to reach IoU >= 0.25. Missing predictions remain "
            "in the eligible denominator with IoU zero.",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    # Page 5: baselines and article readiness.
    story.append(p("Baselines ran, but fair public-dataset comparison is still missing", styles, "Section"))
    concept_metrics = concept.get("metrics", concept.get("summary", concept))
    ls_metrics = langsplat.get("metrics", langsplat.get("summary", langsplat))
    story.append(
        p(
            "ConceptGraphs now has reproducible native evidence on all five captured scenes. "
            "LangSplat executed on its official pretrained sofa asset. These runs prove integration "
            "and execution, but not superiority or a fair same-scene public benchmark.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Baseline", "Completed evidence", "Current boundary", "Next fair test"],
                ["ConceptGraphs", "5 captured scenes; 150 canonical outputs", "Drone map empty; internal GT; no Replica/ScanNet run", "Run selected Replica/ScanNet scenes and identical queries"],
                ["LangSplat", "Official sofa render/query smoke; 1 canonical output", "No GT and no captured/public scene comparison", "Train/convert one shared public scene with GT"],
                ["BBQ", "Paper studied and metrics/protocol aligned", "Official code not reproduced", "Same query subset and oracle/perception split"],
                ["SemanticSplat variants", "Graph, fallback, flat lexical, flat embedding", "Same-input internal comparisons only", "Retain as controlled ablations"],
            ],
            [2.6 * cm, 4.5 * cm, 4.8 * cm, 4.3 * cm],
            styles,
        )
    )
    story.append(p("Where the current hierarchy saves the most", styles, "Subsection"))
    chart_pair = Table(
        [[
            [
                image(FIGURES / "fig_by_query_type.png", 7.7, 6.5),
                p("Savings by query family.", styles, "Small"),
            ],
            [
                image(FIGURES / "fig_by_scene.png", 7.7, 6.5),
                p("Savings by captured scene.", styles, "Small"),
            ],
        ]],
        colWidths=[8.0 * cm, 8.0 * cm],
        hAlign="CENTER",
    )
    chart_pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(chart_pair)
    story.append(p("Article readiness", styles, "Subsection"))
    story.append(
        report_table(
            [
                ["Check", "Status", "Meaning"],
                ["ECCV 2026 review format", "DONE", "Anonymous, line numbered, 10 content pages + references"],
                ["Claim/evidence consistency", "42/42", "Includes 97-view count and forbidden stale wording"],
                ["Related work", "13 citations", "3DGS semantics, scene graphs, grounding, open-vocabulary maps"],
                ["Submission portal", "PENDING", "Paper ID, author profiles/conflicts, upload"],
            ],
            [4.0 * cm, 2.5 * cm, 9.7 * cm],
            styles,
        )
    )
    story.append(PageBreak())

    # Page 6: missing work and risks.
    story.append(p("What is still missing: science and submission actions", styles, "Section"))
    story.append(
        p(
            "The following items are not hidden blockers; they define exactly what the current "
            "paper does not yet prove and what must be completed for submission or a stronger venue.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Gap", "Severity now", "Why it matters", "Completion criterion"],
                ["Public repository publicity", "P0 / policy", "ECCV embargo risk because public materials name the intended venue", "Make repo private or complete owner-approved remediation"],
                ["OpenReview submission actions", "P0 / blocking", "Paper ID, profiles, conflicts, and upload cannot be automated locally", "All author profiles complete; ID inserted; PDF uploaded"],
                ["Oracle semantic candidates", "High science", "Public pilots do not test automatic perception", "Add predicted objects/segments and report non-oracle metrics"],
                ["Graph quality gap", "High science", "Internal hit@k trails flat lexical", "Calibrated pruning/fallback recovers quality with most savings"],
                ["Fair external baseline", "High science", "No same-scene public-dataset BBQ/ConceptGraphs/LangSplat comparison", "Identical scenes, queries, GT, and metrics"],
                ["End-to-end 3DGS evidence", "Medium-high", "Title frames the system around 3DGS, but evaluation is mainly over semantic views/maps", "One reproducible 3DGS reconstruction-to-query experiment"],
                ["Segmentation metrics", "Medium-high", "mAcc/mIoU/fmIoU remain N/A without model predictions", "Paired prediction and GT arrays with evaluator output"],
                ["Statistical breadth", "Medium", "48 ScanNet queries are a pilot, not a broad benchmark", "Expanded queries, per-family results, confidence intervals"],
            ],
            [3.2 * cm, 2.3 * cm, 5.4 * cm, 5.3 * cm],
            styles,
            font_size=7.0,
        )
    )
    story.append(Spacer(1, 0.28 * cm))
    story.append(
        p(
            "<b>Immediate policy action:</b> do not push additional venue-labelled changes to the "
            "public repository until the team resolves visibility/history. The current report was "
            "generated locally and does not change remote visibility.",
            styles,
            "Warning",
        )
    )
    story.append(p("What the current evidence safely supports", styles, "Subsection"))
    story.append(
        bullets(
            [
                "Hierarchy substantially reduces checked context over the same semantic map.",
                "Internal lexical retrieval currently pays a measurable hit@k cost.",
                "Replica and ScanNet oracle-map pilots show candidate-efficiency gains, not semantic-perception accuracy.",
                "ConceptGraphs and LangSplat execution evidence is real but protocol-limited.",
                "No superiority claim over BBQ, ConceptGraphs, or LangSplat is currently justified.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    # Page 7: future plan.
    story.append(p("Roadmap: submit carefully, then strengthen the science", styles, "Section"))
    story.append(
        p(
            "The fastest credible path is to freeze the workshop submission while starting a "
            "parallel research track that removes oracle assumptions and tests the 3DGS claim end to end.",
            styles,
        )
    )
    roadmap = report_table(
        [
            ["Phase", "Window", "Work", "Exit criteria"],
            ["1. Submission freeze", "15-31 Jul 2026", "Resolve publicity; insert paper ID; profiles/conflicts; final human anonymity/citation check; upload", "Valid anonymous PDF accepted by portal before deadline"],
            ["2. Quality recovery", "Aug-Sep 2026", "Tune branch threshold and fallback; relation diagnostics; expand ScanNet queries; confidence intervals", "Graph approaches flat hit@k while retaining most context savings"],
            ["3. Non-oracle evaluation", "Sep-Nov 2026", "Predicted segmentation/object proposals; mIoU/mAcc/fmIoU; public-scene ConceptGraphs or BBQ", "Same-scene, same-query, same-GT baseline table"],
            ["4. 3DGS integration", "Oct-Dec 2026", "Reconstruct/import a 3DGS scene; automatic semantic extraction; query and localization demo", "Reproducible reconstruction-to-answer pipeline"],
            ["5. Main-conference package", "Late 2026-2027", "Broader datasets, ablations, robustness, failure taxonomy, supplementary artifact", "Publication claim supported beyond a workshop-scale pilot"],
        ],
        [3.0 * cm, 2.7 * cm, 6.1 * cm, 4.4 * cm],
        styles,
        header_color=TEAL,
    )
    roadmap.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 1), (0, 1), LIGHT_BLUE),
                ("BACKGROUND", (0, 2), (0, 2), LIGHT_TEAL),
                ("BACKGROUND", (0, 3), (0, 3), LIGHT_TEAL),
                ("BACKGROUND", (0, 4), (0, 4), LIGHT_AMBER),
                ("BACKGROUND", (0, 5), (0, 5), LIGHT_AMBER),
                ("TEXTCOLOR", (0, 1), (0, -1), INK),
            ]
        )
    )
    story.append(roadmap)
    story.append(Spacer(1, 0.3 * cm))
    story.append(p("Recommended priority order", styles, "Subsection"))
    story.append(
        bullets(
            [
                "First protect submission eligibility and finish the external OpenReview actions.",
                "Then recover graph retrieval quality; this is the core algorithmic weakness visible in current evidence.",
                "Next replace oracle maps with predictions and run one fair public baseline.",
                "Finally demonstrate the full 3DGS-to-semantic-query path and scale the benchmark.",
            ],
            styles,
        )
    )
    story.append(
        p(
            "<b>Final recommendation.</b> The current artifact is suitable as an honest TwinWorld "
            "workshop submission after the external submission and publicity actions. A stronger "
            "main-conference paper should wait for quality recovery, non-oracle predictions, a fair "
            "baseline, and explicit end-to-end 3DGS evidence.",
            styles,
            "Callout",
        )
    )
    story.append(p("Evidence base", styles, "Subsection"))
    story.append(
        p(
            "Primary sources: five_scene_graph_vs_flat_v2 metrics; Replica and ScanNet pilot "
            "grounding summaries; ConceptGraphs full-run and LangSplat smoke outputs; TwinWorld "
            "claim/number/submission audits; papers/twinworld/main.pdf. All provider-token, "
            "segmentation, and superiority claims outside these artifacts remain unavailable.",
            styles,
            "Small",
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return PDF_PATH


if __name__ == "__main__":
    print(build_report())
