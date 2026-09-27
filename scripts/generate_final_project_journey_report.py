from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
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
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUT_DIR / "semanticsplat_final_project_journey_report_2026-07-22.pdf"
FIGURES = ROOT / "papers" / "beyond-proximity" / "figures"

INK = colors.HexColor("#17202B")
MUTED = colors.HexColor("#617083")
NAVY = colors.HexColor("#173A5E")
BLUE = colors.HexColor("#1B6CA8")
TEAL = colors.HexColor("#177E78")
GREEN = colors.HexColor("#2F7D4A")
AMBER = colors.HexColor("#B46C16")
RED = colors.HexColor("#A13B3B")
LIGHT_BLUE = colors.HexColor("#EAF3FA")
LIGHT_TEAL = colors.HexColor("#E8F4F1")
LIGHT_GREEN = colors.HexColor("#EAF5ED")
LIGHT_AMBER = colors.HexColor("#FFF3E2")
LIGHT_RED = colors.HexColor("#FBECEC")
GRID = colors.HexColor("#C8D2DE")
ROW = colors.HexColor("#F5F7FA")
WHITE = colors.white


EVIDENCE = {
    "E1": "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/metrics_summary.json",
    "E2": "docs/reports/final/graph_construction_cost.json",
    "E3": "outputs/public_datasets/replica_pilot_v1/",
    "E4": "outputs/public_datasets/scannet_pilot_v1/",
    "E5": "outputs/baselines/conceptgraphs_scannet_full_v1/metrics_summary.json",
    "E6": "outputs/baselines/conceptgraphs_replica_full_v1/metrics_summary.json",
    "E7": "outputs/baselines/langsplat_scannet_full_v1/metrics_summary.json",
    "E8": "TWINWORLD_2026_NEXT_WORK_PLAN.md",
    "E9": "WEEK6_INTEGRATION_REPORT.md",
    "E10": "papers/beyond-proximity/main.pdf",
    "E11": "site/ and scripts/build_project_site.py",
    "E12": "tests/ and scripts/check_academic_paper.py",
    "E13": "docs/baselines/bbq_comparison.md",
    "E14": "PUBLICATION_VENUE_FIT_ASSESSMENT.md",
}


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


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
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=13,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=2,
            spaceAfter=8,
            keepWithNext=True,
            outlineLevel=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
            outlineLevel=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.15,
            leading=12.8,
            textColor=INK,
            spaceAfter=5.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.6,
            textColor=MUTED,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.6,
            leading=9.8,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=4,
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
            fontSize=7.1,
            leading=8.8,
            alignment=TA_CENTER,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.3,
            leading=9.0,
            textColor=WHITE,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.2,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCellSmall",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=6.6,
            leading=8.2,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=12.8,
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
            name="Success",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.1,
            leading=12.6,
            textColor=colors.HexColor("#214F30"),
            backColor=LIGHT_GREEN,
            borderColor=GREEN,
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
            fontSize=9.0,
            leading=12.4,
            textColor=colors.HexColor("#6F350C"),
            backColor=LIGHT_AMBER,
            borderColor=AMBER,
            borderWidth=0.8,
            borderPadding=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=7.3,
            leading=9.6,
            textColor=colors.HexColor("#253244"),
            backColor=colors.HexColor("#F0F3F6"),
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=6,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=INK,
            spaceAfter=12,
        )
    )
    return styles


class ReportDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"H1", "H2"}:
            return
        level = 0 if style_name == "H1" else 1
        text = flowable.getPlainText()
        key = f"heading-{level}-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def p(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str], styles, color=TEAL) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=17,
        bulletFontName="Helvetica",
        bulletFontSize=6,
        bulletColor=color,
        spaceAfter=4,
    )


def report_table(
    data: list[list[str]],
    widths: list[float],
    styles,
    *,
    header_color=NAVY,
    small: bool = False,
    alignments: dict[int, str] | None = None,
) -> Table:
    body = []
    for row_index, row in enumerate(data):
        style_name = "TableHeader" if row_index == 0 else (
            "TableCellSmall" if small else "TableCell"
        )
        body.append([Paragraph(str(value), styles[style_name]) for value in row])
    result = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, header_color),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, header_color),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ROW]),
    ]
    if alignments:
        for column, alignment in alignments.items():
            commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def metric_strip(metrics: list[tuple[str, str]], styles) -> Table:
    page_width = A4[0] - 2.8 * cm
    cell_width = page_width / len(metrics)
    cells = []
    for value, label in metrics:
        cells.append(
            Table(
                [[p(value, styles, "MetricValue")], [p(label, styles, "MetricLabel")]],
                colWidths=[cell_width - 0.12 * cm],
                rowHeights=[0.62 * cm, 0.67 * cm],
            )
        )
    result = Table([cells], colWidths=[cell_width] * len(cells), hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
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
    scale = width_cm * cm / result.imageWidth
    result.drawWidth = width_cm * cm
    result.drawHeight = result.imageHeight * scale
    if max_height_cm is not None and result.drawHeight > max_height_cm * cm:
        scale = max_height_cm * cm / result.imageHeight
        result.drawHeight = max_height_cm * cm
        result.drawWidth = result.imageWidth * scale
    result.hAlign = "CENTER"
    return result


def figure(path: Path, caption: str, styles, width_cm: float, max_height_cm=None):
    return KeepTogether(
        [
            image(path, width_cm, max_height_cm),
            Spacer(1, 0.08 * cm),
            p(caption, styles, "Caption"),
        ]
    )


def two_figures(
    left_path: Path,
    left_caption: str,
    right_path: Path,
    right_caption: str,
    styles,
    width_cm: float = 7.8,
    max_height_cm: float = 6.0,
) -> Table:
    left = [
        image(left_path, width_cm, max_height_cm),
        p(left_caption, styles, "Caption"),
    ]
    right = [
        image(right_path, width_cm, max_height_cm),
        p(right_caption, styles, "Caption"),
    ]
    table = Table([[left, right]], colWidths=[8.2 * cm, 8.2 * cm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 1.08 * cm, width - doc.rightMargin, height - 1.08 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 0.76 * cm, "SemanticSplat")
    canvas.drawRightString(
        width - doc.rightMargin,
        height - 0.76 * cm,
        "Final Project Journey and Evidence Report",
    )
    canvas.line(doc.leftMargin, 0.96 * cm, width - doc.rightMargin, 0.96 * cm)
    canvas.drawString(doc.leftMargin, 0.58 * cm, "Evidence frozen: 22 July 2026")
    canvas.drawRightString(width - doc.rightMargin, 0.58 * cm, f"Page {doc.page}")
    canvas.restoreState()


def add_page(story, title: str, styles):
    story.append(p(title, styles, "H1"))


def build_report() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = make_styles()

    internal = read_json(EVIDENCE["E1"])["summary"]
    construction = read_json(EVIDENCE["E2"])
    replica_graph = read_json(
        "outputs/public_datasets/replica_pilot_v1/graph/grounding_summary.json"
    )["metrics"]
    replica_flat = read_json(
        "outputs/public_datasets/replica_pilot_v1/flat_lexical/grounding_summary.json"
    )["metrics"]
    replica_embedding = read_json(
        "outputs/public_datasets/replica_pilot_v1/flat_embedding/grounding_summary.json"
    )["metrics"]
    scannet_graph = read_json(
        "outputs/public_datasets/scannet_pilot_v1/graph/grounding_summary.json"
    )["metrics"]
    scannet_flat = read_json(
        "outputs/public_datasets/scannet_pilot_v1/flat_lexical/grounding_summary.json"
    )["metrics"]
    scannet_embedding = read_json(
        "outputs/public_datasets/scannet_pilot_v1/flat_embedding/grounding_summary.json"
    )["metrics"]
    scannet_fallback = read_json(
        "outputs/public_datasets/scannet_pilot_v1/graph_fallback/grounding_summary.json"
    )["metrics"]
    scannet_no_relation = read_json(
        "outputs/public_datasets/scannet_pilot_v1/ablation_no_relation/grounding_summary.json"
    )["metrics"]
    cg_scannet = read_json(EVIDENCE["E5"])["metrics"]
    cg_replica = read_json(EVIDENCE["E6"])["metrics"]
    langsplat = read_json(EVIDENCE["E7"])["metrics"]

    avg = internal["averages"]
    quality = internal["quality"]
    totals = construction["totals"]
    branch = git_text("branch", "--show-current")
    revision = git_text("rev-parse", "--short", "HEAD")

    doc = ReportDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.48 * cm,
        bottomMargin=1.34 * cm,
        title="SemanticSplat Final Project Journey and Evidence Report",
        author="SemanticSplat Team",
        subject="Project history, implementation, evaluation, publication artifacts, and next steps",
    )

    story = []

    # Cover.
    story.append(Spacer(1, 0.65 * cm))
    story.append(p("SemanticSplat", styles, "CoverTitle"))
    story.append(p("Graph-Pruned Semantic Search", styles, "CoverTitle"))
    story.append(
        p(
            "Final Project Journey and Evidence Report",
            styles,
            "CoverSubtitle",
        )
    )
    story.append(
        p(
            f"22 July 2026 | Branch: {branch} | Evidence revision: {revision}",
            styles,
            "CoverSubtitle",
        )
    )
    story.append(
        p(
            "<b>Final outcome.</b> The team moved from a small semantic 3DGS query prototype and "
            "legacy simulated demo to a reproducible research package with five captured scenes, "
            "official Replica and ScanNet object-grounding pilots, canonical evaluation schemas, "
            "full native ConceptGraphs runs, a complete hardware-adapted LangSplat execution, an "
            "academic preprint, a public project site, and automated evidence checks. The central "
            "measured result is substantial context reduction with a visible retrieval-quality "
            "trade-off under the current lexical hierarchy.",
            styles,
            "Callout",
        )
    )
    story.append(
        metric_strip(
            [
                ("5", "team-captured scenes"),
                ("97", "captured RGB-D views"),
                ("1,066", "semantic items"),
                ("150", "internal queries"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.14 * cm))
    story.append(
        metric_strip(
            [
                ("16", "public dataset scenes"),
                ("967", "official GT boxes"),
                ("145", "passing tests"),
                ("9", "academic paper pages"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        figure(
            FIGURES / "fig_scene_gallery.png",
            "The captured-scene evidence used for the internal digital-twin track. [E1, E2]",
            styles,
            16.0,
            7.4,
        )
    )
    story.append(PageBreak())

    # Table of contents.
    story.append(p("Contents", styles, "TOCHeading"))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCLevel1",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=15,
            leftIndent=0,
            firstLineIndent=0,
            textColor=INK,
        ),
        ParagraphStyle(
            name="TOCLevel2",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            leftIndent=14,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    story.append(toc)
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        p(
            "<b>How to read this report.</b> Sections 1-3 explain where the project began and how "
            "the team organized the work. Sections 4-11 document the system and quantitative "
            "evidence. Sections 12-15 assess plan completion, limitations, release readiness, and "
            "the next research decisions. Evidence identifiers map to exact repository paths in "
            "Appendix A.",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    # 1. Starting point.
    add_page(story, "1. Where the project started", styles)
    story.append(
        p(
            "SemanticSplat began as a practical question: can natural-language queries be grounded "
            "in a 3D Gaussian Splatting scene without sending every captured view to a model? The "
            "early repository already had a viewer, a query UI, a small semantic tree, and a legacy "
            "graph-versus-flat demo. It did not yet have a publication-grade dataset contract, "
            "independent evaluation denominators, public-dataset evidence, full external baselines, "
            "or a manuscript whose numbers were tied automatically to frozen JSON.",
            styles,
        )
    )
    story.append(p("The initial gap", styles, "H2"))
    story.append(
        report_table(
            [
                ["Dimension", "Starting state", "Final state"],
                ["Scenes", "Default demo and early captures", "5 captured scenes plus 8 Replica and 8 ScanNet scenes"],
                ["Queries", "Small demo set", "150 internal, 56 Replica, and 48 ScanNet queries"],
                ["Ground truth", "Manual view labels; incomplete boxes", "Verified-view labels plus 575 Replica and 392 ScanNet official boxes"],
                ["Metrics", "Legacy latency/token estimates", "hit@k, Recall@1, Acc@0.1/0.25/0.5, 3D IoU, traversal, context, tokens, runtime"],
                ["Baselines", "Flat demo and smoke attempts", "Flat lexical, flat embedding, fallback, ablations, ConceptGraphs, LangSplat"],
                ["Publication", "Draft notes", "TwinWorld package, named-author academic preprint, project site, citation metadata"],
                ["Reproducibility", "Manual reports", "Canonical JSON, schemas, commands, 145 tests, number and paper checks"],
            ],
            [3.1 * cm, 6.2 * cm, 7.1 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "The project therefore changed in kind, not only in scale: from a product demonstration "
            "to a controlled research artifact. The decisive shift was to separate three questions: "
            "whether hierarchy reduces query cost, whether object grounding is correct under official "
            "GT maps, and whether external predicted-map systems can execute in the same environment.",
            styles,
            "Success",
        )
    )
    story.append(PageBreak())

    # 2. Timeline.
    add_page(story, "2. How we reached the final system", styles)
    story.append(
        p(
            "The work progressed through successive evidence gates. Each phase kept the useful "
            "artifacts from the previous phase while replacing claims that could not survive a "
            "controlled evaluation.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Date", "Milestone", "What changed"],
                ["8 Apr", "Repository foundation", "Initial project code and semantic 3DGS direction were established."],
                ["9-14 Jun", "Runnable product loop", "Headless pipeline, launchers, Query Flow, benchmark tooling, and demo video were added."],
                ["23-28 Jun", "Evaluation infrastructure", "Captured scenes, canonical outputs, baseline smoke evidence, and Week 2/3 reports were integrated."],
                ["1-5 Jul", "Five-scene GT and controlled benchmark", "Manual GT package, 150-query graph-versus-flat benchmark, ablations, and scaling evidence were frozen."],
                ["7 Jul", "ConceptGraphs expansion", "Full captured-scene execution, tokenizer accounting, and GT-box evaluation were added."],
                ["11-12 Jul", "Official public datasets", "All eight Replica and eight ScanNet scenes, boxes, and ReferIt3D mappings became ready."],
                ["13-15 Jul", "Four-person Week 6 integration", "BBQ-aligned grounding, baseline audit, TwinWorld paper, and linear branch integration were completed."],
                ["15 Jul", "Native baseline completion", "ConceptGraphs ran on both public tracks; LangSplat completed one full ScanNet pipeline."],
                ["21-22 Jul", "Public release", "Academic preprint, project site, GitHub Pages workflow, citation metadata, and final layout QA were completed."],
            ],
            [2.2 * cm, 4.3 * cm, 9.9 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.24 * cm))
    story.append(
        figure(
            FIGURES / "fig_method_workflow.png",
            "The final pipeline joins capture evidence, deterministic hierarchy construction, controlled graph/flat retrieval, and grounded output. [E10]",
            styles,
            16.2,
            8.8,
        )
    )
    story.append(PageBreak())

    # 3. Four-person plan.
    add_page(story, "3. Four-person plan completion", styles)
    story.append(
        p(
            "The two-week plan divided ownership by dependency: datasets and GT first, algorithm and "
            "evaluation second, baselines and related work third, and paper/submission integration "
            "fourth. The branch history is linear: later person branches include earlier work, and "
            "the integration branch records all four tips. [E8, E9]",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Role", "Acceptance result", "Final evidence", "Assessment"],
                ["Person 1: data and GT", "8 Replica + 8 ScanNet scenes ready; 392/392 ScanNet instances mapped; Nr3D/Sr3D+ mappings validated", "Dataset plans, manifests, import/validation reports", "100% of required acceptance"],
                ["Person 2: algorithm and evaluation", "Object grounding, canonical outputs, eight variants per public track, BBQ metrics, cost fields, and failure labels", "Evaluation script and public-dataset outputs", "100% of required acceptance"],
                ["Person 3: BBQ and baselines", "Living BBQ comparison, valid/invalid comparison rules, baseline matrix, and evidence-backed related work", "BBQ comparison and baseline status docs", "100% of required acceptance"],
                ["Person 4: paper and release", "ECCV-format anonymous paper, figures, claim audit, checks; later named-author preprint and Pages site", "Paper sources, PDFs, checklists, site", "100% repository-side"],
            ],
            [3.1 * cm, 6.2 * cm, 4.2 * cm, 2.9 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        metric_strip(
            [
                ("4/4", "role acceptance records met"),
                ("16/16", "planned public scenes ready"),
                ("2/2", "public grounding tracks complete"),
                ("3", "native external executions"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "<b>Completion does not mean every scientific limitation disappeared.</b> The plan's "
            "acceptance criteria are complete. Remaining items such as independent predicted-map "
            "segmentation, a fair shared-protocol external leaderboard, and improved ScanNet relation "
            "reasoning are next research objectives, not missing branch deliverables.",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # 4. System architecture.
    add_page(story, "4. What the final system does", styles)
    story.append(
        p(
            "SemanticSplat is a semantic query layer over captured scene evidence. It does not replace "
            "3D reconstruction, localization, or Gaussian Splatting. It normalizes scene observations, "
            "builds a scene-zone-region-entity-view hierarchy, parses a natural-language query, prunes "
            "unlikely branches, ranks retained evidence, and emits selected object/view IDs, geometry, "
            "traversal traces, and measured cost fields.",
            styles,
        )
    )
    story.append(
        figure(
            FIGURES / "fig_method_workflow.png",
            "End-to-end method assembled from checked-in capture, ViewJSON, hierarchy, query, and grounded result artifacts. [E10]",
            styles,
            16.3,
            10.0,
        )
    )
    story.append(p("Core design choices", styles, "H2"))
    story.append(
        bullets(
            [
                "<b>Same-map comparison:</b> graph and flat methods use identical semantic records, query strings, labels, and scorers; candidate selection is the controlled difference.",
                "<b>Hierarchical query representation:</b> target, attributes, intent, anchor, and relation determine which scene branches remain active.",
                "<b>Geometry-aware output:</b> object IDs, view IDs, 2D/3D boxes when available, and target-anchor relation evidence are preserved in canonical results.",
                "<b>Fallback and ablations:</b> broad expansion, no-relation variants, and embedding baselines expose where pruning or spatial reranking helps or hurts.",
                "<b>Separate accounting:</b> one-time annotation/tree construction is not mixed with per-query retrieval cost.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    # 5. Data and GT.
    add_page(story, "5. Data, ground truth, and evidence boundaries", styles)
    story.append(
        p(
            "Three evaluation tracks answer different questions. The internal track measures hierarchy "
            "over team-owned captures and verified view labels. Replica and ScanNet measure object "
            "retrieval over official semantic-map candidates and boxes. Native external baselines build "
            "their own predicted representations and therefore cannot be merged into the oracle-map "
            "leaderboard.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Track", "Scenes", "Evidence", "Queries", "Valid interpretation"],
                ["Internal captures", "5", "97 ViewJSON records, 1,066 items, 125 verified-view GT queries", "150", "Same-map graph-versus-flat cost and view retrieval"],
                ["Replica", "8", "575 official 3D boxes", "56", "Oracle-map object retrieval and 3D-box overlap"],
                ["ScanNet", "8", "392 official 3D boxes; Nr3D/Sr3D+ mappings", "48", "Oracle-map object retrieval and relational grounding"],
                ["ConceptGraphs", "8 + 8", "Predicted object maps", "48 + 56", "Native integration diagnostics under a different map protocol"],
                ["LangSplat", "1", "Learned language field on ScanNet 0011_00", "6", "Full native pipeline and view-level diagnostic"],
            ],
            [3.0 * cm, 1.5 * cm, 5.7 * cm, 1.7 * cm, 4.5 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        figure(
            FIGURES / "fig_scene_gallery.png",
            "Internal track: five captured scenes and the default qualitative view. Licensed public-dataset imagery is not redistributed. [E1, E11]",
            styles,
            16.0,
            8.1,
        )
    )
    story.append(
        p(
            "<b>Ground-truth rule.</b> Manual ViewJSON is a reference artifact, not independent public "
            "GT. Official boxes are valid for object retrieval and 3D IoU only when a predicted box "
            "exists. Segmentation mAcc/mIoU/fmIoU remain unavailable because paired predicted and GT "
            "per-point labels were not produced. [E3, E4]",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # 6. Evaluation protocol.
    add_page(story, "6. Evaluation contract and metric definitions", styles)
    story.append(
        p(
            "The final evaluation avoids a common failure in research prototypes: reporting a number "
            "without stating what was eligible to enter its denominator. Every quality metric requires "
            "both a prediction artifact and a compatible reference artifact. Efficiency metrics use "
            "all completed queries that contain the relevant trace field.",
            styles,
        )
    )
    story.append(
        figure(
            FIGURES / "fig_evaluation_protocol.png",
            "Controlled internal comparison and separate public/native tracks. [E1, E3-E7]",
            styles,
            15.8,
            6.5,
        )
    )
    story.append(
        report_table(
            [
                ["Metric", "Prediction requirement", "Reference requirement", "Denominator"],
                ["Views/objects/nodes checked", "Traversal trace", "None", "Completed queries"],
                ["Runtime/context/tokens", "Timed result and count metadata", "None", "Results with the measurement"],
                ["hit@k", "Ranked view IDs", "Verified expected view IDs", "Eligible positive-view queries"],
                ["Recall@1", "Selected object ID", "Official expected object ID", "Object-ID-eligible queries"],
                ["3D IoU", "Valid predicted 3D box", "Valid official 3D box", "Box-eligible predictions"],
                ["Acc@threshold", "3D IoU for a prediction", "IoU threshold", "All box-eligible queries; misses score zero"],
                ["Segmentation mIoU", "Predicted per-point labels", "Paired GT segmentation", "Paired points/classes only"],
            ],
            [3.2 * cm, 4.5 * cm, 4.5 * cm, 4.2 * cm],
            styles,
            small=True,
        )
    )
    story.append(PageBreak())

    # 7. Internal result.
    add_page(story, "7. Main result: hierarchy sharply reduces context", styles)
    story.append(
        p(
            "Across 150 same-input queries, graph traversal checks 4.77 views on average versus 19.4 "
            "for flat search and serializes about 1,014 estimated tokens versus 3,168. This is a 75.5% "
            "reduction in checked views and a 68.2% reduction in estimated context tokens. The result "
            "is deterministic local retrieval cost, not provider billing usage. [E1]",
            styles,
        )
    )
    story.append(
        metric_strip(
            [
                ("75.5%", "fewer checked views"),
                ("68.2%", "fewer estimated tokens"),
                ("6.05 ms", "graph local runtime/query"),
                ("6.61 ms", "flat local runtime/query"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        report_table(
            [
                ["Metric", "Flat", "Graph", "Interpretation"],
                ["Views/query", f"{avg['flat_views_checked']:.2f}", f"{avg['graph_views_checked']:.2f}", "Graph checks 75.5% fewer"],
                ["Estimated context tokens/query", f"{avg['flat_input_tokens']:.1f}", f"{avg['graph_input_tokens']:.1f}", "Graph uses 68.2% fewer"],
                ["Context characters/query", f"{avg['flat_context_chars']:.1f}", f"{avg['graph_context_chars']:.1f}", "Same serialized-evidence trend"],
                ["hit@1 (n=125)", f"{quality['hit_at_1_flat']:.3f}", f"{quality['hit_at_1_graph']:.3f}", "Graph difference: -0.088"],
                ["hit@3 (n=125)", f"{quality['hit_at_3_flat']:.3f}", f"{quality['hit_at_3_graph']:.3f}", "Graph difference: -0.120"],
            ],
            [4.2 * cm, 2.3 * cm, 2.3 * cm, 7.6 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        figure(
            FIGURES / "fig_five_scene_summary.png",
            "Normalized internal efficiency and verified-view quality. Lower cost is achieved with a measured hit@k penalty. [E1]",
            styles,
            15.7,
            7.0,
        )
    )
    story.append(
        p(
            "<b>Conclusion from the main experiment:</b> hierarchy is already effective as a context "
            "budget mechanism, but the current hard lexical gates do not preserve flat-search quality. "
            "The correct claim is a measured quality-cost frontier, not universal superiority.",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # 8. Breakdown and uncertainty.
    add_page(story, "8. Where the savings come from", styles)
    story.append(
        p(
            "Savings remain positive across every query type and captured scene. Simple-object and "
            "functional queries localize to compact branches and save the most context. Relational and "
            "multi-hop queries retain more branches, so their reduction is smaller. Conference Hall "
            "has the largest scene-level token reduction; the larger outdoor-street hierarchy has the "
            "smallest. [E1, E10]",
            styles,
        )
    )
    story.append(
        two_figures(
            FIGURES / "fig_by_query_type.png",
            "Estimated token savings by query type.",
            FIGURES / "fig_by_scene.png",
            "Per-scene view and token savings.",
            styles,
            7.8,
            6.8,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        figure(
            FIGURES / "fig_cumulative.png",
            "Cumulative estimated context: approximately 475k flat tokens versus 152k graph tokens over 150 queries. [E1]",
            styles,
            11.4,
            8.2,
        )
    )
    story.append(
        p(
            "A paired 10,000-resample bootstrap reports 95% percentile intervals of 73.2%-77.7% "
            "for view reduction and 65.3%-71.0% for token reduction. Quality intervals cross or "
            "remain below zero for graph-minus-flat hit@k, reinforcing that the cost improvement is "
            "stable while quality recovery is the next algorithmic problem. [E10]",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    # 9. Public pilots.
    add_page(story, "9. Replica and ScanNet public-dataset pilots", styles)
    story.append(
        p(
            "The public pilots align scene choices and grounding metrics with BBQ where possible. "
            "They use official object identities and boxes as semantic-map candidates. Consequently, "
            "they evaluate retrieval and candidate reduction, not automatic semantic-map prediction. "
            "[E3, E4, E13]",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Track / variant", "Recall@1", "Acc@0.25", "Objects checked", "Est. tokens", "Runtime/query"],
                ["Replica graph", "1.000", "1.000", f"{replica_graph['efficiency']['checked_objects']['mean']:.1f}", f"{replica_graph['efficiency']['estimated_tokens']['mean']:.0f}", f"{replica_graph['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["Replica flat lexical", "1.000", "1.000", f"{replica_flat['efficiency']['checked_objects']['mean']:.1f}", f"{replica_flat['efficiency']['estimated_tokens']['mean']:.0f}", f"{replica_flat['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["Replica flat embedding", "0.833", "0.875", f"{replica_embedding['efficiency']['checked_objects']['mean']:.1f}", f"{replica_embedding['efficiency']['estimated_tokens']['mean']:.0f}", f"{replica_embedding['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["ScanNet graph", "0.271", "0.271", f"{scannet_graph['efficiency']['checked_objects']['mean']:.1f}", f"{scannet_graph['efficiency']['estimated_tokens']['mean']:.0f}", f"{scannet_graph['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["ScanNet flat lexical", "0.271", "0.271", f"{scannet_flat['efficiency']['checked_objects']['mean']:.1f}", f"{scannet_flat['efficiency']['estimated_tokens']['mean']:.0f}", f"{scannet_flat['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["ScanNet graph + fallback", "0.271", "0.271", f"{scannet_fallback['efficiency']['checked_objects']['mean']:.1f}", f"{scannet_fallback['efficiency']['estimated_tokens']['mean']:.0f}", f"{scannet_fallback['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
                ["ScanNet no-relation ablation", "0.292", "0.292", f"{scannet_no_relation['efficiency']['checked_objects']['mean']:.1f}", f"{scannet_no_relation['efficiency']['estimated_tokens']['mean']:.0f}", f"{scannet_no_relation['efficiency']['runtime_seconds']['mean']*1000:.3f} ms"],
            ],
            [4.0 * cm, 2.1 * cm, 2.1 * cm, 2.8 * cm, 2.5 * cm, 2.9 * cm],
            styles,
            small=True,
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"},
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        two_figures(
            FIGURES / "fig_replica_pilot.png",
            "Replica: oracle-candidate ceiling; graph checks 83.2% fewer objects.",
            FIGURES / "fig_scannet_pilot.png",
            "ScanNet: graph matches flat lexical Acc@0.25 while checking 77.1% fewer objects.",
            styles,
            7.8,
            6.7,
        )
    )
    story.append(
        p(
            "Replica quality of 1.0 is a protocol sanity ceiling because the official target identities "
            "and boxes are candidates. ScanNet is harder: graph and flat lexical both reach 0.2708, "
            "while removing relation reranking reaches 0.2917. This isolates relation calibration as a "
            "concrete improvement target.",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # 10. Baselines.
    add_page(story, "10. Native ConceptGraphs and LangSplat executions", styles)
    story.append(
        p(
            "External baselines were progressed from smoke checks to native executions. ConceptGraphs "
            "builds predicted object maps for all eight selected ScanNet scenes and all eight Replica "
            "scenes. LangSplat completes RGB 3DGS optimization, segmentation/CLIP preprocessing, "
            "autoencoder compression, language-field optimization, rendering, and retrieval on "
            "ScanNet scene0011_00 under a 6 GB GPU profile. [E5-E7]",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Execution", "Scenes", "Queries", "Measured quality", "Runtime/query", "Local tokenizer total"],
                ["ConceptGraphs / ScanNet", "8", "48", f"Acc@0.25 {cg_scannet['bbox_acc_at_0_25']['value']:.4f}; mean IoU {cg_scannet['bbox_3d_iou']['value']:.4f}", f"{cg_scannet['runtime_seconds']['mean']:.3f} s", str(cg_scannet['token_usage']['total'])],
                ["ConceptGraphs / Replica", "8", "56", f"Acc@0.25 {cg_replica['bbox_acc_at_0_25']['value']:.4f}; mean IoU {cg_replica['bbox_3d_iou']['value']:.4f}", f"{cg_replica['runtime_seconds']['mean']:.3f} s", str(cg_replica['token_usage']['total'])],
                ["LangSplat / ScanNet", "1", "6", f"Expected-view hit {langsplat['expected_view_hit']['value']:.4f}; 3D IoU N/A", f"{langsplat['runtime_seconds']['mean']:.3f} s", str(langsplat['token_usage']['total'])],
            ],
            [4.0 * cm, 1.5 * cm, 1.7 * cm, 4.7 * cm, 2.3 * cm, 2.5 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(p("What these runs prove", styles, "H2"))
    story.append(
        bullets(
            [
                "The external repositories, models, checkpoints, preprocessing stages, native map construction, retrieval entry points, adapters, and canonical result export all execute locally.",
                "Token totals are measured local text-encoder non-padding tokens: 627 for ConceptGraphs ScanNet, 459 for ConceptGraphs Replica, and 74 for LangSplat.",
                "ConceptGraphs produces predicted 3D boxes, enabling GT IoU and Acc@threshold diagnostics.",
                "LangSplat produces relevance points/views rather than compatible predicted 3D boxes, so 3D Acc@threshold is not applicable.",
            ],
            styles,
        )
    )
    story.append(p("What these runs do not prove", styles, "H2"))
    story.append(
        p(
            "Map construction, frame sampling, query interfaces, output geometry, tokenizers, and "
            "hardware profiles differ from the SemanticSplat oracle-map pilots. The numbers therefore "
            "establish integration and diagnostic coverage, not a fair speed or quality ranking. "
            "LangSplat uses a hardware-adapted 3,000-iteration profile rather than the paper's "
            "30,000-iteration scale. BBQ official code has not been executed in this release.",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # 11. Construction accounting.
    add_page(story, "11. Graph construction and token accounting", styles)
    story.append(
        p(
            "The final audit separates historical manual annotation artifacts, deterministic local "
            "tree construction, serialized representation size, and query-time context. The builder "
            "makes zero provider calls. Its token-equivalent values are compact JSON character counts "
            "divided by four, not API billing tokens. [E2]",
            styles,
        )
    )
    story.append(
        metric_strip(
            [
                (f"{totals['source_viewjson_estimated_tokens_chars_div_4']:,}", "annotation payload token-equivalent"),
                (f"{totals['constructed_tree_estimated_tokens_chars_div_4']:,}", "constructed tree token-equivalent"),
                (f"{totals['serialized_io_estimated_tokens_chars_div_4']:,}", "serialized input + output"),
                (f"{totals['median_runtime_seconds_sum']:.3f} s", "sum of scene median rebuilds"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    scene_rows = [["Scene", "Views", "Items", "Nodes", "Source tok-eq.", "Tree tok-eq.", "Median rebuild"]]
    for scene in construction["scenes"]:
        scene_rows.append(
            [
                scene["scene_id"],
                str(scene["view_count"]),
                str(scene["semantic_item_count"]),
                str(scene["node_count"]),
                f"{scene['source_viewjson']['estimated_tokens_chars_div_4']:,}",
                f"{scene['constructed_tree_nodes']['estimated_tokens_chars_div_4']:,}",
                f"{scene['runtime_seconds']['median']:.3f} s",
            ]
        )
    story.append(
        report_table(
            scene_rows,
            [4.5 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm],
            styles,
            small=True,
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"},
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        report_table(
            [
                ["Token field", "Meaning", "Comparable to"],
                ["Internal estimated context tokens", "Serialized query context / 4", "Graph vs flat within the same internal protocol"],
                ["Construction token-equivalent", "Canonical JSON representation size / 4", "Artifact size across captured scenes"],
                ["ConceptGraphs tokens", "OpenCLIP ViT-H-14 non-padding tokens", "Only that native execution"],
                ["LangSplat tokens", "OpenCLIP ViT-B-16 non-padding tokens", "Only that native execution"],
                ["Provider tokens", "API input/output billing tokens", "None: no provider calls in the measured construction/query tracks"],
            ],
            [4.2 * cm, 7.6 * cm, 4.6 * cm],
            styles,
            small=True,
        )
    )
    story.append(PageBreak())

    # 12. Qualitative cases.
    add_page(story, "12. Qualitative evidence and failure analysis", styles)
    story.append(
        p(
            "Quantitative averages hide how hierarchy behaves. Conference Hall exposes three regimes: "
            "an isolated screen target is found from two graph views instead of a 19-view flat scan; a "
            "zone-wide projector retains sixteen views; and an exit-sign query remains a shared miss, "
            "although graph search stops after three views. [E10]",
            styles,
        )
    )
    story.append(
        figure(
            FIGURES / "fig_failure_cases.png",
            "Room-scoped case studies. Green indicates correct graph retrieval; red indicates a shared miss.",
            styles,
            15.7,
            7.0,
        )
    )
    story.append(
        figure(
            FIGURES / "fig_qualitative_bbox.png",
            "Grounded screen evidence in stage view v018 and a flat full-scan distractor view.",
            styles,
            15.7,
            6.2,
        )
    )
    story.append(p("Failure taxonomy", styles, "H2"))
    story.append(
        report_table(
            [
                ["Failure", "Observed meaning", "Research response"],
                ["Wrong object", "Lexical match selects a same-class distractor", "Improve object labels, embeddings, and reranking"],
                ["Wrong relation", "Target label is plausible but anchor geometry is misused", "Calibrate relation confidence; do not override stronger label evidence"],
                ["Wrong view/room", "Evidence is semantically related but spatially wrong", "Retain zone constraints and verified-view evaluation"],
                ["Shared miss", "Graph and flat both lack a strong candidate", "Improve semantic map coverage rather than widening traversal alone"],
                ["Unavailable geometry", "Method output cannot be compared to GT boxes", "Report metric as N/A with a concrete reason"],
            ],
            [3.1 * cm, 6.5 * cm, 6.8 * cm],
            styles,
            small=True,
        )
    )
    story.append(PageBreak())

    # 13. Publication and release.
    add_page(story, "13. Article, repository, and public project page", styles)
    story.append(
        p(
            "The research is now packaged in two manuscript forms. The TwinWorld source preserves an "
            "anonymous ECCV-workshop review artifact and submission checklists. The current public-facing "
            "paper is a named-author, academic/arXiv-style preprint titled <b>SemanticSplat: Graph-Pruned "
            "Semantic Search</b>. Its nine-page PDF includes the expanded method, metric contract, "
            "public pilots, native baseline evidence, limitations, reproducibility commands, and 21 "
            "references. [E10]",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Release component", "Status", "What a reviewer can inspect"],
                ["Academic preprint", "DONE", "Named authors, method figures, tables, limitations, references, and reproducibility section"],
                ["TwinWorld package", "DONE repository-side", "ECCV-style anonymous source/PDF, claim audit, number checks, and submission checklist"],
                ["Public repository", "DONE", "Backend/query code, experiment scripts, schemas, canonical outputs, documentation, and citation metadata"],
                ["GitHub Pages", "ONLINE; refresh pending", "Latest site is tested on revision 8f92633; Leo-owned deployment awaits the upstream branch merge"],
                ["Licensed data handling", "DONE", "Scene IDs and access links are public; raw Replica/ScanNet data are excluded"],
            ],
            [4.1 * cm, 3.1 * cm, 9.2 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        figure(
            FIGURES / "fig_tree_traversal.png",
            "The public page exposes the same hierarchy and evidence figures used by the paper, avoiding a separate marketing-only story. [E10, E11]",
            styles,
            12.5,
            6.9,
        )
    )
    story.append(
        p(
            "Public URL: <link href='https://leopython2006.github.io/beyond-proximity-public/'>"
            "https://leopython2006.github.io/beyond-proximity-public/</link><br/>"
            "Local rehearsal URL: <link href='http://127.0.0.1:4173/'>http://127.0.0.1:4173/</link><br/>"
            "Latest public-source branch: BearAx/codex/project-site-method-figure @ 8f92633<br/>"
            "Deployment gate: merge that branch into LeoPython2006/codex/project-site-method-figure<br/>"
            "Repository: <link href='https://github.com/LeoPython2006/beyond-proximity'>"
            "https://github.com/LeoPython2006/beyond-proximity</link>",
            styles,
            "CodeBlock",
        )
    )
    story.append(PageBreak())

    # 14. Reproducibility.
    add_page(story, "14. Reproducibility and verification", styles)
    story.append(
        p(
            "The final package is designed so claims can be regenerated or checked without reading "
            "the paper manually. Canonical results are JSON, figures consume frozen JSON, and automated "
            "checks compare manuscript/site strings with the same metric sources. [E9, E12]",
            styles,
        )
    )
    story.append(
        metric_strip(
            [
                ("145", "passing backend/tests"),
                ("100%", "schema-valid public pilot outputs"),
                ("14", "versioned evidence groups cited"),
                ("2", "paper consistency gates"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(p("Core verification commands", styles, "H2"))
    story.append(
        p(
            "python -B -m pytest -q tests -p no:cacheprovider<br/>"
            "python -B scripts/check_academic_paper.py<br/>"
            "python -B scripts/check_twinworld_numbers.py<br/>"
            "python -B scripts/build_project_site.py<br/>"
            "python -B scripts/generate_final_project_journey_report.py",
            styles,
            "CodeBlock",
        )
    )
    story.append(p("Reproducibility layers", styles, "H2"))
    story.append(
        report_table(
            [
                ["Layer", "Frozen artifact", "Purpose"],
                ["Data contract", "Scene manifests, schemas, ViewJSON, official access notes", "Defines what enters each track"],
                ["Experiment contract", "Configs, exact scene/query lists, variant names", "Fixes the comparison basis"],
                ["Result contract", "Per-query JSON and aggregate summaries", "Preserves predictions, traces, failures, and metrics"],
                ["Paper contract", "Number checks and claim audit", "Prevents stale counts and unsupported language"],
                ["Release contract", "Pages builder, workflow, CITATION.cff", "Publishes only permitted artifacts"],
                ["Visual QA", "Rendered paper/report page images", "Catches clipping, overlap, unreadable tables, and float errors"],
            ],
            [3.3 * cm, 6.7 * cm, 6.4 * cm],
            styles,
            small=True,
        )
    )
    story.append(
        p(
            "The latest validation run for this report passed all 145 tests, the academic release "
            "consistency check, the manuscript/site number gate, the project-site build, PDF parsing, "
            "and visual page inspection.",
            styles,
            "Success",
        )
    )
    story.append(PageBreak())

    # 15. Plan verdict.
    add_page(story, "15. How well the plan was completed", styles)
    story.append(
        p(
            "Against the written TwinWorld two-week plan, all repository-side acceptance criteria are "
            "complete. The final artifact goes beyond the original plan by adding official ScanNet and "
            "Replica coverage, native public-dataset ConceptGraphs runs, a complete LangSplat execution, "
            "a named-author academic preprint, construction/token accounting, and a public project page.",
            styles,
        )
    )
    story.append(
        report_table(
            [
                ["Plan requirement", "Final status", "Evidence / boundary"],
                ["TwinWorld topic and ECCV format", "DONE", "Workshop source and audit exist; current public paper is intentionally venue-neutral"],
                ["Replica and ScanNet", "DONE", "Exact eight-scene BBQ subsets, official boxes, and query pilots"],
                ["BBQ alignment", "DONE", "Scenes, query sources, grounding metrics, and comparison rules; no official BBQ run"],
                ["Five captured scenes", "DONE", "97 views, 1,066 items, 150 queries"],
                ["Public grounding metrics", "DONE", "Recall@1 and Acc@0.1/0.25/0.5; segmentation metrics correctly remain N/A"],
                ["Efficiency and construction cost", "DONE", "Views/nodes/objects, context, tokens, runtime, and one-time construction"],
                ["Required baselines", "DONE", "Graph, flat lexical, flat embedding, fallback, ablations, and external statuses/runs"],
                ["Visual evidence", "DONE", "Workflow, tree, scenes, protocol, results, and grounded failure cases"],
                ["Reproducibility", "DONE", "Frozen outputs, schemas, commands, tests, number gate, data-license notes"],
                ["Limitations and claim discipline", "DONE", "Hard limitations section and no fair-superiority claim"],
                ["External submission actions", "TEAM ACTION", "Venue account/profile checks, final upload, registration, and presentation"],
            ],
            [5.2 * cm, 2.7 * cm, 8.5 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "<b>Overall assessment:</b> 4/4 role acceptance records and 10/10 repository-side "
            "readiness categories are complete. The plan was completed at the implementation and "
            "artifact level. It should not be reinterpreted as 100% scientific accuracy: ScanNet "
            "quality is still low, internal graph hit@k trails flat search, segmentation prediction "
            "is absent, and external methods are not yet compared under one shared protocol.",
            styles,
            "Success",
        )
    )
    story.append(PageBreak())

    # 16. Limitations and next research.
    add_page(story, "16. Conclusions, limitations, and the next research step", styles)
    story.append(
        p(
            "The project now supports one strong conclusion: a deterministic semantic hierarchy can "
            "remove most scene context before query grounding. It also exposes the cost of hard "
            "pruning: relevant but weakly summarized paths can be discarded. Public oracle-map pilots "
            "show that hierarchy can preserve flat lexical quality when candidate labels align with "
            "queries, while the ScanNet relation ablation shows that uncertain geometry can reduce "
            "accuracy. External executions prove integration breadth but not superiority.",
            styles,
            "Callout",
        )
    )
    story.append(p("Remaining limitations", styles, "H2"))
    story.append(
        bullets(
            [
                "Internal ViewJSON and expected-view labels are manual reference artifacts and may contain correlated labeling bias.",
                "The internal ranker is deterministic lexical logic, not a live provider-backed vision-language model.",
                "Replica and ScanNet SemanticSplat pilots use official semantic objects as map candidates and do not measure automatic perception.",
                "ScanNet exact-object/Acc@0.25 quality is 0.2708; the no-relation ablation is better at 0.2917.",
                "ConceptGraphs and LangSplat use different construction, geometry, resource, and query protocols.",
                "Historical human annotation wall-clock time was not logged; artifact size and deterministic rebuild time are measured instead.",
            ],
            styles,
            color=AMBER,
        )
    )
    story.append(p("Recommended next research sequence", styles, "H2"))
    story.append(
        report_table(
            [
                ["Priority", "Work", "Success criterion"],
                ["P0", "Calibrated multi-branch pruning and confidence-based fallback", "Recover flat hit@k while retaining most context reduction"],
                ["P1", "Predicted-map SemanticSplat on public RGB-D scenes", "Report perception + retrieval separately; add segmentation/box prediction metrics"],
                ["P2", "Shared-scene, shared-query baseline protocol", "Compare map construction, retrieval quality, runtime, memory, and tokens fairly"],
                ["P3", "Independent GT review and annotation-time study", "Reduce label correlation risk and report human cost"],
                ["P4", "Scale to more public scenes and query distributions", "Tighter uncertainty intervals and stronger generalization"],
                ["P5", "Publication packaging", "Select venue, fit page limits, update venue-specific policy checks, and submit"],
            ],
            [1.5 * cm, 8.2 * cm, 6.7 * cm],
            styles,
        )
    )
    story.append(
        p(
            "Venue direction from the existing assessment is still sensible: a workshop submission is "
            "the best fit for the current system-and-evidence package, while WACV/3DV becomes the "
            "stronger main-conference route after calibrated pruning and predicted-map evaluation. "
            "AAAI main requires a broader algorithmic contribution than the current deterministic "
            "retrieval prototype. [E14]",
            styles,
            "Warning",
        )
    )
    story.append(PageBreak())

    # Appendix A.
    add_page(story, "Appendix A. Evidence index", styles)
    story.append(
        p(
            "Every identifier used in the report maps to a local, versioned artifact. Raw licensed "
            "Replica and ScanNet scene data are intentionally excluded from the public release.",
            styles,
        )
    )
    evidence_rows = [["ID", "Repository evidence path"]]
    for evidence_id, evidence_path in EVIDENCE.items():
        evidence_rows.append([evidence_id, evidence_path])
    story.append(
        report_table(
            evidence_rows,
            [2.0 * cm, 14.4 * cm],
            styles,
            small=True,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(p("Primary frozen run identifiers", styles, "H2"))
    story.append(
        report_table(
            [
                ["Purpose", "Run / artifact"],
                ["Internal graph vs flat", "five_scene_graph_vs_flat_v2"],
                ["Replica oracle-map pilot", "replica_pilot_v1"],
                ["ScanNet oracle-map pilot", "scannet_pilot_v1"],
                ["ConceptGraphs ScanNet", "conceptgraphs_scannet_full_v1"],
                ["ConceptGraphs Replica", "conceptgraphs_replica_full_v1"],
                ["LangSplat ScanNet", "langsplat_scannet_full_v1"],
                ["Week 6 integration", "b23359a / provenance freeze 9ec4360"],
                ["Current report revision", revision],
            ],
            [5.5 * cm, 10.9 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "Report generator: <b>scripts/generate_final_project_journey_report.py</b><br/>"
            f"Generated PDF: <b>{PDF_PATH.relative_to(ROOT).as_posix()}</b><br/>"
            "Demo script: <b>docs/reports/final/DEMO_VIDEO_SCRIPT_8_MIN.md</b>",
            styles,
            "CodeBlock",
        )
    )

    doc.multiBuild(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return PDF_PATH


if __name__ == "__main__":
    print(build_report())
