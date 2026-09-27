from __future__ import annotations

import json
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
PDF_PATH = OUT_DIR / "semanticsplat_progress_next_steps_report.pdf"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.2,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.2,
            textColor=colors.HexColor("#374151"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.9,
            leading=10.2,
            textColor=colors.HexColor("#111827"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.white,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#E0F2FE"),
            borderColor=colors.HexColor("#0284C7"),
            borderWidth=0.8,
            borderPadding=7,
            spaceAfter=10,
        )
    )
    return styles


def p(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet_list(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles, "Body"), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontName="Helvetica",
        bulletFontSize=6,
        bulletColor=colors.HexColor("#0F766E"),
    )


def table(data: list[list[str]], col_widths: list[float], styles) -> Table:
    body = []
    for row_index, row in enumerate(data):
        row_style = "TableHeader" if row_index == 0 else "TableCell"
        body.append([p(str(cell), styles, row_style) for cell in row])
    t = Table(body, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return t


def image(path: Path, width_cm: float) -> Image:
    img = Image(str(path))
    max_width = width_cm * cm
    ratio = max_width / img.imageWidth
    img.drawWidth = max_width
    img.drawHeight = img.imageHeight * ratio
    img.hAlign = "CENTER"
    return img


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 1.2 * cm, width - doc.rightMargin, height - 1.2 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(doc.leftMargin, height - 0.85 * cm, "SemanticSplat")
    canvas.drawRightString(width - doc.rightMargin, height - 0.85 * cm, "Progress and Next Steps")
    canvas.line(doc.leftMargin, 1.1 * cm, width - doc.rightMargin, 1.1 * cm)
    canvas.drawString(doc.leftMargin, 0.72 * cm, "Generated 2026-07-06")
    canvas.drawRightString(width - doc.rightMargin, 0.72 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = read_json(ROOT / "outputs" / "graph_vs_flat" / "five_scene_graph_vs_flat_v2" / "metrics_summary.json")
    summary = metrics["summary"]
    avg = summary["averages"]
    quality = summary["quality"]

    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.55 * cm,
        title="SemanticSplat Progress and Next Steps Report",
        author="SemanticSplat Team",
    )

    story = []
    story.append(Spacer(1, 2.0 * cm))
    story.append(p("Beyond Proximity: Graph-Pruned Semantic Search over 3D Scene Records", styles, "ReportTitle"))
    story.append(p("Progress Report and Next Steps", styles, "ReportTitle"))
    story.append(
        p(
            "Status date: 2026-07-06 | Branch: codex/research-upgrade-baselines-datasets | "
            "Frozen evidence commit: 8700f26",
            styles,
            "Subtitle",
        )
    )
    story.append(
        p(
            "Short answer: the project has a credible workshop-ready semantic-search prototype "
            "with reproducible graph-vs-flat evidence. The next scientific step is to close the "
            "quality gap and add public-dataset/fair-baseline evidence before aiming at a stronger "
            "main-conference submission.",
            styles,
            "Callout",
        )
    )
    story.append(
        table(
            [
                ["Recommended target", "Why"],
                [
                    "TwinWorld @ ECCV 2026",
                    "Best topical fit for digital twins, semantic 3D, 3DGS, scene understanding, "
                    "and honest workshop-scale evidence.",
                ],
                [
                    "WACV 2027 Round 2",
                    "Best next main-conference path after adding embedding retrieval, calibrated "
                    "pruning, public-data evaluation, and stronger failure analysis.",
                ],
                [
                    "Avoid AAAI 2027 main for now",
                    "Current evidence is too small and too systems-oriented; graph hit@k is lower "
                    "than flat under the current lexical ranker.",
                ],
            ],
            [5.0 * cm, 11.1 * cm],
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("1. What We Completed", styles, "SectionTitle"))
    story.append(
        bullet_list(
            [
                "Integrated the Week 4 research branch into a reproducible evidence state and continued on codex/research-upgrade-baselines-datasets.",
                "Updated the project truth state: five captured pilot scenes now contain 97 captured views and 1,066 semantic items.",
                "Regenerated the primary five-scene graph-vs-flat benchmark on 150 queries with 125 verified-view-label queries.",
                "Added graph-pruning controls and tests for tighter, default, broad, and affordance-aware traversal variants.",
                "Added ablation and scaling stress scripts plus checked-in evidence outputs.",
                "Reran LangSplat and ConceptGraphs smoke baselines and documented them as setup/adaptor evidence only.",
                "Completed official BBQ-aligned Replica and ScanNet oracle-GT-map object-grounding pilots with reproducible validation and metrics.",
                "Updated the paper source and rebuilt the article PDF around the correct claim: strong cost reduction with a measured quality trade-off.",
                "Created a publication venue fit assessment recommending TwinWorld first and WACV next.",
            ],
            styles,
        )
    )

    story.append(p("Current Evidence Snapshot", styles, "SmallTitle"))
    story.append(
        table(
            [
                ["Artifact", "Status", "Evidence path"],
                ["Five-scene benchmark", "DONE", "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/"],
                ["Ablation study", "DONE", "outputs/graph_vs_flat/ablations_v1/"],
                ["Scaling stress test", "DONE", "outputs/graph_vs_flat/scaling_stress_v1/"],
                ["LangSplat baseline", "SMOKE ONLY", "outputs/baselines/langsplat_smoke_v1/"],
                ["ConceptGraphs baseline", "SMOKE ONLY", "outputs/baselines/conceptgraphs_smoke_v1/"],
                ["Public Replica", "GT PILOT DONE", "docs/experiments/public_datasets/replica_bbq_aligned_v3_metrics_summary.md"],
                ["ScanNet", "GT PILOT DONE", "docs/experiments/public_datasets/scannet_bbq_grounding_v2_metrics_summary.md"],
                ["Article PDF", "UPDATED", "papers/beyond-proximity/main.pdf"],
            ],
            [4.0 * cm, 3.0 * cm, 9.1 * cm],
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("2. Main Quantitative Result", styles, "SectionTitle"))
    story.append(
        p(
            "The central result is a same-input comparison: graph and flat methods use the same "
            "scenes, semantic items, query strings, and verified view labels. This isolates the "
            "effect of hierarchical pruning.",
            styles,
        )
    )
    story.append(
        table(
            [
                ["Metric", "Flat", "Graph", "Interpretation"],
                ["Views checked", f"{avg['flat_views_checked']:.2f}", f"{avg['graph_views_checked']:.2f}", "75.5% fewer views"],
                ["Estimated input tokens", f"{avg['flat_input_tokens']:.1f}", f"{avg['graph_input_tokens']:.1f}", "68.2% fewer tokens"],
                ["Context chars", f"{avg['flat_context_chars']:.1f}", f"{avg['graph_context_chars']:.1f}", "Less query-time context"],
                ["Infra elapsed ms", f"{avg['flat_elapsed_ms']:.2f}", f"{avg['graph_elapsed_ms']:.2f}", "Small local runtime difference"],
                ["hit@1", f"{quality['hit_at_1_flat']:.3f}", f"{quality['hit_at_1_graph']:.3f}", "Graph lower than flat"],
                ["hit@3", f"{quality['hit_at_3_flat']:.3f}", f"{quality['hit_at_3_graph']:.3f}", "Graph lower than flat"],
            ],
            [4.0 * cm, 2.3 * cm, 2.3 * cm, 7.5 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    fig_summary = ROOT / "papers" / "beyond-proximity" / "figures" / "fig_five_scene_summary.png"
    if fig_summary.exists():
        story.append(image(fig_summary, 14.8))
    story.append(
        p(
            "Correct interpretation: SemanticSplat currently proves cost control, not superiority. "
            "The graph prunes aggressively and saves context, but under the lexical stub ranker it "
            "also loses some correct candidates.",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    story.append(p("3. Ablations And Scaling", styles, "SectionTitle"))
    story.append(
        p(
            "The ablation study gives the next algorithmic direction. Tight graph traversal saves "
            "the most cost but hurts hit@k; broad traversal recovers recall but loses token "
            "efficiency. The project needs calibrated pruning or confidence-based fallback.",
            styles,
        )
    )
    story.append(
        table(
            [
                ["Variant", "Views", "Token savings", "hit@1", "hit@3", "Takeaway"],
                ["flat_lexical", "19.40", "0.0%", "0.768", "0.928", "Current exhaustive baseline"],
                ["flat_affordance", "19.40", "0.0%", "0.784", "0.944", "Best current quality reference"],
                ["graph_tight", "2.07", "85.77%", "0.600", "0.720", "High savings, too much pruning"],
                ["graph_default", "4.77", "67.99%", "0.680", "0.808", "Current paper setting"],
                ["graph_broad", "19.40", "-100.33%", "0.728", "0.920", "Recall improves but cost doubles"],
                ["graph_affordance", "4.99", "66.47%", "0.680", "0.816", "Small hit@3 gain"],
            ],
            [3.0 * cm, 1.65 * cm, 2.3 * cm, 1.55 * cm, 1.55 * cm, 6.1 * cm],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    fig_type = ROOT / "papers" / "beyond-proximity" / "figures" / "fig_by_query_type.png"
    if fig_type.exists():
        story.append(image(fig_type, 14.8))
    story.append(p("Scaling Stress Test", styles, "SmallTitle"))
    story.append(
        table(
            [
                ["Multiplier", "Flat views", "Graph views", "View savings", "Token savings"],
                ["1x", "20.0", "4.32", "78.42%", "71.20%"],
                ["2x", "40.0", "7.43", "81.42%", "75.38%"],
                ["5x", "100.0", "16.78", "83.22%", "77.89%"],
                ["10x", "200.0", "32.37", "83.82%", "78.73%"],
            ],
            [2.7 * cm, 2.5 * cm, 2.5 * cm, 3.0 * cm, 3.0 * cm],
            styles,
        )
    )
    story.append(
        p(
            "Guardrail: the scaling study duplicates existing semantic views in memory. It is useful "
            "query-time stress evidence, not a substitute for Replica or ScanNet.",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("4. What Is Still Limited", styles, "SectionTitle"))
    story.append(
        table(
            [
                ["Limitation", "How serious?", "Do we need to fix it?"],
                [
                    "Manual/reference labels, not independent GT",
                    "Medium for workshop, high for main conference",
                    "Yes before WACV/3DV/AAAI if making accuracy claims.",
                ],
                [
                    "Graph hit@k lower than flat",
                    "High for algorithmic claims",
                    "Yes. Add calibrated pruning, fallback expansion, or embedding node scoring.",
                ],
                [
                    "LangSplat and ConceptGraphs are smoke only",
                    "Acceptable for workshop scope, weak for main conference",
                    "Yes before claiming baseline comparison.",
                ],
                [
                    "Public tracks use oracle GT maps rather than semantic predictions",
                    "Low for retrieval claims, high for perception claims",
                    "Label the scope explicitly and add predicted segmentation before mIoU claims.",
                ],
                [
                    "No 3D IoU or metric localization",
                    "High for 3D localization/navigation claims",
                    "Only needed if the paper claims object boxes, metric grounding, or navigation.",
                ],
                [
                    "Article is not anonymized or venue-formatted",
                    "Blocking for submission",
                    "Yes. Prepare review PDF in the selected template.",
                ],
            ],
            [4.1 * cm, 4.4 * cm, 7.6 * cm],
            styles,
        )
    )
    story.append(p("Publication Fit", styles, "SmallTitle"))
    story.append(
        table(
            [
                ["Venue", "Current fit", "After realistic improvements", "Decision"],
                ["TwinWorld @ ECCV 2026", "78/100", "88/100", "Best immediate target"],
                ["WACV 2027", "66/100", "84/100", "Best main-conference path"],
                ["3DV 2027", "61/100", "80/100", "Good after stronger 3D grounding"],
                ["NeurIPS 2026 workshops", "55/100", "75/100", "Opportunistic backup"],
                ["ICRA 2027", "43/100", "70/100", "Needs embodied-agent evaluation"],
                ["AAAI 2027 main", "38/100", "68/100", "Too risky now"],
            ],
            [4.6 * cm, 2.5 * cm, 3.5 * cm, 5.5 * cm],
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("5. Next Steps", styles, "SectionTitle"))
    story.append(p("Immediate TwinWorld Sprint: 2026-07-06 to 2026-07-31", styles, "SmallTitle"))
    story.append(
        table(
            [
                ["Priority", "Task", "Owner type", "Output"],
                ["P0", "Choose venue strategy: short 4-page vs full 8-10-page TwinWorld paper.", "Team decision", "Submission plan"],
                ["P0", "Convert paper to ECCV LNCS style and anonymize.", "Paper lead", "Review PDF"],
                ["P0", "Add system overview and query-flow figures.", "Figures/UX lead", "Two paper figures"],
                ["P0", "Add real scene screenshots or graph overlays.", "Data/visual lead", "At least one qualitative figure"],
                ["P0", "Insert ablation table and explain cost/quality trade-off.", "Evaluation lead", "Updated experiments section"],
                ["P1", "Freeze commit hash, run IDs, and reproduction commands.", "Engineering lead", "Supplementary README"],
                ["P1", "Prepare demo/project page or short video if time allows.", "Demo lead", "Submission supplement"],
            ],
            [1.5 * cm, 7.2 * cm, 3.0 * cm, 4.4 * cm],
            styles,
        )
    )
    story.append(p("Main-Conference Research Sprint: August 2026", styles, "SmallTitle"))
    story.append(
        bullet_list(
            [
                "Add a flat embedding baseline and compare it to flat lexical, graph default, and graph affordance.",
                "Implement calibrated graph pruning or low-confidence fallback expansion to recover hit@k while keeping cost low.",
                "Improve the completed ScanNet relational grounding baseline using target-anchor spatial scoring.",
                "Turn ConceptGraphs and/or LangSplat from smoke evidence into fair same-scene baseline evidence where feasible.",
                "Add current five-scene failure analysis focused on missed graph candidates, wrong zones, and missing affordances.",
                "Prepare an anonymized supplementary package with code, query set, schemas, and reproduction commands.",
            ],
            styles,
        )
    )
    story.append(p("Decisions Needed From The Team", styles, "SmallTitle"))
    story.append(
        table(
            [
                ["Decision", "Recommended answer"],
                ["Where to submit first?", "TwinWorld @ ECCV 2026 if a workshop publication is acceptable."],
                ["Should the TwinWorld paper be short or full?", "Short if preserving future main-conference novelty matters; full if TwinWorld is the main target."],
                ["Do we have Replica/ScanNet now?", "Yes for oracle-GT-map object grounding; no for model-predicted semantic segmentation."],
                ["Can we claim superiority over LangSplat/ConceptGraphs?", "No. Current evidence supports smoke execution only."],
                ["Can we say graph preserves quality?", "No. Current evidence shows lower hit@k; say cost/quality trade-off."],
            ],
            [5.0 * cm, 11.1 * cm],
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("6. Final Recommendation", styles, "SectionTitle"))
    story.append(
        p(
            "The project is in a good place for a careful workshop paper. It has real scenes, a "
            "reproducible benchmark, ablations, scaling stress evidence, baseline smoke tests, "
            "and an honest paper draft. The strongest current claim is not that the graph is more "
            "accurate; it is that hierarchy substantially reduces context/search cost and exposes a "
            "clear recall-cost frontier.",
            styles,
        )
    )
    story.append(
        p(
            "For TwinWorld, polish the paper and show the system clearly. For WACV/3DV, improve "
            "the science: add embedding retrieval, calibrated pruning, public dataset evaluation, "
            "and fair external baselines. That is the path from a solid prototype paper to a serious "
            "main-conference submission.",
            styles,
            "Callout",
        )
    )
    story.append(p("Referenced Project Artifacts", styles, "SmallTitle"))
    story.append(
        bullet_list(
            [
                "papers/beyond-proximity/main.tex",
                "papers/beyond-proximity/main.pdf",
                "PUBLICATION_VENUE_FIT_ASSESSMENT.md",
                "docs/reports/final/claim_audit.md",
                "docs/reports/final/research_upgrade_status.md",
                "docs/datasets/public_dataset_readiness.md",
                "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/metrics_summary.json",
                "outputs/graph_vs_flat/ablations_v1/ablation_summary.md",
                "outputs/graph_vs_flat/scaling_stress_v1/scaling_summary.md",
            ],
            styles,
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return PDF_PATH


if __name__ == "__main__":
    path = build_report()
    print(path)
