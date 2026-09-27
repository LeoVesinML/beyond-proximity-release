# BBQ-aligned grounding metrics

`scripts/evaluate_grounding.py` evaluates canonical query-result files against
benchmark grounding GT. It produces:

- `grounding_summary.json`: complete metrics, strata, coverage, and per-query records;
- `grounding_summary.csv`: one row per benchmark query;
- `grounding_summary.md`: human-readable summary;
- `grounding_failures.json`: aggregate taxonomy and per-query failures.

Run it with:

```bash
python -m scripts.evaluate_grounding \
  --benchmark docs/benchmarks/replica_scannet_pilot_queries.json \
  --results outputs/<run>/query_results \
  --out outputs/<run>/grounding_evaluation \
  --run-config outputs/<run>/run_config.json
```

To generate the canonical graph, fallback, flat lexical, local CPU embedding,
and four ablation runs before evaluation:

```bash
python -B scripts/run_grounding_variants.py \
  --config configs/replica_benchmark.yaml \
  --out outputs/public_datasets/replica_pilot_v1
python -B scripts/run_grounding_variants.py \
  --config configs/scannet_benchmark.yaml \
  --out outputs/public_datasets/scannet_pilot_v1
```

The runner freezes a JSON/CSV/MD comparison and per-variant canonical results.
Its embedding backend is `fastembed` on CPU and records the exact model and
cache location. It never silently substitutes lexical retrieval.

## Eligibility and formulas

A query is 3D-box eligible when it has at least one finite, positive-volume
`expected_bbox_3d` or `expected_bboxes_3d` record and its verification status
is neither `missing_gt` nor `ambiguous`. `gt_3d_reliable` and the benchmark's
verification source should describe the GT provenance. Invalid GT boxes do not
create an eligible sample.

For prediction box \(P\) and GT box \(G\):

\[
\operatorname{IoU}_{3D}(P,G)=
\frac{\operatorname{volume}(P\cap G)}
{\operatorname{volume}(P\cup G)}.
\]

If multiple GT boxes are acceptable, the evaluator uses
\(\max_G \operatorname{IoU}_{3D}(P,G)\). Every GT-eligible query contributes
exactly once. A missing result, unavailable result, missing prediction box, or
invalid prediction box contributes IoU 0; it is not removed from the
denominator.

\[
\operatorname{Acc}@t =
\frac{\sum_q [\operatorname{IoU}_{3D}(q)\ge t]}
{N_{\text{3D-box eligible}}}
\quad\text{for }t\in\{0.1,0.25,0.5\}.
\]

Recall@1 is exact object-ID retrieval:

\[
\operatorname{Recall}@1 =
\frac{\sum_q [\hat{o}_q\in O_q]}
{N_{\text{queries with verified expected object IDs}}}.
\]

Labels, selected views, and oracle-retrieved objects are not substitutes for
an exact object ID.

## Quality, efficiency, and construction cost

The same summary reports 3D IoU, Acc thresholds, and Recall@1 alongside:

- wall-clock query runtime;
- checked/visited nodes;
- checked views;
- checked objects (or a labeled inherited scan-count alias);
- serialized context characters;
- estimated input tokens.

Each efficiency field includes a mean, total, and denominator so missing
instrumentation is visible. Construction/index/map cost is reported in a
separate top-level section because it is normally paid once and must not be
mixed into per-query runtime or search cost.

Metrics are stratified independently by scene, source dataset, query type, and
search variant. `method` is used as the inherited search-variant fallback when
no explicit `search_variant` exists.

## Failure taxonomy

Per-query and aggregate failures use these labels:

- `wrong object`: a valid available prediction has the wrong exact object ID;
- `wrong relation`: a relational or multi-hop result explicitly fails its
  relation check;
- `invalid/unavailable prediction`: missing files, unavailable outputs,
  schema-invalid outputs, or `found=false`;
- `bad bbox`: a 3D-box-eligible query has no valid prediction box;
- `missing GT`: neither verified box GT nor verified object-ID GT is available.

A query may have more than one failure label. Counts therefore need not sum to
the query count.

## Segmentation metrics

Conditional mAcc, mIoU, and frequency-weighted mIoU (fmIoU) are computed only
when a prediction label array/mask and a same-sized GT label array/mask both
exist. For class \(c\):

\[
\operatorname{Acc}_c=TP_c/N^{GT}_c,\qquad
\operatorname{IoU}_c=TP_c/(TP_c+FP_c+FN_c).
\]

mAcc and mIoU are unweighted class means. fmIoU weights each class IoU by its
GT pixel/point frequency. Declared ignore labels are excluded.

When joint segmentation arrays are absent, all three metrics are explicitly
`N/A` with a reason. They are never inferred from successful object retrieval,
oracle object IDs, 3D boxes, or semantic-index labels.

## Schema and legacy handling

Canonical inputs use `semanticsplat.query_result.v1`. The evaluator checks the
query and scene identity, `result.found`, `result.bbox_3d`, an object
identifier, and required grounding/efficiency fields. Older inherited shapes
can be adapted from top-level/`answer` and `metrics` fields, but are labeled
`legacy_adapted`, are not counted as canonical/schema-valid, and retain their
validation errors. This prevents silent promotion of legacy records.

## GT provenance, exclusions, and claims

Report the benchmark's real GT source: official dataset annotations, manual
boxes, or ViewJSON-derived references. Manual and ViewJSON-derived references
are internal evaluation GT, not independent dataset GT. Queries with missing,
ambiguous, malformed, or unsupported-modality GT are excluded only from the
metric that requires that GT; their result coverage and failures remain
visible.

These metrics are aligned with common BBQ-style 3D grounding measures, but
this evaluator alone does **not** establish a fair BBQ benchmark comparison.
Do not claim BBQ reproduction, BBQ leaderboard performance, baseline
superiority, semantic accuracy, or segmentation quality without matching
dataset splits, official GT, candidate construction, prediction protocol, and
baseline execution conditions.
