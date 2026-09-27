# Query Fairness Review

Baseline families expose different native capabilities. Results must be reported per bucket and per native/extended method variant; one aggregate score would hide material task mismatch.

| Bucket | Language-field baselines | Graph-based baselines | SemanticSplat | Fair reporting rule |
|---|---|---|---|---|
| Simple object | Native and fair. | Native and fair if the object mapper detects the target. | Native semantic-index query. | Compare all methods; separate detection/mapping failures. |
| Attribute | Fair for visually grounded attributes represented by the encoder. | Fair when native object captions/features preserve the attribute. | Fair when source view descriptions preserve the attribute. | Restrict to visible attributes; report caption/feature omissions. |
| Relational | Not native without object extraction and spatial post-processing. | Native and fair when graph edges encode the relation. | Intended native query type. | Report language-field post-processing as a separate extended variant. |
| Zone/hierarchy | No native hierarchy. | Partially fair only if room/zone nodes exist natively. | Favors SemanticSplat's explicit hierarchy. | Report separately; do not include in a universal aggregate. |
| Functional | Usually not native; may rely on text-feature priors. | Possible with captions plus language reasoning, but not guaranteed native. | Intended VLM/LLM reasoning type in live mode. | Require visible evidence and record whether external LLM reasoning is used. |
| Aggregation | Relevancy fields do not naturally enumerate complete instances. | Fair only if object recall is measured and sufficiently complete. | Intended tree/index operation but depends on view coverage. | Report count error plus upstream object-recall coverage; never infer missing GT. |
| Negative | Requires a fixed confidence threshold and calibrated absence rule. | Requires complete-enough mapping and an explicit no-match rule. | Requires explicit no-match behavior and coverage caveat. | Tune no threshold on test queries; report false positives and false negatives separately. |

## Current Benchmark Mapping

`benchmark_queries_v1.json` currently uses `simple_object`, `attribute`, `relational`, `multi_hop`, `functional`, and `negative`. `multi_hop` should be split by analysis tag into relational versus zone/hierarchy behavior. A dedicated aggregation bucket is not yet present and must not be claimed as evaluated.

The current expected view/node/zone identifiers come from the repository's semantic index, not independent object-level ground truth. They support deterministic regression tests, not final comparative accuracy claims.

## Required Result Slices

Every comparison table must show:

- coverage and failures before accuracy;
- each query bucket separately;
- native method versus adapter-extended method;
- semantic-only metrics versus geometry-dependent metrics;
- live measured runs separately from cached, stub, and simulated runs;
- construction/preprocessing time separately from query latency;
- `N/A` where a method or scene cannot support a metric.

## Fairness Decisions for Week 2/3

- Mandatory future smoke candidates are LangSplat for language fields and ConceptGraphs for object graphs.
- The first baseline smoke may use only compatible simple-object queries and must not be presented as a full comparison.
- Relational or hierarchy conclusions require bucket-specific evidence, not the current stub results.
- Constant-depth `default` data is allowed for semantic plumbing tests but not for baseline 3D localization comparison.
