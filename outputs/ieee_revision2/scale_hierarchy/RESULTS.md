# Scale hierarchy (Algorithm 1 without oracle scan ID) on ReferIt3D pools

Generated 2026-09-27 (run `generated_at` 2026-09-27T19:26:02Z). Every number here is
copied from `summary.json`, `scale_hierarchy_curves.csv`, `tuning_grid.csv`,
`eval_grid_sensitivity.csv`, `per_query.json`, or from
`reproduction_public_scale_ann_v1/` in this directory.

## Commands

```bash
# 1. Official ReferIt3D annotations (repo script, unchanged). gdown==6.1.0 pinned by the
#    script needs Python>=3.10, so gdown 5.2.0 (and usearch 2.23.0 for step 2) were
#    installed with `pip install --target <scratch>/pytools gdown==5.2.0 usearch`
#    (the numpy that pip pulled in was deleted so system numpy 1.26.4 is used).
PYTHONPATH=<scratch>/pytools python3 scripts/download_scannet_grounding_annotations.py \
  --out-dir external/datasets/scanrefer_public --acknowledge-source-terms

# 2. Reproduction of the existing experiment (script unchanged; only --out differs).
#    Also builds the BGE embedding cache (fastembed ONNX CPU, 1077 s encode).
PYTHONPATH=<scratch>/pytools python3 scripts/run_ieee_public_scale_ann.py \
  --out outputs/ieee_revision2/scale_hierarchy/reproduction_public_scale_ann_v1

# 3. New experiment (defaults: sizes 1000,10000,50000,100000,155536; 500 eval + 500 tuning
#    queries per pool; 10,000 bootstrap resamples, seed 20260901).
PYTHONPATH=<scratch>/pytools python3 scripts/run_ieee_scale_hierarchy.py
```

Logs: `logs/reproduction.log`, `logs/scale_hierarchy.log`.

## Data provenance

| File | Source | sha256 | Bytes |
|---|---|---|---|
| `external/datasets/scanrefer_public/nr3d.csv` | https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view (ReferIt3D, https://referit3d.github.io/) | `5de4f1b47130803c88f7c57903e7b6df5473f1b903c32cf28d06fa9c25996a67` | 10,700,336 |
| `external/datasets/scanrefer_public/sr3d/sr3d+.csv` | https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV (file id `1kcxscHp8yA_iKLT3BdTNghlY3SBfKAQI`) | `b6cd28ed18cbce716a3103198851d39a3dfb221d55eea66899d5f2c07a597d0d` | 28,887,195 |
| `external/datasets/scanrefer_public/referit3d_bge_small.npy` | built locally, BAAI/bge-small-en-v1.5 via fastembed (qdrant ONNX-q) | `74958b6f739808d927df706e713ead5f43a65bbaabeafb5ee27698982028aa79` | 155,535 x 384 float32 |

Both CSV checksums are identical to those recorded in
`outputs/ieee_revision/public_scale_ann_v1/scale_ann_summary.json`. The whole Sr3D folder
(also `sr3d.csv`, `sr3d_train.csv`, `sr3d_test.csv`; 80,645,015 bytes total) is listed with
checksums in `external/datasets/scanrefer_public/download_manifest.json`.
Raw rows: Nr3D 41,503, Sr3D+ 114,532. After deduplication: **155,535 records, 1,436 scenes,
16,650 objects, 76 instance types**, which matches the original run.

## Reproduction of `public_scale_ann_v1` (script unchanged)

All 6,264 per-query rows use the same query records as the original run. Scene-scoped
(oracle scan ID) numbers match exactly except hit@1 at 50k (0.622 vs 0.624). Flat and HNSW
differ by a few queries:

| Pool | Method | Q | hit@1 orig → rerun | hit@3 orig → rerun | median ms orig → rerun | records scored |
|---|---|---|---|---|---|---|
| 1,000 | flat_dense_exact | 88 | 0.159 → 0.159 | 0.273 → 0.273 | 0.161 → 0.022 | 1000 → 1000 |
| 1,000 | scene_scoped_dense_exact | 88 | 0.807 → 0.807 | 1.000 → 1.000 | 0.020 → 0.005 | 4.43 → 4.43 |
| 1,000 | hnsw_dense | 88 | 0.159 → 0.148 | 0.261 → 0.261 | 0.207 → 0.057 | — |
| 10,000 | flat_dense_exact | 500 | 0.070 → 0.072 | 0.156 → 0.156 | 1.053 → 0.299 | 10000 → 10000 |
| 10,000 | scene_scoped_dense_exact | 500 | 0.626 → 0.626 | 0.850 → 0.850 | 0.029 → 0.007 | 22.45 → 22.45 |
| 10,000 | hnsw_dense | 500 | 0.070 → 0.070 | 0.154 → 0.156 | 0.380 → 0.153 | — |
| 50,000 | flat_dense_exact | 500 | 0.108 → 0.108 | 0.204 → 0.204 | 3.907 → 1.581 | 50000 → 50000 |
| 50,000 | scene_scoped_dense_exact | 500 | 0.622 → 0.624 | 0.846 → 0.846 | 0.033 → 0.062 | 91.80 → 91.80 |
| 50,000 | hnsw_dense | 500 | 0.106 → 0.110 | 0.202 → 0.204 | 0.381 → 0.247 | — |
| 100,000 | flat_dense_exact | 500 | 0.090 → 0.088 | 0.200 → 0.198 | 7.757 → 4.889 | 100000 → 100000 |
| 100,000 | scene_scoped_dense_exact | 500 | 0.674 → 0.674 | 0.858 → 0.858 | 0.040 → 1.397 | 176.12 → 176.12 |
| 100,000 | hnsw_dense | 500 | 0.090 → 0.084 | 0.196 → 0.200 | 0.387 → 0.491 | — |
| 155,535 | flat_dense_exact | 500 | 0.138 → 0.134 | 0.242 → 0.236 | 12.253 → 4.948 | 155535 → 155535 |
| 155,535 | scene_scoped_dense_exact | 500 | 0.824 → 0.824 | 0.948 → 0.948 | 0.051 → 0.672 | 283.63 → 283.63 |
| 155,535 | hnsw_dense | 500 | 0.132 → 0.134 | 0.236 → 0.228 | 0.398 → 0.302 | — |

Why flat differs: the flat hit@3 value changes for 16 query/pool pairs and hit@1 for 14. In
**all 30** of them the score gap between the decisive ranks (rank 3/4 for hit@3, rank 1/2
for hit@1) is below 1e-6 (it prints as 0.0). These are exact score ties: many Sr3D+ template sentences occur
word for word for different objects. `_top_indices` breaks such ties arbitrarily through
`argpartition`. Exact ties at rank 3/4 are common: 190 of 500 queries at 155,535, 137 at 100k,
97 at 50k, 27 at 10k, 1 at 1k (gap < 1e-6; `near_tie_*` in `summary.json`). So the
original 0.242 vs rerun 0.236 is tie-order noise, not a data difference. HNSW differences
also reflect the multithreaded (non-deterministic) graph build. Latencies are from different
hardware (original: Windows; rerun: macOS arm64 with system load average about 20–40 on 14
cores, since other jobs were running). The rerun latencies are not comparable to the paper's
values and should not replace them.

## C4: how many queries are evaluated

The logic is in `_query_indices(records, size, 500)`. For each pool, a record is eligible if
its (scan_id, target_id) has at least 2 expressions inside that pool. Eligible records are
sorted by `sha256("20260901:query:" + record_key)` and the first `min(500, eligible)` are kept.

| Pool | Eligible | Evaluated (Q) | Tuning (this study) |
|---|---|---|---|
| 1,000 | 88 | **88** | 0 |
| 10,000 | 5,523 | 500 | 500 |
| 50,000 | 46,333 | 500 | 500 |
| 100,000 | 98,125 | 500 | 500 |
| 155,535 | 155,528 | 500 | 500 |

Total evaluated (query, pool) pairs: 88 + 4 x 500 = **2,088**. Only 1,444 of these are
distinct query records, because pools are nested and the same record can be sampled in
several pools. Each of the 2,088 pairs is run once per method (the original per_query.json
has 3 x 2,088 = 6,264 rows). "Q=500" is therefore correct for every pool except the 1k pool,
where Q=88.

## A3: Algorithm 1 over scene branches, no oracle scan ID

**Reimplementation** (`scripts/run_ieee_scale_hierarchy.py`). Branch = ScanNet scan ID (574,
1,251, 1,419, 1,430, 1,436 branches per pool). The gate is the adaptive-mass rule of
`recall_controlled._choose_branches`: branches are taken in descending proxy score, softmax
mass with temperature T is accumulated, and selection stops once mass ≥ tau or the cap B is
reached. The safety shortlist is the top-m records. C is the union of the selected branches
and the shortlist. The final top-3 is ranked by exact BGE cosine inside C. The confidence
fallback is off, which is the recall_controlled default (`confidence_margin_delta=0`). No
second (object) level was run. The three proxies are:

- **maxleaf**: branch = max exact cosine over its records (O(N)). Shortlist = exact global top-m.
- **summary**: branch = cosine to the normalized mean of its records (O(#branches)).
  Shortlist = HNSW top-m, using the same USearch parameters as `hnsw_dense`, search count 12.
  For leakage control, the query vector is subtracted from its own branch mean
  (leave-one-out). This uses only the query record's own membership, not the target.
- **hnswproxy** ("HNSW + scene hierarchy"): branch = max similarity among HNSW top-64 hits in
  that branch. Branches without a hit get zero mass. Shortlist = HNSW top-m.

**Tuning/eval split.** The grid is tau {0.5, 0.7, 0.9} x T {0.01, 0.02, 0.05, 0.1} x B
{1, 3, 5, 10} x m {0, 1, 3} = 144 configurations per proxy. Tuning queries are, per pool ≥10k,
the eligible records ranked after the first 500 in the same hash order. Any record that is an
evaluation query in any pool is excluded, and the first 500 are kept: 2,000 (query, pool)
pairs and 1,710 distinct records. Overlap with evaluation is 0 (asserted). The 1k pool has no
spare eligible records, so it gets no tuning queries and uses the frozen configuration. One
configuration per proxy and rule is selected on the pooled tuning set:

- `accuracy`: maximum tuning hit@3, ties broken by hit@1 and then by fewer leaf records.
- `budget`: fewest leaf records with tuning hit@3 ≥ flat hit@3 (0.1825). Only maxleaf
  qualifies, and its pick equals its `accuracy` pick.
- `gate_only` (m=0) and `budget_1pp` (hit@3 ≥ flat − 0.01) were **added after seeing the
  tuning-only results of a first run**, in which every accuracy pick used m=3. Evaluation
  results had not been viewed at that point.

Flat tuning hit@1/hit@3 = 0.092/0.1825.

**Canonical quality and timing.** hit@k for every method, flat included, comes from one exact
score vector per query with deterministic tie-breaking, so that ties do not create spurious
paired differences. The timed end-to-end runs give the latency (per-query minimum over 5
interleaved repetitions; the median first-run latency is also shown) and the record counts.
The timed paths reproduce the canonical hits exactly for all hierarchy methods. The unchanged
`_top_indices` flat path disagrees on 20/2,088 queries, all tie-order cases (flat timed-path
hit@3 at 155,535 is 0.236 vs canonical 0.234).

**Evaluation results.** Δ is the paired difference vs flat, from a 10,000-resample bootstrap
(seed 20260901), with 95% CI. "proxy recs" counts branch/record proxy comparisons. "HNSW dist"
is USearch `computed_distances`. "leaf recs" is |C|, which for maxleaf reuses proxy scores.

| Pool | Method | Config (tau,T,B,m) | hit@1 | hit@3 | Δhit@1 [95% CI] | Δhit@3 [95% CI] | proxy recs | HNSW dist | leaf recs | branches | median ms (min-of-5) | median ms (1st run) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | flat exact | — | 0.159 | 0.273 | — | — | 0.0 | 0.0 | 1000.0 | — | 0.029 | 0.047 |
| 1,000 | maxleaf_accuracy | 0.5, 0.01, 1, 3 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1000.0 | 0.0 | 4.8 | 1.00 | 0.063 | 0.117 |
| 1,000 | maxleaf_gate_only | 0.9, 0.02, 3, 0 | 0.159 | 0.261 | +0.000 [+0.000, +0.000] | -0.011 [-0.034, +0.000] | 1000.0 | 0.0 | 7.9 | 2.78 | 0.050 | 0.086 |
| 1,000 | summary_accuracy | 0.5, 0.01, 3, 3 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 574.0 | 457.2 | 4.9 | 1.38 | 0.118 | 0.247 |
| 1,000 | summary_gate_only | 0.9, 0.02, 10, 0 | 0.148 | 0.227 | -0.011 [-0.034, +0.000] | -0.045 [-0.091, -0.011] | 574.0 | 0.0 | 22.6 | 8.95 | 0.059 | 0.104 |
| 1,000 | summary_budget_1pp | 0.5, 0.01, 1, 3 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 574.0 | 457.2 | 4.1 | 1.00 | 0.115 | 0.181 |
| 1,000 | hnswproxy_accuracy | 0.7, 0.02, 3, 3 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 457.2 | 7.4 | 2.39 | 0.104 | 0.158 |
| 1,000 | hnswproxy_gate_only | 0.5, 0.05, 3, 0 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 457.2 | 8.7 | 3.00 | 0.100 | 0.143 |
| 1,000 | hnswproxy_budget_1pp | 0.5, 0.01, 1, 3 | 0.159 | 0.273 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 457.2 | 4.8 | 1.00 | 0.100 | 0.151 |
| 10,000 | flat exact | — | 0.074 | 0.158 | — | — | 0.0 | 0.0 | 10000.0 | — | 0.196 | 0.306 |
| 10,000 | maxleaf_accuracy | 0.5, 0.01, 1, 3 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 10000.0 | 0.0 | 20.5 | 1.00 | 0.263 | 0.420 |
| 10,000 | maxleaf_gate_only | 0.9, 0.02, 3, 0 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 10000.0 | 0.0 | 54.7 | 2.96 | 0.151 | 0.245 |
| 10,000 | summary_accuracy | 0.5, 0.01, 3, 3 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1251.0 | 645.4 | 18.2 | 2.10 | 0.214 | 0.463 |
| 10,000 | summary_gate_only | 0.9, 0.02, 10, 0 | 0.038 | 0.074 | -0.036 [-0.058, -0.014] | -0.084 [-0.116, -0.054] | 1251.0 | 0.0 | 88.7 | 9.94 | 0.089 | 0.140 |
| 10,000 | summary_budget_1pp | 0.5, 0.01, 1, 3 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1251.0 | 645.4 | 9.0 | 1.00 | 0.201 | 0.249 |
| 10,000 | hnswproxy_accuracy | 0.7, 0.02, 3, 3 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 645.4 | 51.3 | 2.77 | 0.185 | 0.220 |
| 10,000 | hnswproxy_gate_only | 0.5, 0.05, 3, 0 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 645.4 | 55.4 | 3.00 | 0.182 | 0.210 |
| 10,000 | hnswproxy_budget_1pp | 0.5, 0.01, 1, 3 | 0.074 | 0.158 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.0 | 645.4 | 20.4 | 1.00 | 0.178 | 0.193 |
| 50,000 | flat exact | — | 0.106 | 0.208 | — | — | 0.0 | 0.0 | 50000.0 | — | 1.413 | 2.762 |
| 50,000 | maxleaf_accuracy | 0.5, 0.01, 1, 3 | 0.106 | 0.208 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 50000.0 | 0.0 | 93.8 | 1.00 | 1.687 | 3.384 |
| 50,000 | maxleaf_gate_only | 0.9, 0.02, 3, 0 | 0.106 | 0.208 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 50000.0 | 0.0 | 272.4 | 2.99 | 1.110 | 2.835 |
| 50,000 | summary_accuracy | 0.5, 0.01, 3, 3 | 0.106 | 0.210 | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.006] | 1419.0 | 633.7 | 58.8 | 2.38 | 0.692 | 3.589 |
| 50,000 | summary_gate_only | 0.9, 0.02, 10, 0 | 0.074 | 0.120 | -0.032 [-0.060, -0.004] | -0.088 [-0.124, -0.052] | 1419.0 | 0.0 | 300.5 | 10.00 | 0.178 | 3.157 |
| 50,000 | summary_budget_1pp | 0.5, 0.01, 1, 3 | 0.106 | 0.210 | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.006] | 1419.0 | 633.7 | 20.7 | 1.00 | 0.289 | 1.729 |
| 50,000 | hnswproxy_accuracy | 0.7, 0.02, 3, 3 | 0.108 | 0.210 | +0.002 [+0.000, +0.006] | +0.002 [+0.000, +0.006] | 0.0 | 633.7 | 264.8 | 2.90 | 0.283 | 1.328 |
| 50,000 | hnswproxy_gate_only | 0.5, 0.05, 3, 0 | 0.108 | 0.210 | +0.002 [+0.000, +0.006] | +0.002 [+0.000, +0.006] | 0.0 | 633.7 | 272.8 | 3.00 | 0.283 | 1.895 |
| 50,000 | hnswproxy_budget_1pp | 0.5, 0.01, 1, 3 | 0.108 | 0.210 | +0.002 [+0.000, +0.006] | +0.002 [+0.000, +0.006] | 0.0 | 633.7 | 93.9 | 1.00 | 0.219 | 1.121 |
| 100,000 | flat exact | — | 0.090 | 0.196 | — | — | 0.0 | 0.0 | 100000.0 | — | 2.837 | 4.870 |
| 100,000 | maxleaf_accuracy | 0.5, 0.01, 1, 3 | 0.090 | 0.196 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 100000.0 | 0.0 | 176.2 | 1.00 | 3.184 | 5.357 |
| 100,000 | maxleaf_gate_only | 0.9, 0.02, 3, 0 | 0.090 | 0.196 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 100000.0 | 0.0 | 515.2 | 2.99 | 2.131 | 4.032 |
| 100,000 | summary_accuracy | 0.5, 0.01, 3, 3 | 0.090 | 0.198 | +0.000 [+0.000, +0.000] | +0.002 [-0.006, +0.010] | 1430.0 | 605.5 | 113.7 | 2.45 | 2.018 | 5.740 |
| 100,000 | summary_gate_only | 0.9, 0.02, 10, 0 | 0.078 | 0.130 | -0.012 [-0.038, +0.014] | -0.066 [-0.104, -0.028] | 1430.0 | 0.0 | 552.1 | 10.00 | 1.843 | 5.923 |
| 100,000 | summary_budget_1pp | 0.5, 0.01, 1, 3 | 0.090 | 0.198 | +0.000 [+0.000, +0.000] | +0.002 [-0.006, +0.010] | 1430.0 | 605.5 | 40.1 | 1.00 | 0.953 | 4.284 |
| 100,000 | hnswproxy_accuracy | 0.7, 0.02, 3, 3 | 0.088 | 0.192 | -0.002 [-0.006, +0.000] | -0.004 [-0.010, +0.000] | 0.0 | 605.5 | 508.9 | 2.94 | 0.684 | 3.036 |
| 100,000 | hnswproxy_gate_only | 0.5, 0.05, 3, 0 | 0.088 | 0.192 | -0.002 [-0.006, +0.000] | -0.004 [-0.010, +0.000] | 0.0 | 605.5 | 517.2 | 3.00 | 0.818 | 2.880 |
| 100,000 | hnswproxy_budget_1pp | 0.5, 0.01, 1, 3 | 0.088 | 0.192 | -0.002 [-0.006, +0.000] | -0.004 [-0.012, +0.004] | 0.0 | 605.5 | 175.4 | 1.00 | 0.599 | 2.942 |
| 155,535 | flat exact | — | 0.138 | 0.234 | — | — | 0.0 | 0.0 | 155535.0 | — | 3.591 | 4.613 |
| 155,535 | maxleaf_accuracy | 0.5, 0.01, 1, 3 | 0.138 | 0.234 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 155535.0 | 0.0 | 274.6 | 1.00 | 3.828 | 5.283 |
| 155,535 | maxleaf_gate_only | 0.9, 0.02, 3, 0 | 0.138 | 0.234 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 155535.0 | 0.0 | 806.3 | 2.99 | 2.226 | 3.575 |
| 155,535 | summary_accuracy | 0.5, 0.01, 3, 3 | 0.136 | 0.232 | -0.002 [-0.006, +0.000] | -0.002 [-0.012, +0.008] | 1436.0 | 583.7 | 167.2 | 2.49 | 0.435 | 1.944 |
| 155,535 | summary_gate_only | 0.9, 0.02, 10, 0 | 0.070 | 0.116 | -0.068 [-0.100, -0.038] | -0.118 [-0.160, -0.078] | 1436.0 | 0.0 | 807.1 | 10.00 | 0.158 | 1.288 |
| 155,535 | summary_budget_1pp | 0.5, 0.01, 1, 3 | 0.136 | 0.232 | -0.002 [-0.006, +0.000] | -0.002 [-0.012, +0.008] | 1436.0 | 583.7 | 56.9 | 1.00 | 0.193 | 0.823 |
| 155,535 | hnswproxy_accuracy | 0.7, 0.02, 3, 3 | 0.134 | 0.228 | -0.004 [-0.010, +0.000] | -0.006 [-0.014, +0.000] | 0.0 | 583.7 | 797.0 | 2.95 | 0.228 | 0.635 |
| 155,535 | hnswproxy_gate_only | 0.5, 0.05, 3, 0 | 0.134 | 0.228 | -0.004 [-0.010, +0.000] | -0.006 [-0.014, +0.000] | 0.0 | 583.7 | 807.2 | 3.00 | 0.239 | 0.661 |
| 155,535 | hnswproxy_budget_1pp | 0.5, 0.01, 1, 3 | 0.134 | 0.228 | -0.004 [-0.010, +0.000] | -0.006 [-0.016, +0.004] | 0.0 | 583.7 | 274.5 | 1.00 | 0.180 | 0.551 |

The query's own scene is inside C for only 3.0% (summary_budget_1pp) to 26.8%
(maxleaf_gate_only) of 155,535 evaluation queries (`query_scene_selected_rate`), compared
with 100% for the oracle scan-ID baseline (hit@3 0.948).

**Reading.**

1. Without the scan ID, the scene hierarchy does not recover the known-scene gain (0.948).
   At 155,535 records every frozen configuration except summary_gate_only is within 0.006
   hit@3 of flat exact (0.234), and their Δhit@3 CIs all include 0. summary_gate_only is
   significantly worse (−0.118 [−0.160, −0.078]).
2. With exact reranking inside C, a shortlist of m=3 taken from the exact scores makes
   maxleaf identical to flat by construction, because C contains the global top-3. The tuning
   rules picked m=3 or tied-with-flat configurations for this reason.
3. Even choosing the best of the 144 configurations **on the evaluation set**, which is
   optimistic and not a valid result (`eval_grid_sensitivity.csv`), no proxy beats flat at
   155,535: the best hit@3 is 0.234 for maxleaf and summary, and 0.230 for hnswproxy. The
   summary proxy with m=0 reaches only 0.116. The gains available from gating are small:
   +0.002 (maxleaf, 100k) and +0.004 (summary with m=3, 100k).
4. Cost: the summary and HNSW proxies reduce distinct vector comparisons from 155,535 to about
   2,073 (summary_budget_1pp: 1,436 branch + 583.7 HNSW + 56.9 leaf) or about 858
   (hnswproxy_budget_1pp), at unchanged-within-CI hit@3. That accuracy comes from the HNSW
   shortlist, not from the branch gate. maxleaf still costs O(N).

## Task 3: flat misses whose top-3 are all from other scenes

| Pool | Flat hit@3 misses | Top-3 all other scenes | Fraction |
|---|---|---|---|
| 1,000 | 64 | 61 | 0.953 |
| 10,000 | 421 | 410 | 0.974 |
| 50,000 | 396 | 381 | 0.962 |
| 100,000 | 402 | 386 | 0.960 |
| 155,535 | 383 | 371 | 0.969 |
| all 2,088 pairs | 1,666 | 1,609 | 0.966 |

(Canonical tie-breaking; flat misses of the evaluation queries.)

## Figure CSV

`scale_hierarchy_curves.csv` has one row per (method, pool) with `hit_at_1`, `hit_at_3`,
`median_latency_ms` (min-of-5), `mean_proxy_records`, `mean_hnsw_computed_distances`,
`mean_leaf_records`, `mean_scored_records` (distinct vector comparisons; for maxleaf this is
N because leaf scores are reused), and `config`. For a fourth curve in
`fig_public_scale_ann`, use `method == "hier_summary_budget_1pp"` (tau=0.5, T=0.01, B=1, m=3),
with `hit_at_3`, `median_latency_ms` and `mean_scored_records` for panels (a)-(c). Note that
its latency was measured on a different machine and under load, so do not plot it against the
paper's original latency values without rerunning all methods on the same machine. Flat
latency from this run is in the same CSV.

## Caveats

- Latency was measured on macOS arm64 (numpy 1.26.4, OpenBLAS) with load average 16.5 → 27.4
  over the run (other jobs were using the machine). Min-of-5 repeats a warm query, and the
  first-run medians are 1.3x (flat, 155,535) to about 18x (summary_gate_only, 50k) higher.
- The USearch HNSW build is multithreaded and not bit-reproducible. Rerunning can change
  summary/hnswproxy picks and hit@k by a few queries (observed between development runs).
- Flat hit@k at scale depends on tie order for 190/500 queries at 155,535 (exact duplicate
  template sentences). Canonical tie-breaking makes the paired comparisons fair but is not
  the original script's arbitrary order.
- The 1k pool has no tuning queries, and the object-level second hierarchy level was not run.
