# IEEE revision 2: hierarchy-construction analyses

Run date: 2026-09-27. Branch `ieee`. All numbers below come from files in this directory, written by one script:

```bash
python3 -B scripts/run_rev2_hierarchy_analyses.py            # defaults: --out outputs/ieee_revision2/hierarchy --threads 6
```

Inputs: `docs/benchmarks/benchmark_queries_v2.json` (150 five-scene queries, 125 positives), `docs/benchmarks/manual_zone_gt_v2.json`,
`outputs/ieee_revision/recall_frontier_v1/split.json` (dev 50 / test 75), `outputs/raw_rgbd_hierarchy/five_scene_clip_v1/`,
`outputs/ieee_revision/hierarchy_scorer_grid_v1/`. CLIP `openai/clip-vit-base-patch32` (snapshot 3d74acf9…, downloaded for this run, CPU),
BGE `BAAI/bge-small-en-v1.5` (fastembed ONNX CPU). The bootstrap is `backend.query.recall_controlled.paired_bootstrap_difference`
(10,000 resamples, seed 20260901, percentile 95% CI).

Reproduction checks: the recomputed CLIP features reproduce the saved raw RGB-D clusters exactly in all five scenes
(`construction_audit.json: raw_default_reproduces_saved_clusters`). All 36 saved grid cells are reproduced exactly
(`grid_reproduction_check.json: all_cells_match = true`). The saved E3 means are also reproduced (`e3_two_branch_ci.json: recomputed_gate_means`).

---

## T1 Vocabulary and labels (`label_audit.json`)

The code has one fixed vocabulary of 20 labels, `CLUSTER_LABELS` in `scripts/run_raw_rgbd_hierarchy_benchmark.py`, used with the prompt `"a photograph of a {label}"`:
auditorium, banquet hall, conference room, corridor, entrance, exhibition gallery, facade, garden, information desk, lobby, lounge,
museum display, outdoor plaza, reception desk, service counter, stage, street, theater seating, walkway, wayfinding area.

- **Outdoor categories are present.** Five are explicitly outdoor: facade, garden, outdoor plaza, street, walkway. Entrance and wayfinding area could be either.
- **Labels do not affect retrieval.** Each label is the argmax of the text embeddings against the cluster's CLIP centroid, and it is computed after k-means (`construct_raw_hierarchy`), so it cannot change which views are in a cluster. In E3, `rank_method`/`cluster_centroids` read only `cluster["view_ids"]`. In the grid, `_zones_by_scene` copies `predicted_label` into `zone_name`, but `_matched_rank` reads only `view_ids` and uses `zone_id` for tie-breaks. Max-leaf Algorithm 1 (T6) reads only branch membership. The category names are branch labels only.
- **outdoor-street labels** (7 clusters, sizes 4/3/1/1/6/3/3): facade, wayfinding area, street, garden, walkway, entrance, facade.
- **outdoor-drone labels** (2 clusters, sizes 6/10): **stage** and **stage**. Scores are 0.2225 and 0.2249; the runners-up are wayfinding area and lobby. The outdoor drone scene gets an indoor label.

## T2 Constructors, inputs and wall time (`construction_audit.json`)

**Confirmed (code: `backend/geometry/spatial_graph.py`).**
- **Pose-only** = connected components of the camera-position graph. Views are linked when their distance is ≤ t = alpha · median(nearest-neighbour camera distance), with alpha = `SPATIAL_SCALE_FACTOR` = 3.5 (`backend/config.py`).
- **Pose+semantic** = the same components, then merged with union-find. Two components merge if (Rule 1) any facing value in one and any in the other form a complementary pair ({stage, audience}, {entrance, exit}, {bar, reception}), or (Rule 2) they share a room_type and at least one visible landmark.
- **Correction to the task description:** the rules compare the union of values over each component, not a "dominant" value as the docstring says. Merges are transitive.

**Inputs actually opened during construction.** These were measured by wrapping `open` (unique files are shown; open-call counts are in the JSON):

| Constructor | transforms.json | authored ViewJSON (views/*.json) | RGB images | depth arrays |
|---|---:|---:|---:|---:|
| pose-only | 5 | 0 | 0 | 0 |
| pose+semantic | 5 | **97** (room_type, facing, visible_landmarks via `list_view_summaries`) | 0 | 0 |
| raw RGB-D | 5 | 0 | 97 | 97 |

Pose-only reads poses only. Pose+semantic depends on the authored ViewJSON records. In the grid and reviewer scripts, `_cluster_records` also reads the summaries to name pose-only zones, but the names are unused for retrieval.

**Wall time** (CPU, 5 scenes, 97 views, torch 6 threads; the machine was shared with another agent's MPS job):

| Constructor | median total | min–max | repeats |
|---|---:|---:|---:|
| pose-only | 0.00078 s | 0.00073–0.00193 s | 20 |
| pose+semantic | 0.00634 s | 0.00557–0.0238 s | 20 |
| raw RGB-D (read RGB/depth + CLIP image encoding + k-means/silhouette + labels) | 17.0 s | 11.95–22.73 s | 3 |

The raw RGB-D time excludes a one-time CLIP load (3.54 s in this run). The saved 2026-07-24 run logged 14.15 s.

Structural facts:
- Pose-only yields **one cluster** in Museume and Theater, sizes [19,1] in ConferenceHall, [10,10,1] in street, and [10,4,2] in drone.
- Pose+semantic yields one cluster in the three indoor scenes.
- **The raw constructor does not enforce ≥3 views per cluster.** Only K ≤ floor(n/3) bounds the average, and the code comment claiming a minimum is wrong. The saved raw hierarchy has 9 singleton clusters (3/2/2/2/0 per scene).

## T3 Why lexical hit@1 varies but BGE/CLIP hit@1 does not (`hit1_invariance_audit.json`)

**How the gate ranks.** `_matched_rank` (in `scripts/run_ieee_hierarchy_scorer_grid.py`):
1. It orders branches by **max leaf score**.
2. It adds whole branches until they hold ≥ k views.
3. It ranks the union by the **same global leaf order**.

The first branch contains the flat argmax, so the rank-1 view equals flat rank-1 whenever the argmax is unique and covered by a branch. There is no separate branch score.

**Lexical ties.** The grid's lexical score is `overlap + 1e-3·overlap/|q|`, a monotone function of an integer overlap, so views tie at the maximum. Tied branches are ordered by `zone_id` and tied views by `view_id`, both descending. The rank-1 view therefore depends on which tied branch is expanded first.

**Evidence over 125 positives × 4 hierarchies × 3 budgets = 1,500 cells per scorer:**

| Scorer | queries with tied top score | cells whose rank-1 = flat rank-1 | queries whose hit@1 differs across cells | of those, with a tie |
|---|---:|---:|---:|---:|
| lexical | 82 | 1369 | 25 | 25 |
| BGE | 0 | 1497 | 0 | 0 |
| CLIP | 0 | 1491 | 0 | 0 |

The 3 BGE and 9 CLIP exceptions all involve Theater `v008`. It is the flat argmax for one BGE query (qv2_067) and three CLIP queries (qv2_063, 070, 071), and it belongs to no manual zone (the only uncovered view). The manual gate therefore returns a different rank-1 view, but v008 is not a target in any of those queries, so hit@1 is unchanged.

## T4 E3 fixed two-branch CLIP gate (`e3_two_branch_ci.json`, n = 125)

Means: flat hit@1/hit@3 0.432/0.624, manual 0.424/0.592, raw RGB-D 0.368/0.528.

| Paired difference | estimate | 95% CI |
|---|---:|---:|
| raw − manual, hit@1 | −0.056 | [−0.120, 0.000] |
| raw − manual, hit@3 | −0.064 | [−0.136, 0.008] |
| raw − flat, hit@1 | −0.064 | [−0.112, −0.016] |
| raw − flat, hit@3 | −0.096 | [−0.168, −0.032] |

Queries whose target lies outside the two best-ranked branches:

| Hierarchy | queries (of 125) | mean views checked |
|---|---:|---:|
| manual | 21 | 10.54 |
| raw RGB-D | 29 | 10.15 |
| pose-only (recomputed, same gate) | 2 | 18.67 |
| pose+semantic (recomputed, same gate) | 2 | 18.87 |
| flat | 0 | 19.40 |

Pose-only and pose+semantic hit@1/hit@3 are 0.432/0.624, identical to flat, because their two top branches cover almost every view. The saved 10.76/10.00 views were averaged over all 150 queries; the values above are over the 125 positives.

## T5 Grid paired CIs vs manual zones (`grid_paired_ci.json`, n = 125, seed 20260901)

| Cell | k | hit@1 (var / manual), Δ [95% CI] | hit@3 (var / manual), Δ [95% CI] |
|---|---:|---|---|
| BGE raw RGB-D | 3 | 0.680/0.680, 0.000 [0.000, 0.000] | 0.848/0.880, −0.032 [−0.088, 0.024] |
| BGE raw RGB-D | 6 | 0.680/0.680, 0.000 [0.000, 0.000] | 0.864/0.888, −0.024 [−0.072, 0.024] |
| BGE raw RGB-D | 12 | 0.680/0.680, 0.000 [0.000, 0.000] | 0.920/0.920, 0.000 [−0.032, 0.032] |
| lexical pose-only | 3 | 0.680/0.712, −0.032 [−0.088, 0.024] | 0.888/0.864, +0.024 [−0.024, 0.080] |
| lexical pose-only | 6 | 0.680/0.696, −0.016 [−0.064, 0.032] | 0.888/0.856, +0.032 [−0.016, 0.080] |
| lexical pose-only | 12 | 0.696/0.672, +0.024 [−0.016, 0.064] | 0.928/0.880, +0.048 [0.016, 0.088] |

The saved grid used seed 20260902. Its hit@3 CIs are identical except lexical pose-only k=3, where the upper bound is 0.072 instead of 0.080.

## T6 Hierarchy × scorer grid with Algorithm 1 (`alg1_grid_aggregate.csv`, `alg1_grid_summary.json`, `alg1_grid_per_query.json`)

**Setup.**
- Code: lexical uses `backend/query/recall_controlled.search_prepared_scene` unchanged; BGE and CLIP use the dense twin `scripts/run_ieee_bge_frontier._search`.
- Branches = hierarchy zones. selection = adaptive_mass, branch score = **max_leaf_proxy** for all scorers, max branches B = 5, safety shortlist m = 1, no flat fallback, candidate budget k ∈ {3, 6, 12}.

**Configs.**
- lexical: tau 0.7, T 0.25.
- BGE: tau 0.5, T 0.1. The paper's BGE operating point used `node_text` branch scoring; max-leaf is used here so that label-free hierarchies are comparable.
- **CLIP: tau 0.5, T 0.01.** T = 1/exp(logit_scale), CLIP's own contrastive temperature; the measured logit scale is 100.0. It was not tuned on any split.

**Results.** n = 125 positives. Each hierarchy has one hit@1/hit@3 value for every k: hit@3 depends only on the top three candidates, and those do not depend on k ≥ 3 once the branch set is fixed. The views-checked columns are k = 3 / 6 / 12.

| Scorer | Hierarchy | hit@1 | hit@3 | views checked (k=3/6/12) | hit@3 Δ vs manual [95% CI] |
|---|---|---:|---:|---|---|
| lexical | flat | 0.696 | 0.904 | 19.40 | |
| lexical | manual | 0.696 | 0.904 | 2.94/5.57/9.52 | — |
| lexical | pose-only | 0.696 | 0.904 | 2.98/5.90/11.21 | 0.000 [0, 0] |
| lexical | pose+semantic | 0.696 | 0.904 | 2.98/5.90/11.44 | 0.000 [0, 0] |
| lexical | raw RGB-D | 0.696 | 0.904 | 2.78/5.03/8.28 | 0.000 [0, 0] |
| BGE | flat | 0.680 | 0.928 | 19.40 | |
| BGE | manual | 0.680 | 0.920 | 3.00/5.92/10.70 | — |
| BGE | pose-only | 0.680 | 0.920 | 2.97/5.92/11.32 | 0.000 [−0.024, 0.024] |
| BGE | pose+semantic | 0.680 | 0.920 | 2.97/5.92/11.50 | 0.000 [−0.024, 0.024] |
| BGE | raw RGB-D | 0.680 | 0.904 | 2.97/5.88/10.23 | −0.016 [−0.048, 0.016] |
| CLIP | flat | 0.432 | 0.624 | 19.40 | |
| CLIP | manual | 0.432 | 0.600 | 3.00/5.74/9.46 | — |
| CLIP | pose-only | 0.432 | 0.624 | 2.98/5.96/11.18 | +0.024 [−0.024, 0.080] |
| CLIP | pose+semantic | 0.432 | 0.624 | 3.00/6.00/11.65 | +0.024 [−0.024, 0.072] |
| CLIP | raw RGB-D | 0.432 | 0.592 | 2.78/4.94/6.52 | −0.008 [−0.072, 0.056] |

- **hit@1:** every non-manual hit@1 Δ is 0.000 [0.000, 0.000].
- **Branch coverage:** mean selected-branch ratio (manual / pose-only / pose+semantic / raw) is 0.45 / 0.72 / 0.81 / 0.38 (lexical), 0.45 / 0.72 / 0.81 / 0.41 (BGE), and 0.36 / 0.64 / 0.77 / 0.28 (CLIP).
- **Target-in-candidates rates** and dev/test hit@3 are in the CSV.

With the safety shortlist and max-leaf branch scoring, no hierarchy's quality differs measurably from manual zones under any scorer.

## T7 Construction-constant sensitivity (`sensitivity.csv`, `sensitivity.json`)

**Protocol.** CLIP scorer, budget-filling gate (`_matched_rank`), k = 3. One constant varies at a time; all others stay at their defaults. "Dev" = the 50 frozen dev positives; "all" = the 125 positives. F1 = macro pairwise F1 vs manual zones. hit@1 is 0.480 (dev) and 0.432 (all) in every row, as T3 explains.

References: flat hit@3 dev/all 0.680/0.624; manual 0.560/0.552.

| Hierarchy | Constant | Value | K per scene | singletons | F1 | hit@3 dev | hit@3 all |
|---|---|---|---|---:|---:|---:|---:|
| raw | w_p | 0 | 6,6,6,4,2 | 8 | 0.497 | 0.60 | 0.592 |
| raw | w_p | 0.25 / **0.5** / 1 | 6,6,6,7,2 | 9 | 0.487 | 0.64 | 0.592 |
| raw | w_d | 0 | 6,6,6,7,2 | 9 | 0.499 | 0.62 | 0.608 |
| raw | w_d | 0.1 / **0.2** / 0.5 | 6,6,6,7,2 | 9 | 0.487 | 0.64 | 0.592 |
| raw | min views | **none (shipped)** | 6,6,6,7,2 | 9 | 0.487 | 0.64 | 0.592 |
| raw | min views | 2 | 2,4,1,2,2 | 0 | 0.545 | 0.72 | 0.616 |
| raw | min views | 3 | 2,2,1,2,2 | 0 | 0.525 | 0.66 | 0.600 |
| raw | min views | 4 | 1,2,1,2,2 | 0 | 0.538 | 0.68 | 0.632 |
| raw | K_max | floor(n/4) | 4,5,2,5,2 | 5 | 0.496 | 0.68 | 0.608 |
| raw | K_max | **floor(n/3)** | 6,6,6,7,2 | 9 | 0.487 | 0.64 | 0.592 |
| raw | K_max | floor(n/2) | 9,10,10,10,8 | 22 | 0.314 | 0.60 | 0.568 |
| pose | alpha | 2.0 | 5,2,5,7,8 | 11 | 0.475 | 0.64 | 0.584 |
| pose | alpha | 2.5 | 4,1,4,5,7 | 8 | 0.508 | 0.60 | 0.576 |
| pose | alpha | 3.0 | 3,1,3,5,4 | 4 | 0.560 | 0.64 | 0.616 |
| pose | alpha | **3.5** | 2,1,1,3,3 | 2 | 0.588 | 0.66 | 0.624 |
| pose | alpha | 4.0 | 2,1,1,2,3 | 2 | 0.573 | 0.68 | 0.624 |
| pose | alpha | 5.0 | 1,1,1,2,2 | 1 | 0.600 | 0.68 | 0.624 |

**Key finding (code, `run_raw_rgbd_hierarchy_benchmark.py` lines 366–367).** `combined = standardize(concat(f, 0.5·std(p), 0.2·std(d)))` re-standardizes every column. Any non-zero w_p or w_d is therefore cancelled, and only zero versus non-zero matters. Any non-zero value gives the same features, which is why those rows are identical. After this step, pose contributes 3 of 521 unit-variance dimensions and depth 6.

**min views.** The shipped code enforces no minimum, so a min-views sensitivity requires a new rule. The rule used here: among K = 2..K_max, admit only K whose k-means solution has every cluster ≥ m; if none qualifies, use K = 1.

**Larger alpha.** The pose-only rows for alpha ≥ 3.5 approach flat retrieval, because they collapse to one or a few clusters.

**Origin of the constants.**
- `SPATIAL_SCALE_FACTOR = 3.5` exists since the initial commit e49ffc0 (2026-04-08) with only the comment "× median nearest-neighbour distance". It first appears in `docs/archive/deprecated/superpowers_plans/2026-04-05-semantic-3dgs-navigator-plan.md`, with no justification.
- The weights 0.5/0.2 and K ≤ n/3 were introduced in 57e8b23 (2026-07-24, "Complete TwinWorld agent evaluation and paper") with no recorded rationale or sweep.
- 612ae79 (2026-09-03) removed a `min(8, ·)` cap on K. This has no effect, because n//3 ≤ 7 in every scene.
- `docs/IEEE_ACCESS_REVISION_PLAN.md` item W11 lists these constants as unjustified.
- No tuning evidence was found in git history or docs.

## T8 Structural agreement figure data (`structural_agreement.csv`, `structural_agreement.json`)

The metric is macro pairwise F1 vs manual zones, from `_zone_assignment_metrics`, averaged over the 5 scenes. The random baseline is a cluster-size-matched random partition: 1,000 trials per scene with the seeding of `run_matched_hierarchy_controls.py`, macro taken per trial. The 95% range is the 2.5–97.5 percentiles. The Cursor/MCP point is excluded.

| Hierarchy | macro F1 | random mean | random 95% range | P(random ≥ observed) |
|---|---:|---:|---:|---:|
| pose-only | 0.588 | 0.541 | [0.523, 0.566] | 0.001 |
| pose+semantic | 0.570 | 0.563 | [0.554, 0.571] | 0.038 |
| raw RGB-D | 0.487 | 0.366 | [0.324, 0.414] | 0.000 |
| best disjoint projection of manual zones | 0.877 | — | — | — |

- **Cross-check.** The per-scene F1 values and the per-scene random means for pose-only and pose+semantic equal `outputs/reviewer_revision/matched_hierarchy_v1/structure.csv` and `random_baseline.csv` exactly. The raw RGB-D F1 of 0.487 matches `metrics_summary.json`. The reviewer files had no raw RGB-D random baseline; it is new here.
- **Caveat.** Pose-only and pose+semantic use a single cluster in two or three indoor scenes. A one-cluster partition gets F1 equal to its random baseline, so their high F1 largely reflects coarse, near-trivial partitions.
