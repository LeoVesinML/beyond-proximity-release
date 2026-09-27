# IEEE revision 2: predicted-map and oracle-map grounding audit

Every number below was computed by the commands in this file from saved artifacts. The machine-readable copy is `summary.json`.
Nothing under `papers/` or `/Users/leo/beyond proximity/revision/` was edited.

## Commands

```bash
cd "/Users/leo/beyond proximity/repo"
# B1, B2 (sanity, frame checks, corrected native ConceptGraphs) -> predicted_map_audit.json, replica_queries_corrected_gt_boxes.json
python3 -B scripts/audit_predicted_map_frames.py
# Reproduce the saved v3 oracle runs (configs/*.json point to the local ScanNet scenes in ../beyond-proximity-week6-person2)
for t in scannet_core scannet_extended replica; do
  python3 -B scripts/run_grounding_variants.py --config outputs/ieee_revision2/predicted_maps/configs/$t.json \
    --out outputs/ieee_revision2/predicted_maps/oracle_repro_v3/$t --run graph --run graph_fallback \
    --run flat_lexical --run flat_embedding --run ablation_no_relation
done
# Same-scorer oracle BGE, Algorithm 1 on oracle maps, B11 sweeps -> oracle_controls/oracle_controls_summary.json
python3 -B scripts/run_ieee_revision2_oracle_controls.py
# Corrected predicted-map BGE rerun: BLOCKED here (maps missing, see B2). Run this once the maps are restored:
python3 -B scripts/run_predicted_map_pair_corrected.py --scannet-root <dir with native_data> --replica-root <dir with native_data>
```

**Data availability.** The raw ConceptGraphs maps (`native_data/<scene>/exps/*/pcd_*.pkl.gz`) are not on this machine. `run_config.json` points to `C:\GitProjects\beyond-proximity`. The ScanNet benchmark `docs/benchmarks/scannet_bbq_grounding_pilot_v1.json` is gitignored and missing from this repo, so I used the local copy in `/Users/leo/beyond-proximity-week6-person2`. The saved predicted-map per-query rows (`outputs/ieee_revision/predicted_map_pair_v1/predicted_map_pair_per_query.json`) keep per-query IoUs and ranked object IDs but no boxes. Native ConceptGraphs saved one box per query: 37 unique ScanNet boxes and 45 unique Replica boxes.

## B1: the Replica denominator

In `scripts/run_predicted_map_pair.py`, `run()` takes every benchmark query in the map scenes, which is 56 for Replica. `_result_row` computes `best_bbox_iou_3d(...) or 0.0`. For the 8 negatives `expected_bboxes_3d == []`, so the function returns `None` and each negative is scored as a miss. `_summarize` then divides by `len(rows)`, which is 56. Native ConceptGraphs (`metrics_summary.json`) already used 48.

| Positives only (original GT) | n | Acc@0.25 | Acc@0.5 | top-3 IoU≥0.25 | mean top-1 IoU |
|---|---:|---:|---:|---:|---:|
| ScanNet flat BGE | 48 | 0.125 | 0.021 | 0.188 | 0.084 |
| ScanNet hierarchical BGE | 48 | 0.125 | 0.021 | 0.188 | 0.084 |
| ScanNet native ConceptGraphs | 48 | 0.0625 | 0.000 | – | 0.070 |
| Replica flat BGE | 48 | 0.0625 (3/48) | 0.000 | 0.083 | 0.028 |
| Replica hierarchical BGE | 48 | 0.0625 (3/48) | 0.000 | 0.083 | 0.028 |
| Replica native ConceptGraphs | 48 | 0.0625 | 0.042 | – | 0.062 |

For ScanNet, all 48 queries are positive, so its numbers do not change. Each Replica value reported over 56 (0.054 / 0.000 / 0.071) becomes 3/48 = 0.0625, 0.000 and 4/48 = 0.083.

## B2: sanity test and frame checks

**Sanity test.** I fed the GT boxes in as predictions, both directly through `best_bbox_iou_3d` and through `run_predicted_map_pair._result_row`. The input was every GT object of the scene, with a one-hot similarity on the target, for both the flat and the hierarchical method. Acc@0.25 = Acc@0.5 = 1.000 (48/48) on ScanNet and on Replica. The IoU code and the min/max versus center/size handling are correct.

**Frame checks.** "Known predicted boxes" are the saved native ConceptGraphs boxes. Recall over them is a lower bound on the recall ceiling.

| Dataset / transform | median best IoU of pred vs any GT | GT target → nearest pred center, median (any label / same label) | targets with a pred IoU≥0.25 (LB) | native CG Acc@0.25 / Acc@0.5 / mIoU |
|---|---:|---:|---:|---:|
| ScanNet identity (as evaluated) | **0.241** | **0.65 m / 1.18 m** | **0.125** | 0.0625 / 0.000 / 0.070 |
| ScanNet pred ← axisAlignment·pred | 0.000 | 3.37 / 4.84 | 0.000 | 0.000 / 0.000 / 0.001 |
| ScanNet pred ← inv(axisAlignment)·pred | 0.000 | 3.28 / 3.88 | 0.042 | 0.000 / 0.000 / 0.001 |
| ScanNet pred Y-up→Z-up | 0.009 | 2.03 / 2.27 | 0.000 | 0.000 / 0.000 / 0.007 |
| Replica GT as used (abb, orientation dropped) | 0.065 | 1.13 / 1.56 | 0.083 | 0.0625 / 0.042 / 0.062 |
| **Replica GT R·abb+t (orientation applied)** | **0.407** | **0.54 / 0.36** | **0.292** | **0.271 / 0.146 / 0.191** |
| Replica GT Rᵀ·abb+t | 0.036 | 1.32 / 2.19 | 0.083 | 0.0625 / 0.042 / 0.052 |
| Replica pred Y-up→Z-up / Z-up→Y-up | 0.000 / 0.000 | 1.69 / 1.93 | 0.000 / 0.000 | 0.000 / 0.000 |

Units are metres on both sides. The median cube-root box volume is 0.87 m for predictions and 0.89 m for matched GT (ScanNet), and 0.69 m versus 0.69 m (Replica, corrected). Details are in `predicted_map_audit.json` → `B2b_frame_checks`.

**ScanNet: no frame bug.** `prepare_scannet_scene.py` applies `axisAlignment` both to the mesh vertices (GT) and to the camera poses fed to ConceptGraphs (`relative_pose=False`). Identity is the best of the four transforms by a wide margin. ScanNet's low accuracy therefore comes from map quality and coverage, not geometry: the maps were built from only 3–7 RGB-D frames per scene (39 in total), and at most 12.5% of targets have any saved predicted box at IoU≥0.25.

**Replica: real bug in the GT boxes.** `scripts/import_replica_semantic_mesh.py::boxes_from_info_semantic` uses `oriented_bbox.abb` center and sizes as a world AABB. That box is in the object's local frame, and the importer drops `orientation.rotation`. The rotation is non-trivial: median angle 12–76° per scene, and more than 5° for 68–100% of objects. The translation is 0 for all 575 objects. With world = R·abb + t (quaternion stored as [x,y,z,w]), the predicted-box/GT agreement jumps (table above). Rᵀ and the axis swaps do not help, which rules out a Y-up/Z-up or mesh-versus-pose mismatch: the NICE-SLAM poses and the habitat mesh share the frame once the orientation is applied. This explains the 0.0 Acc@0.5 and the ~0.06–0.07 mean IoU on Replica.

The corrected queries are in `replica_queries_corrected_gt_boxes.json` (AABB of the rotated box, so slightly loose for rotated objects; the mesh is not on disk).

**Corrected numbers (Replica, n = 48 positives)**
- Native ConceptGraphs: Acc@0.25 **0.271** (13/48, Wilson 0.166–0.410), Acc@0.5 **0.146** (7/48), mean IoU 0.191. The original GT gave 0.0625 / 0.042 / 0.062.
- Flat and hierarchical BGE: **blocked**. The maps are missing, and only 8/48 top-1 objects have a saved box. On that non-representative subset, Acc@0.25 goes from 0/8 to 4/8 with the corrected GT, identically for flat and hierarchical. `scripts/run_predicted_map_pair_corrected.py` (positives only, corrected GT) is ready; a smoke test on a synthetic GT-box map ran end to end, and no number from it is reported.
- ScanNet: unchanged (no bug).
- Also affected: the Replica oracle maps use the same unrotated boxes. Their Acc@k is self-consistent (the predicted box is the GT box), but "near" geometry on Replica uses the wrong centers. The saved v3 relation audit reports that relations change no Replica top-1.

**What the "hierarchical BGE" on predicted maps is.** `run_predicted_map_pair.py` does its own traversal:
- a recursive spatial median split with at most 12 objects per leaf;
- branch score = max BGE leaf proxy;
- softmax T = 0.1, cumulative mass 0.7, cap 5;
- plus the global top-1 (m = 1);
- keep k = 3 by the same BGE score.

This follows Algorithm 1's structure but is not `backend/query/recall_controlled.py` code, and it uses τ = 0.7 rather than the paper's BGE setting (0.5, 0.1, 5, 1). The leaf scorer equals the proxy and the flat top-1 is always in C, so top-1 is identical to flat by construction (104/104 rows). Flat and hierarchical Acc@k can never differ.

## Same-scorer oracle comparison (48 positives each)

These runs use the predicted-map scorer (BGE-small, full query versus object text, argmax) on the oracle maps.

| Oracle map | object text | flat BGE Acc@0.25 | Acc@0.5 | R@1 | R@3 |
|---|---|---:|---:|---:|---:|
| ScanNet | GT label (analogue of class_name) | **0.354** (17/48) | 0.333 | 0.333 | 0.854 |
| ScanNet | full candidate text | 0.250 | 0.250 | 0.250 | 0.667 |
| Replica | GT label | **1.000** | 1.000 | 1.000 | 1.000 |
| Replica | full candidate text | 0.938 | 0.938 | 0.938 | 1.000 |

The hierarchical variant gives the same Acc@0.25 in every row. For comparison, the saved v3 `flat_embedding` (parsed target, relation rerank) reproduces at 0.3125 (ScanNet) and 0.979 (Replica). With one scorer, oracle versus predicted is 0.354 vs 0.125 on ScanNet, and 1.000 vs 0.0625 on Replica with the original GT (corrected BGE value blocked).

## B8: the oracle "graph" rows and Algorithm 1

All v3 oracle rows reproduce exactly (`oracle_repro_v3/*/variant_comparison.csv`). ScanNet graph gives 0.604 / 0.8125 / 4.729 objects / 219.3 tokens; flat lexical 49.0 objects; Replica graph 12.107 objects; ScanNet-extended graph 4.375 objects.

**How the "graph" rows select candidates.** They come from `backend/query/grounding.py::graph_pruned_candidate_ids`, not Algorithm 1.
- The oracle trees are flat: root → one node per GT object, plus one region node.
- So each "branch" is a single object node. Its score is |target tokens ∩ node tokens| / |target tokens|.
- Nodes scoring at least 0.5 × best are kept. If best = 0, only the lexicographically first child is kept.
- The kept set is then lexically scored and relation-reranked.

4.73 is therefore the mean size of a token-overlap filter on object names. Per-query sizes on ScanNet are 1 (9 queries), 2 (17), 3 (8), 4 (5), 5 (4), 9 (2), 17 (1), 28 (1) and 57 (1). On Replica, 12.11 is inflated by queries that match the region node ("official Replica ..."), which pulls in every object.

**Algorithm 1 rerun.** Settings: spatial median split with at most 12 objects per leaf; lexical leaf proxy; max-leaf branch score; T = 0.25, τ = 0.7, B = 5; m = 1; k = 3. The shared v3 scorer (lexical + gated relations) is applied to C. The cached proxy is computed for all objects: 49.0 per query on ScanNet, 71.9 on Replica.

| Track | R@1 | R@3 | Acc@0.25 | objects/query (\|C\|) | est. tokens/query |
|---|---:|---:|---:|---:|---:|
| ScanNet core, Algorithm 1 | 0.604 | 0.8125 | 0.604 (29/48) | 21.29 | 927.6 |
| ScanNet core, v3 graph | 0.604 | 0.8125 | 0.604 | 4.73 | 219.3 |
| ScanNet core, flat lexical | 0.625 | 0.833 | 0.604 | 49.00 | 2116.8 |
| ScanNet extended, Algorithm 1 | 0.589 | 0.768 | 0.589 (33/56) | 24.23 | 1053.0 |
| Replica, Algorithm 1 | 1.000 | 1.000 | 1.000 (48/48) | 30.27 | 748.0 |
| Replica, v3 graph | 1.000 | 1.000 | 1.000 | 12.11 | 302.7 |

Tokens use the same chars/4 estimate as v3, over the scored universe C (not the k = 3 kept objects).

## B11: constant sweeps (ScanNet core, n = 48, Acc@0.25)

| Setting | graph | graph+fallback | flat lexical |
|---|---:|---:|---:|
| relation boost 0.20 (code default), threshold 0.45 | 0.604 | 0.604 | 0.604 |
| relation boost 0.35 | 0.604 | 0.604 | 0.604 |
| relation boost 0.50 | 0.604 | 0.604 | 0.604 |
| threshold 0.35, boost 0.20 | 0.604 | 0.604 | 0.604 |
| threshold 0.45 (default) | 0.604 | 0.604 | 0.604 |
| threshold 0.55 | 0.604 | 0.604 | 0.604 |

R@1 and objects per query were also unchanged (graph 0.604 / 4.73; fallback 0.625 / 36.75; flat 0.625 / 49.0).

The sweep is flat for mechanical reasons, not because the method is robust:
- **The boost can never add anything.** All 14 relation-applied top-1 objects have a pre-relation score of 1.0, the `min(1.0, …)` cap. The relation effect comes only from the sort-key tie-break (relation_applied, relation_confidence).
- **The thresholds split the same way.** Graph top-1 scores are either at most 0.375 or at least 0.75, so 0.45 and 0.55 split queries identically. At 0.35, three queries scoring 0.375 are no longer low-confidence, but they still fall back because the relation is unresolved.

The code default is 0.20. The paper text says "adds 0.35, subtracts 0.35×0.35", which is the superseded v1 scorer (removed in 57e8b23).

## B4: were the 48 ScanNet core queries used to develop v3?

Yes; the v3 ScanNet relation numbers are in-sample.
- The benchmark builder was added in `4cc58e8` (2026-07-12).
- The v1 scorer was added in `ae9f120` (2026-07-13) and evaluated on the same 48 queries (`outputs/public_datasets/scannet_pilot_v1`, R@1 0.271 versus no-relation 0.292). `docs/experiments/public_datasets/person2_grounding_results.md` records that the relation reranker did not improve the pilot.
- The v3 rewrite of `backend/query/grounding.py` came in `57e8b23` (2026-07-24): one bulk commit, with its audit `phase6_7_v3/scannet_relation_audit.json` run on `scannet_bbq_grounding_pilot_v1.json`.
- The papers justify v3 with the v1 failure on this pilot (for example `papers/ieee-access/main.tex:621`). `papers/twinworld/main.tex:204-207` says the constants were fixed "during development".
- The extended set (`build_scannet_extended_queries.py:205`) contains the same 48 plus 8 functional and 8 negative queries, and none of the new queries are relational.
- The 50-dev / 75-test split in the IEEE revision applies to the captured-scene view benchmark, not to ScanNet.
- No document says in so many words "tuned on these queries".

## Negative accuracy (Replica 8, ScanNet-extended 8)

`scripts/evaluate_grounding.py`: a query is negative-eligible when `expected_output_type == "not_found"` and it has verified-absence status. It counts as correct iff `result.found is False`. `run_grounding_variants.py` sets `found = (search.status == "ok")`, and status is "ok" iff the top score is above 0.

So the lexical variants abstain only when no object has any label, text, synonym or affordance token overlap with the target. There is no calibrated abstention threshold: 0.45 only triggers fallback. The 8 Replica negatives are the same query, "Find the fire hydrant", in 8 scenes. The ScanNet-extended negatives are "Find the bathtub/bed/refrigerator." Every lexical variant scores 0 on them, giving 1.000. `flat_embedding` never abstains because its score is (cos+1)/2 > 0, giving 0.000. Native ConceptGraphs also always returns an object, giving 0.000.

## Evidence files
- `outputs/ieee_revision2/predicted_maps/predicted_map_audit.json`: B1, B2 sanity, frame checks, corrected evaluation.
- `outputs/ieee_revision2/predicted_maps/replica_queries_corrected_gt_boxes.json`: the corrected Replica GT.
- `outputs/ieee_revision2/predicted_maps/oracle_repro_v3/{scannet_core,scannet_extended,replica}/`: the v3 reproduction.
- `outputs/ieee_revision2/predicted_maps/oracle_controls/oracle_controls_summary.json`, plus `algorithm1/` and `b11/` per-query results.
- Scripts: `scripts/audit_predicted_map_frames.py`, `scripts/run_ieee_revision2_oracle_controls.py`, `scripts/run_predicted_map_pair_corrected.py`.
