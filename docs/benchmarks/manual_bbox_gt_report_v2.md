# Manual BBox GT Report V2

Status date: 2026-07-01.

This GT-box package covers every positive query in `benchmark_queries_v2.json`.
Negative queries correctly have no expected target box.

The boxes are manual/coarse benchmark reference boxes, not official dataset GT and not pixel-perfect instance masks.

## Counts

| Metric | Count |
|---|---:|
| Positive queries with bbox GT | 125 |
| Negative queries not applicable | 25 |
| Query records with bbox GT | 125 |
| Unique object observation boxes | 174 |
| Query-box links | 290 |

## Evidence

- GT JSON: `docs/benchmarks/manual_bbox_gt_v2.json`
- Benchmark with expected boxes: `docs/benchmarks/benchmark_queries_v2.json`
- Preview overlays: `docs/benchmarks/bbox_gt_v2_previews/`
- Scene manifests: `backend/data/scenes/<scene_id>/scene_manifest.json`

## Limitations

- Boxes are coarse object/region reference boxes from manual ViewJSON location evidence.
- These boxes are suitable for internal same-input benchmark gating.
- Do not report them as official Replica, ScanNet, or independent dataset boxes.
- Use 3D IoU as a coarse regression signal, not a final localization claim.
