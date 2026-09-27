# BBQ-aligned grounding summary

- Benchmark: `scannet_bbq_grounding_extended_v2`
- Queries: 64
- Canonical / legacy-adapted results: 64 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 0.5017 | 56 eligible |
| Acc@0.1 | 0.5000 | 28 / 56 |
| Acc@0.25 | 0.5000 | 28 / 56 |
| Acc@0.5 | 0.5000 | 28 / 56 |
| Recall@1 exact object ID | 0.5357 | 30 / 56 |
| Recall@3 exact object ID | 0.8214 | 46 / 56 |
| Recall@5 exact object ID | 0.8571 | 48 / 56 |
| MRR exact object ID | 0.6846 | 56 eligible |
| Verified-negative accuracy | 1.0000 | 8 / 8 |

## Segmentation

| Metric | Status | Value |
|---|---|---:|
| mAcc | N/A | N/A |
| mIoU | N/A | N/A |
| fmIoU | N/A | N/A |

## Failure taxonomy

| Failure | Count |
|---|---:|
| wrong object | 20 |
| wrong relation | 1 |
| false positive | 0 |
| invalid/unavailable prediction | 8 |
| bad bbox | 8 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 0.0018 | 0.1130 | 64 |
| checked_nodes | 1.5000 | 96.0000 | 64 |
| checked_views | 3.4219 | 219.0000 | 64 |
| checked_objects | 29.2812 | 1874.0000 | 64 |
| context_chars | 5026.7344 | 321711.0000 | 64 |
| estimated_tokens | 1267.9688 | 81150.0000 | 64 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
