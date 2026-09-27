# Manual GT Coverage Report V2

Status date: 2026-07-01.

## Scope

The benchmark provides expected IDs derived from manual captured-scene ViewJSON annotations and generated tree nodes.

| GT type | Status | Count | Claim allowed |
|---|---|---:|---|
| Manual semantic-index expected views | available | 125 queries | Semantic-index regression / same-input retrieval quality. |
| Manual semantic-index expected nodes | available | 125 queries | Tree-node selection checks where current tree IDs exist. |
| Manual region expectations | partial | 50 queries | Region-level reference where ViewJSON contains matching region labels. |
| Manual zone expectations | available, tree-native after rebuild | 125 queries | Zone-denominator checks and graph-enrichment target labels. |
| Tree-native zone traversal | available after rebuild | 5 captured scenes | Current trees include stable `manual_zone_*` nodes. |
| Independent visual/dataset GT | unavailable | 0 | Do not report independent semantic accuracy. |
| Manual coarse GT boxes | available after bbox builder | 125 positive queries | Internal coarse regression signal only, not official dataset localization. |
| Pixel-perfect GT boxes / masks | unavailable | 0 | Do not report official 3D localization accuracy. |

## Verification Status

- `verified_from_view_json`: 150
- `verified_dataset_gt`: 0
- `missing_gt`: 0 in v2, because every query has a manual semantic-index reference label.

## Important Limitation

These labels are suitable for evaluating whether methods retrieve the same manual semantic-index entries. They do not prove the annotations are visually correct or dataset-ground-truth correct.
