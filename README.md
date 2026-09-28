# Beyond Proximity — release

Code, queries, and results for **"Beyond Proximity: How Much of a 3D Scene Should a Language Model Read? A Controlled Study of Hierarchical Context Reduction"** (submitted to IEEE Access).

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX sources, figures, and PDF of the manuscript |
| `backend/` | Retrieval, hierarchy, and grounding code (Algorithm 1 is in `backend/query/recall_controlled.py`) |
| `backend/data/scenes/*/views`, `transforms.json`, `tree` | Authored view records, camera poses, and hierarchies of the five internal scenes |
| `docs/benchmarks/` | The 150 internal queries with expected views, manual zones, and public-benchmark query files |
| `scripts/` | Every experiment and figure script |
| `outputs/ieee_revision2/` | Results of the revised study; each folder has a `RESULTS.md` with the exact commands |
| `outputs/ieee_revision/`, `outputs/raw_rgbd_hierarchy/` | Earlier result files that the revised analyses read |
| `tests/` | Unit tests |

## Main experiments

```bash
python3 -B scripts/run_ieee_revision2_frontier.py      # lexical/BGE: Algorithm 1 vs flat top-k, abstention, ablation
python3 -B scripts/run_ieee_llm_cascade.py --help      # Qwen2.5-7B cascade (A2) and answer selection (B10)
python3 -B scripts/run_rev2_hierarchy_analyses.py      # hierarchy grid, construction constants, structure
python3 -B scripts/run_ieee_scale_hierarchy.py --help  # 155,535-record ReferIt3D scale study
python3 -B scripts/audit_predicted_map_frames.py       # predicted-map frame audit, Replica GT correction
cd scripts && python3 -B export_ieee_revision2_figures.py --out ../paper/figures
```

## Not included

- **Rendered RGB-D views of the five internal scenes.** They are renders of public 3D Gaussian splatting scenes from SuperSplat (see Scene credits). We do not redistribute the reconstructions or the renders. Experiments that read images (the raw RGB-D constructor and the CLIP scorer) therefore cannot be rerun from this release alone. Their saved outputs are included.
- **ScanNet, Replica, ReferIt3D (Nr3D/Sr3D+).** Obtain these from their providers under their licenses. Only derived records and identifiers are included.
- **ConceptGraphs predicted maps and model weights.**

## Scene credits

The five internal scenes are public 3D Gaussian splatting reconstructions published on [SuperSplat](https://superspl.at) by their authors:

| Scene in the paper | SuperSplat scene | Author |
|---|---|---|
| Outdoor drone (island) | [Alcatraz Island](https://superspl.at/scene/6fd4ec6c) | vwoo |
| Museum | [Castle Reichenstein - Museum](https://superspl.at/scene/c0da7173) | schindelar3d |
| Conference hall | to be added | |
| Theater | to be added | |
| Outdoor street | to be added | |
