#!/usr/bin/env python3
"""Patch pinned LangSplat preprocessing to accept local low-resource checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    replacements = {
        "pretrained=self.config.clip_model_pretrained,": (
            "pretrained=os.environ.get('SEMANTICSPLAT_OPENCLIP_CHECKPOINT', "
            "self.config.clip_model_pretrained),"
        ),
        'sam_model_registry["vit_h"]': (
            'sam_model_registry[os.environ.get("SEMANTICSPLAT_SAM_MODEL", "vit_h")]'
        ),
        "points_per_side=32,": (
            'points_per_side=int(os.environ.get("SEMANTICSPLAT_SAM_POINTS_PER_SIDE", "32")),'
        ),
        "pred_iou_thresh=0.7,": (
            'points_per_batch=int(os.environ.get("SEMANTICSPLAT_SAM_POINTS_PER_BATCH", "64")),\n'
            "        pred_iou_thresh=0.7,"
        ),
        "crop_n_layers=1,": (
            'crop_n_layers=int(os.environ.get("SEMANTICSPLAT_SAM_CROP_LAYERS", "1")),'
        ),
        "index = scores.topk(3).indices": (
            "index = scores.topk(min(3, scores.numel())).indices"
        ),
        "keep_conf[index, 0] = True": "keep_conf[index] = True",
        "keep_inner_u[index, 0] = True": "keep_inner_u[index] = True",
        "keep_inner_l[index, 0] = True": "keep_inner_l[index] = True",
        "    for masks_lvl in (args):\n        seg_pred =": (
            "    for masks_lvl in (args):\n"
            "        if not masks_lvl:\n"
            "            masks_new += (masks_lvl,)\n"
            "            continue\n"
            "        seg_pred ="
        ),
        """    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
""": """    seg_images, seg_maps = {}, {}
    for mode, masks in (
        ('default', masks_default), ('s', masks_s), ('m', masks_m), ('l', masks_l)
    ):
        if len(masks) != 0:
            seg_images[mode], seg_maps[mode] = mask2segmap(masks, image)
        else:
            seg_maps[mode] = -np.ones(image.shape[:2], dtype=np.int32)
""",
        """        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
""": """        tiles = seg_images.get(mode)
        if tiles is None:
            clip_embeds[mode] = torch.empty((0, 512), dtype=torch.float16)
            continue
        tiles = tiles.to("cuda")
""",
    }
    for old, new in replacements.items():
        if old not in source:
            raise SystemExit(f"Pinned LangSplat source no longer contains: {old}")
        source = source.replace(old, new)
    args.out.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
