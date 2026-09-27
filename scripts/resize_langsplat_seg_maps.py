#!/usr/bin/env python3
"""Resize LangSplat segmentation-ID maps to their source image dimensions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    image_dir = args.scene / "images"
    changed = 0
    seg_paths = []
    for dirname in ("language_features", "language_features_dim3"):
        seg_paths.extend(sorted((args.scene / dirname).glob("*_s.npy")))
    for seg_path in seg_paths:
        image_path = next(
            (path for suffix in (".jpg", ".png", ".jpeg") if (path := image_dir / f"{seg_path.stem[:-2]}{suffix}").exists()),
            None,
        )
        if image_path is None:
            raise SystemExit(f"No source image for {seg_path}")
        with Image.open(image_path) as image:
            target = image.size
        seg_maps = np.load(seg_path)
        if seg_maps.ndim != 3:
            raise SystemExit(f"Expected [levels,height,width] map: {seg_path}")
        if (seg_maps.shape[2], seg_maps.shape[1]) == target:
            continue
        resized = np.stack(
            [
                np.asarray(
                    Image.fromarray(level).resize(target, Image.Resampling.NEAREST),
                    dtype=seg_maps.dtype,
                )
                for level in seg_maps
            ]
        )
        np.save(seg_path, resized)
        changed += 1
    print(f"LangSplat segmentation maps resized: {changed}")


if __name__ == "__main__":
    main()
