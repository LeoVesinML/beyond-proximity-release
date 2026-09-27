#!/usr/bin/env python3
"""Patch the pinned LangSplat SAM fork to handle empty scale buckets."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    old = """        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
"""
    new = """        # Empty semantic-size buckets are valid for some frames.
        if len(data[\"boxes\"]) == 0:
            data.to_numpy()
            return data

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
"""
    if old not in source:
        raise SystemExit("Pinned SAM source no longer matches empty-bucket patch")
    args.source.write_text(source.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
