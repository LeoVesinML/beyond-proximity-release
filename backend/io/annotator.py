"""Draw 2-D bounding boxes on scene images and return base64-encoded PNGs.

Also provides crop_for_refinement / map_refined_bbox for the two-stage bbox
refinement pipeline: rough VLM bbox → padded crop → LLM re-estimates within
crop → map coordinates back to original image space.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from backend.config import DATA_DIR


# Colour palette: one per object (cycles if more than 8)
_PALETTE = [
    "#FF4444",  # red
    "#44AAFF",  # blue
    "#44FF88",  # green
    "#FFD700",  # gold
    "#FF88FF",  # pink
    "#FF8C00",  # orange
    "#00CED1",  # teal
    "#BDB76B",  # khaki
]

_FONT_SIZE = 14


def _get_font(size: int = _FONT_SIZE):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _load_image(scene_id: str, view_id: str) -> Optional[Image.Image]:
    scene_dir = DATA_DIR / scene_id
    img_path = scene_dir / "images" / f"{view_id}.png"
    if not img_path.exists():
        img_path = scene_dir / "images" / f"{view_id}.jpg"
    if not img_path.exists():
        return None
    return Image.open(img_path).convert("RGB")


# Placeholder captures (e.g. stub PNGs) are nearly black; real views are much brighter.
MIN_VIEW_LUMINANCE = 100.0


def view_image_mean_luminance(scene_id: str, view_id: str) -> Optional[float]:
    """Mean RGB of the view PNG, or None if missing."""
    img = _load_image(scene_id, view_id)
    if img is None:
        return None
    import numpy as np
    return float(np.array(img).mean())


def view_image_usable(scene_id: str, view_id: str, min_mean: float = MIN_VIEW_LUMINANCE) -> bool:
    """False for missing images or dark placeholder stubs unsuitable for Query Flow."""
    mean = view_image_mean_luminance(scene_id, view_id)
    return mean is not None and mean >= min_mean


def annotate_view(
    scene_id: str,
    view_id: str,
    boxes: List[dict],
    max_width: int = 1200,
) -> Optional[str]:
    """Draw boxes on the view image and return a base64-encoded PNG data URL.

    Each box in `boxes` is:
      {
        "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],  # 0-1 normalised
        "label": "<text>",               # optional
        "color": "#RRGGBB",              # optional, else palette
        "confidence": 0.0-1.0,          # optional
      }
    """
    img = _load_image(scene_id, view_id)
    if img is None:
        return None

    # Resize if very large (keeps network transfer reasonable)
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)

    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font()

    for i, box in enumerate(boxes):
        bbox = box.get("bbox_2d") or box.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1_n, y1_n, x2_n, y2_n = bbox
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = int(x2_n * w)
        y2 = int(y2_n * h)

        color_hex = box.get("color") or _PALETTE[i % len(_PALETTE)]
        # Semi-transparent fill
        fill_rgb = _hex_to_rgb(color_hex)
        draw.rectangle([x1, y1, x2, y2], fill=(*fill_rgb, 35), outline=color_hex, width=3)

        # Label pill
        label = box.get("label", "")
        conf = box.get("confidence")
        if conf is not None:
            label = f"{label} ({int(float(conf) * 100)}%)" if label else f"{int(float(conf) * 100)}%"
        if label:
            try:
                # Pillow 10+ textbbox
                bbox_text = draw.textbbox((0, 0), label, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)
            pad = 4
            pill_y0 = max(0, y1 - th - pad * 2)
            pill_x1 = min(x1 + tw + pad * 2, w)
            draw.rectangle([x1, pill_y0, pill_x1, y1], fill=color_hex)
            draw.text((x1 + pad, pill_y0 + pad), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


BBOX_REFINEMENT_PROMPT_TEMPLATE = """\
You are looking at a CROPPED region of a larger scene image.
The crop was extracted because it is the approximate location of: "{label}"

Your task: find "{label}" in this cropped image and return a PRECISE bounding box.

Use this step-by-step process:
  a) Is "{label}" visible in this crop? (yes / no / partially)
  b) Horizontal extent: where does the object's left edge fall in the crop? (0=left, 1=right)
     Where does its right edge fall?
  c) Vertical extent: where does the object's top edge fall? (0=top, 1=bottom)
     Where does its bottom edge fall?
  d) Refine to nearest 0.02 (not 0.05 — be precise at this zoom level).
  e) The bbox should tightly wrap the object — not the crop itself.

Return ONLY valid JSON (no markdown):
{{
  "found_in_crop": true | false,
  "confidence": "high" | "medium" | "low",
  "bbox_2d_in_crop": [<x1_norm>, <y1_norm>, <x2_norm>, <y2_norm>] | null,
  "reasoning": "<one sentence>"
}}
"""


def crop_for_refinement(
    scene_id: str,
    view_id: str,
    rough_bbox: List[float],
    label: str = "",
    padding: float = 1.0,
    max_crop_size: int = 640,
) -> Optional[Dict]:
    """Expand rough_bbox by `padding` fraction, crop image, return crop metadata.

    Default padding=1.0 (100% of bbox size on each side) is intentionally generous
    because VLM bbox estimates can be off by more than 15% of image height for small
    objects.  The LLM re-estimates within the crop, which is then mapped back.

    Returns a dict with:
      - crop_b64: base64 PNG data URL of the cropped region
      - crop_region: [cx1_n, cy1_n, cx2_n, cy2_n] in original normalised coords
      - prompt: VLM prompt to paste into the LLM
      - original_size: [w, h] of the original image
    Returns None if image not found.
    """
    img = _load_image(scene_id, view_id)
    if img is None:
        return None

    W, H = img.size
    x1_n, y1_n, x2_n, y2_n = rough_bbox

    # Expand bbox by padding fraction of its own size.
    # Minimum absolute padding of 0.15 ensures we capture objects that the VLM
    # placed up to ~15% of the image dimension away from their true position.
    bw = x2_n - x1_n
    bh = y2_n - y1_n
    expand_x = max(bw * padding, 0.15)
    expand_y = max(bh * padding, 0.15)

    cx1 = max(0.0, x1_n - expand_x)
    cy1 = max(0.0, y1_n - expand_y)
    cx2 = min(1.0, x2_n + expand_x)
    cy2 = min(1.0, y2_n + expand_y)

    # Convert to pixels
    px1, py1 = int(cx1 * W), int(cy1 * H)
    px2, py2 = int(cx2 * W), int(cy2 * H)

    # Ensure minimum 32px crop size
    if px2 - px1 < 32:
        px2 = min(W, px1 + 32)
    if py2 - py1 < 32:
        py2 = min(H, py1 + 32)

    crop = img.crop((px1, py1, px2, py2))

    # Resize if the crop is still large
    cw, ch = crop.size
    if max(cw, ch) > max_crop_size:
        scale = max_crop_size / max(cw, ch)
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    crop_data_url = f"data:image/png;base64,{b64}"

    return {
        "crop_b64": crop_data_url,
        "crop_region": [cx1, cy1, cx2, cy2],
        "original_size": [W, H],
        "rough_bbox": rough_bbox,
        "label": label,
        "prompt": BBOX_REFINEMENT_PROMPT_TEMPLATE.format(label=label or "the target object"),
    }


def map_refined_bbox(
    crop_region: List[float],
    bbox_in_crop: List[float],
) -> List[float]:
    """Map a bbox expressed relative to a crop back to original image coordinates.

    crop_region: [cx1_n, cy1_n, cx2_n, cy2_n] — the crop in original normalised coords
    bbox_in_crop: [rx1, ry1, rx2, ry2] — bbox expressed 0-1 within the crop
    Returns: [x1_n, y1_n, x2_n, y2_n] in original image normalised coordinates
    """
    cx1, cy1, cx2, cy2 = crop_region
    cw = cx2 - cx1
    ch = cy2 - cy1

    rx1, ry1, rx2, ry2 = bbox_in_crop
    x1 = cx1 + rx1 * cw
    y1 = cy1 + ry1 * ch
    x2 = cx1 + rx2 * cw
    y2 = cy1 + ry2 * ch

    # Clamp to [0, 1]
    return [
        round(max(0.0, min(1.0, x1)), 4),
        round(max(0.0, min(1.0, y1)), 4),
        round(max(0.0, min(1.0, x2)), 4),
        round(max(0.0, min(1.0, y2)), 4),
    ]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
