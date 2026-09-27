from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from PIL import Image


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_depth(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".npy":
        arr = np.load(src)
        np.save(dst.with_suffix(".npy"), arr)
        return dst.with_suffix(".npy"), arr
    shutil.copy2(src, dst)
    return dst, None


def main():
    parser = argparse.ArgumentParser(description="Prepare one scene in unified Replica-style RGB-D format.")
    parser.add_argument("--input", required=True, help="Input scene folder, e.g. data/scenes/default")
    parser.add_argument("--output", required=True, help="Output folder, e.g. data/replica/pilot_scene_001")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--source-dataset", default="provided_data")
    parser.add_argument(
        "--image-policy",
        choices=["scale_intrinsics", "resize_to_intrinsics"],
        default="scale_intrinsics",
        help="scale_intrinsics keeps actual image size and scales intrinsics; resize_to_intrinsics resizes RGB to transforms.json w/h.",
    )
    args = parser.parse_args()

    input_scene = Path(args.input)
    output_scene = Path(args.output)
    scene_name = args.scene_name or input_scene.name
    transforms = load_json(input_scene / "transforms.json")
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("No frames in transforms.json")

    rgb_out = output_scene / "rgb"
    depth_out = output_scene / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    original_modes = set()
    sizes = []
    pose_frames = []
    depth_stats = []

    target_w = int(transforms["w"])
    target_h = int(transforms["h"])

    for i, frame in enumerate(frames):
        rgb_src = input_scene / frame["file_path"]
        depth_src = input_scene / frame["depth_file_path"]
        if not rgb_src.exists():
            raise FileNotFoundError(rgb_src)
        if not depth_src.exists():
            raise FileNotFoundError(depth_src)

        with Image.open(rgb_src) as img:
            original_modes.add(img.mode)
            img = img.convert("RGB")  # removes alpha channel; final output is true RGB
            if args.image_policy == "resize_to_intrinsics":
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            sizes.append(img.size)
            rgb_name = f"{i:06d}.png"
            img.save(rgb_out / rgb_name)

        depth_name = f"{i:06d}.npy"
        depth_dst, arr = save_depth(depth_src, depth_out / depth_name)
        if arr is not None:
            depth_stats.append({
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            })

        pose_frames.append({
            "frame_index": i,
            "view_id": frame.get("view_id", f"v{i + 1:03d}"),
            "rgb_file": f"rgb/{rgb_name}",
            "depth_file": f"depth/{depth_dst.name}",
            "transform_matrix": frame["transform_matrix"],
        })

    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) != 1:
        raise ValueError(f"Inconsistent output RGB sizes: {unique_sizes}")

    out_w, out_h = unique_sizes[0]

    if args.image_policy == "scale_intrinsics":
        sx = out_w / float(transforms["w"])
        sy = out_h / float(transforms["h"])
        intrinsics = {
            "camera_model": transforms.get("camera_model", "OPENCV"),
            "width": out_w,
            "height": out_h,
            "fx": transforms["fl_x"] * sx,
            "fy": transforms["fl_y"] * sy,
            "cx": transforms["cx"] * sx,
            "cy": transforms["cy"] * sy,
            "source": "transforms.json scaled to actual RGB resolution",
            "original_intrinsics": {
                "width": transforms["w"],
                "height": transforms["h"],
                "fx": transforms["fl_x"],
                "fy": transforms["fl_y"],
                "cx": transforms["cx"],
                "cy": transforms["cy"],
            },
            "scale": {"x": sx, "y": sy},
        }
    else:
        intrinsics = {
            "camera_model": transforms.get("camera_model", "OPENCV"),
            "width": target_w,
            "height": target_h,
            "fx": transforms["fl_x"],
            "fy": transforms["fl_y"],
            "cx": transforms["cx"],
            "cy": transforms["cy"],
            "source": "transforms.json; RGB resized to match intrinsics resolution",
        }

    depth_all_constant = bool(depth_stats) and all(s["min"] == s["max"] for s in depth_stats)
    warnings = []
    if "RGBA" in original_modes:
        warnings.append("Input images were RGBA. Output images were converted to RGB by removing alpha.")
    if depth_all_constant:
        warnings.append("Depth maps are constant-valued, so they cannot produce meaningful real 3D geometry.")
    if args.image_policy == "scale_intrinsics":
        warnings.append("Intrinsics were scaled to output RGB size. This is preferable when preserving original image pixels.")
    else:
        warnings.append("RGB images were resized to match original intrinsics. This is preferable if the pipeline strictly requires 800x600.")

    write_json(output_scene / "intrinsics.json", intrinsics)
    write_json(output_scene / "poses.json", {
        "pose_convention": "camera_to_world_assumed_from_transforms_json",
        "frames": pose_frames,
    })
    write_json(output_scene / "metadata.json", {
        "scene_name": scene_name,
        "source_dataset": args.source_dataset,
        "num_frames": len(frames),
        "has_rgb": True,
        "rgb_mode": "RGB",
        "original_rgb_modes": sorted(original_modes),
        "has_depth": True,
        "depth_format": ".npy",
        "has_poses": True,
        "has_intrinsics": True,
        "has_ground_truth_labels": False,
        "image_policy": args.image_policy,
        "output_rgb_size": {"width": out_w, "height": out_h},
        "depth_stats_first_5": depth_stats[:5],
        "depth_all_constant": depth_all_constant,
        "warnings": warnings,
    })

    print(f"Prepared: {output_scene}")
    print(f"Frames: {len(frames)}")
    print(f"RGB output mode: RGB")
    print(f"RGB output size: {out_w}x{out_h}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
