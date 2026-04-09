#!/usr/bin/python3
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.pkl_frame_loader import summarize_frame


def validate_outdoor_pkl(input_path):
    input_path = Path(input_path)
    with input_path.open("rb") as f:
        first_obj = pickle.load(f)
    if isinstance(first_obj, list):
        raise ValueError(f"{input_path} looks like an indoor.pkl list payload, not an outdoor.pkl stream")
    return np.asarray(first_obj)


def load_outdoor_frame(input_path, frame_index):
    if frame_index < 0:
        raise IndexError(f"frame_index must be >= 0, got {frame_index}")

    input_path = Path(input_path)
    with input_path.open("rb") as f:
        try:
            first_obj = pickle.load(f)
        except EOFError as exc:
            raise ValueError(f"{input_path} is empty") from exc

        if isinstance(first_obj, list):
            raise ValueError(f"{input_path} looks like an indoor.pkl list payload, not an outdoor.pkl stream")
        if frame_index == 0:
            return np.asarray(first_obj)

        current_index = 0
        while True:
            try:
                frame = pickle.load(f)
            except EOFError as exc:
                raise IndexError(f"frame {frame_index} exceeds outdoor pkl length") from exc
            current_index += 1
            if current_index == frame_index:
                return np.asarray(frame)


def build_frame_report(input_path, frame_index, preview_points=5):
    frame = load_outdoor_frame(input_path, frame_index)
    report = summarize_frame(frame)
    report["input"] = str(Path(input_path))
    report["mode"] = "outdoor"
    report["frame_index"] = int(frame_index)
    report["preview_points"] = frame[: max(int(preview_points), 0)].tolist()
    return report


def as_matrix3x3(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    return arr


def as_matrix4x4(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}")
    return arr


def project_lidar_points_to_image(points_xyz, intrinsic, lidar_to_camera, image_shape, lidar_to_world=None):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must have shape (N, >=3), got {points_xyz.shape}")

    intrinsic = as_matrix3x3(intrinsic, "intrinsic")
    lidar_to_camera = as_matrix4x4(lidar_to_camera, "lidar_to_camera")

    if len(image_shape) < 2:
        raise ValueError(f"image_shape must contain at least (height, width), got {image_shape}")
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image_shape must be positive, got {(image_h, image_w)}")

    points_lidar = points_xyz[:, :3]
    input_frame = "lidar"
    if lidar_to_world is not None:
        lidar_to_world = as_matrix4x4(lidar_to_world, "lidar_to_world")
        world_to_lidar = np.linalg.inv(lidar_to_world)
        hom_world = np.concatenate((points_lidar, np.ones((len(points_lidar), 1), dtype=np.float64)), axis=1)
        points_lidar = (world_to_lidar @ hom_world.T).T[:, :3]
        input_frame = "world"

    hom_lidar = np.concatenate((points_lidar, np.ones((len(points_lidar), 1), dtype=np.float64)), axis=1)
    points_camera = (lidar_to_camera @ hom_lidar.T).T[:, :3]

    depth = points_camera[:, 2]
    in_front_mask = depth > 1e-6
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    if np.any(in_front_mask):
        camera_valid = points_camera[in_front_mask]
        uvw = (intrinsic @ camera_valid.T).T
        pixels[in_front_mask] = uvw[:, :2] / uvw[:, 2:3]

    in_bounds_mask = (
        in_front_mask
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_w))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_h))
    )
    out_of_bounds_mask = in_front_mask & ~in_bounds_mask
    behind_camera_mask = ~in_front_mask

    warnings = []
    if np.any(out_of_bounds_mask):
        warnings.append(
            f"{int(np.count_nonzero(out_of_bounds_mask))} projected points fall outside image bounds "
            f"{image_w}x{image_h}"
        )
    if np.any(behind_camera_mask):
        warnings.append(f"{int(np.count_nonzero(behind_camera_mask))} points are behind the camera")

    return {
        "pixels": pixels,
        "points_camera": points_camera,
        "depth": depth,
        "in_front_mask": in_front_mask,
        "in_bounds_mask": in_bounds_mask,
        "out_of_bounds_mask": out_of_bounds_mask,
        "behind_camera_mask": behind_camera_mask,
        "image_shape": [image_h, image_w],
        "input_frame": input_frame,
        "warnings": warnings,
    }


def project_outdoor_frame_with_config(
    input_path,
    frame_index,
    config_path="config/outdoor_config.json",
    image_shape=None,
    lidar_to_world=None,
):
    frame = load_outdoor_frame(input_path, frame_index)
    with open(config_path, "r") as f:
        config = json.load(f)

    intrinsic = config["intrinsic"]
    lidar_to_camera = config["extrinsic"]
    if image_shape is None:
        image_shape = (
            int(config.get("image_height", 720)),
            int(config.get("image_width", 1280)),
        )

    projection = project_lidar_points_to_image(
        frame[:, :3],
        intrinsic=intrinsic,
        lidar_to_camera=lidar_to_camera,
        image_shape=image_shape,
        lidar_to_world=lidar_to_world,
    )

    return {
        "frame": frame,
        "projection": projection,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Read one frame from outdoor.pkl and optionally project it to image.")
    parser.add_argument(
        "-i",
        "--input",
        default="result/outdoor/outdoor.pkl",
        help="Path to outdoor.pkl",
    )
    parser.add_argument(
        "-f",
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to read. Defaults to 0-based indexing.",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --frame-index as 1-based.",
    )
    parser.add_argument(
        "--print-points",
        type=int,
        default=5,
        help="How many leading points to include in preview_points.",
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Also run LiDAR-to-image projection using config intrinsics/extrinsics.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/outdoor_config.json",
        help="Projection config path.",
    )
    parser.add_argument("--image-height", type=int, default=720, help="Image height used for bounds checking.")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width used for bounds checking.")
    return parser.parse_args()


def main():
    args = parse_args()
    frame_index = args.frame_index - 1 if args.one_based else args.frame_index
    report = build_frame_report(args.input, frame_index, preview_points=args.print_points)

    if args.project:
        result = project_outdoor_frame_with_config(
            args.input,
            frame_index,
            config_path=args.config,
            image_shape=(args.image_height, args.image_width),
        )
        projection = result["projection"]
        report["projection"] = {
            "input_frame": projection["input_frame"],
            "image_shape": projection["image_shape"],
            "points_total": int(len(projection["pixels"])),
            "points_in_front": int(np.count_nonzero(projection["in_front_mask"])),
            "points_in_bounds": int(np.count_nonzero(projection["in_bounds_mask"])),
            "points_out_of_bounds": int(np.count_nonzero(projection["out_of_bounds_mask"])),
            "points_behind_camera": int(np.count_nonzero(projection["behind_camera_mask"])),
            "warnings": projection["warnings"],
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
