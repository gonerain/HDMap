#!/usr/bin/python3
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path:
    first_path = Path(sys.path[0] or ".").resolve()
    if first_path == SCRIPT_DIR:
        sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

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


def count_outdoor_frames(input_path):
    input_path = Path(input_path)
    frame_count = 0
    with input_path.open("rb") as f:
        while True:
            try:
                frame = pickle.load(f)
            except EOFError:
                break
            if isinstance(frame, list):
                raise ValueError(f"{input_path} looks like an indoor.pkl list payload, not an outdoor.pkl stream")
            frame_count += 1
    return frame_count


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


def quaternion_to_rotation_matrix(quaternion_xyzw):
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_pose_csv(trajectory_path):
    pose = np.loadtxt(trajectory_path, delimiter=",")
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        pose = pose.reshape(1, -1)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"{trajectory_path} must have shape (N, 7), got {pose.shape}")
    return pose


def pose_row_to_matrix4x4(pose_row):
    pose_row = np.asarray(pose_row, dtype=np.float64)
    if pose_row.shape != (7,):
        raise ValueError(f"pose row must have shape (7,), got {pose_row.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(pose_row[3:])
    transform[:3, 3] = pose_row[:3]
    return transform


def resolve_image_path(images_dir, frame_index, one_based_images=True):
    image_number = frame_index + 1 if one_based_images else frame_index
    return Path(images_dir) / f"{image_number:06d}.png"


def count_contiguous_images(images_dir, one_based_images=True):
    images_dir = Path(images_dir)
    count = 0
    while resolve_image_path(images_dir, count, one_based_images=one_based_images).exists():
        count += 1
    return count


def summarize_aligned_range(input_path, trajectory_path, images_dir, one_based_images=True):
    pkl_frames = count_outdoor_frames(input_path)
    pose_rows = len(load_pose_csv(trajectory_path))
    image_frames = count_contiguous_images(images_dir, one_based_images=one_based_images)
    aligned_frames = min(pkl_frames, pose_rows, image_frames)
    return {
        "pkl_frames": int(pkl_frames),
        "pose_rows": int(pose_rows),
        "image_frames": int(image_frames),
        "aligned_frames": int(aligned_frames),
    }


def colorize_depth(depth_values, valid_mask):
    depth_values = np.asarray(depth_values, dtype=np.float64)
    colors_bgr = np.zeros((len(depth_values), 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return colors_bgr

    valid_depth = depth_values[valid_mask]
    min_depth = float(valid_depth.min())
    max_depth = float(valid_depth.max())
    if max_depth - min_depth < 1e-6:
        normalized = np.zeros_like(valid_depth, dtype=np.float32)
    else:
        normalized = ((valid_depth - min_depth) / (max_depth - min_depth)).astype(np.float32)
    colormap = cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    colors_bgr[valid_mask] = colormap.reshape(-1, 3)
    return colors_bgr


def draw_projection_overlay(image, projection, point_radius=2):
    overlay = image.copy()
    valid_mask = projection["in_bounds_mask"]
    pixels = projection["pixels"]
    colors_bgr = colorize_depth(projection["depth"], valid_mask)
    draw_order = np.argsort(projection["depth"][valid_mask])[::-1]
    valid_indices = np.flatnonzero(valid_mask)[draw_order]
    for idx in valid_indices:
        u, v = pixels[idx]
        cv2.circle(overlay, (int(round(u)), int(round(v))), point_radius, colors_bgr[idx].tolist(), -1, lineType=cv2.LINE_AA)
    return overlay


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


def render_outdoor_projection_overlay(
    input_path,
    frame_index,
    config_path="config/outdoor_config.json",
    trajectory_path="result/outdoor/pose.csv",
    images_dir="result/outdoor/originpics",
    output_path=None,
):
    range_summary = summarize_aligned_range(input_path, trajectory_path, images_dir, one_based_images=True)
    if frame_index >= range_summary["aligned_frames"]:
        raise IndexError(
            f"frame {frame_index} exceeds aligned frame range {range_summary['aligned_frames']} "
            f"(pkl={range_summary['pkl_frames']}, pose={range_summary['pose_rows']}, images={range_summary['image_frames']})"
        )
    poses = load_pose_csv(trajectory_path)

    image_path = resolve_image_path(images_dir, frame_index, one_based_images=True)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    lidar_to_world = pose_row_to_matrix4x4(poses[frame_index])
    result = project_outdoor_frame_with_config(
        input_path,
        frame_index,
        config_path=config_path,
        image_shape=image.shape[:2],
        lidar_to_world=lidar_to_world,
    )
    overlay = draw_projection_overlay(image, result["projection"])

    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"failed to write overlay image: {output_path}")

    return {
        "range_summary": range_summary,
        "image_path": str(image_path),
        "output_path": str(output_path) if output_path is not None else None,
        "overlay": overlay,
        **result,
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
    parser.add_argument("--overlay", action="store_true", help="Render projected points on the corresponding origin image.")
    parser.add_argument("--trajectory", default="result/outdoor/pose.csv", help="Path to pose.csv used for world-to-lidar transform.")
    parser.add_argument("--images-dir", default="result/outdoor/originpics", help="Directory containing saved origin images.")
    parser.add_argument("--output-image", default="demo_output/frame_overlay.png", help="Where to save the overlay image.")
    parser.add_argument(
        "--show-aligned-range",
        action="store_true",
        help="Report effective aligned frame count across outdoor.pkl, pose.csv, and origin images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_index = args.frame_index - 1 if args.one_based else args.frame_index
    report = build_frame_report(args.input, frame_index, preview_points=args.print_points)

    if args.overlay or args.show_aligned_range:
        report["aligned_range"] = summarize_aligned_range(
            args.input,
            args.trajectory,
            args.images_dir,
            one_based_images=True,
        )

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

    if args.overlay:
        rendered = render_outdoor_projection_overlay(
            args.input,
            frame_index,
            config_path=args.config,
            trajectory_path=args.trajectory,
            images_dir=args.images_dir,
            output_path=args.output_image,
        )
        projection = rendered["projection"]
        report["overlay"] = {
            "image_path": rendered["image_path"],
            "output_path": rendered["output_path"],
            "aligned_frames": int(rendered["range_summary"]["aligned_frames"]),
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
