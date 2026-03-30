#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from collections import deque
from typing import Deque, Iterator, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from tf import transformations


Array = np.ndarray


def quat_to_yaw(quat: Array) -> float:
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def iter_pickled_arrays(path: str) -> Iterator[Array]:
    with open(path, "rb") as f:
        while True:
            try:
                arr = pickle.load(f)
            except EOFError:
                break
            if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"Expected Nx4 array, got {type(arr)!r} with shape {getattr(arr, 'shape', None)}")
            yield arr.astype(np.float32, copy=False)


def load_road_frames(path: str, road_class: int, max_frames: int) -> list:
    road_frames = []
    for frame_idx, frame in enumerate(iter_pickled_arrays(path)):
        if frame_idx >= max_frames:
            break
        road = frame[frame[:, 3].astype(np.int32) == road_class][:, :3]
        road_frames.append(road.astype(np.float32, copy=False))
    return road_frames


def pcd_trans(pcd: Array, dt: Array, dr: Array, inverse: bool = False) -> Array:
    if len(pcd) == 0:
        width = pcd.shape[1] if isinstance(pcd, np.ndarray) and pcd.ndim == 2 else 3
        return np.empty((0, width), dtype=np.float32)
    pcd = np.asarray(pcd, dtype=np.float32)
    length = len(pcd)
    pcd_t = pcd.T.copy()
    transpcd = np.vstack((pcd_t[:3], np.ones((1, length), dtype=np.float32)))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    if inverse:
        mat44 = np.asarray(np.matrix(mat44).I)
    pcd_t[:3] = np.dot(mat44, transpcd)[:3]
    return pcd_t.T


def crop_local_box(points_xyz: Array, forward: float, lateral: float, z_min: float, z_max: float) -> Array:
    if len(points_xyz) == 0:
        return points_xyz
    mask = (
        (points_xyz[:, 1] >= 0.0)
        & (points_xyz[:, 1] <= forward)
        & (np.abs(points_xyz[:, 0]) <= lateral)
        & (points_xyz[:, 2] >= z_min)
        & (points_xyz[:, 2] <= z_max)
    )
    return points_xyz[mask]


def rotate_xy(points_xy: Array, yaw: float) -> Array:
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_xy.astype(np.float32, copy=False) @ rot.T


def unrotate_xy(points_xy: Array, yaw: float) -> Array:
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_xy.astype(np.float32, copy=False) @ rot.T


def aligned_edge_to_world(edge_xy: Array, yaw: float, pose: Array) -> Array:
    if len(edge_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    local_xy = unrotate_xy(edge_xy, yaw)
    local_xyz = np.zeros((len(local_xy), 3), dtype=np.float32)
    local_xyz[:, :2] = local_xy
    world_xyz = pcd_trans(local_xyz, pose[:3], pose[3:7], False)
    return world_xyz[:, :2].astype(np.float32, copy=False)


def rasterize_points(points_xy: Array, resolution: float, min_points_per_cell: int) -> Tuple[Array, Array, Array]:
    if len(points_xy) == 0:
        return np.zeros((0, 0), dtype=bool), np.zeros(2, dtype=np.float32), np.zeros((0, 0), dtype=np.uint16)
    mins = points_xy.min(axis=0)
    ij = np.floor((points_xy - mins) / resolution).astype(np.int32)
    width = int(ij[:, 0].max()) + 1
    height = int(ij[:, 1].max()) + 1
    counts = np.zeros((height, width), dtype=np.uint16)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    mask = counts >= min_points_per_cell
    return mask, mins.astype(np.float32), counts


def binary_dilate(mask: Array, iterations: int = 1) -> Array:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        h, w = out.shape
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + h, dx:dx + w]
        out = expanded
    return out


def binary_erode(mask: Array, iterations: int = 1) -> Array:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        shrunk = np.ones_like(out, dtype=bool)
        h, w = out.shape
        for dy in range(3):
            for dx in range(3):
                shrunk &= padded[dy:dy + h, dx:dx + w]
        out = shrunk
    return out


def binary_close(mask: Array, iterations: int = 1) -> Array:
    if mask.size == 0:
        return mask
    return binary_erode(binary_dilate(mask, iterations), iterations)


def keep_large_components(mask: Array, min_size: int) -> Array:
    if mask.size == 0:
        return mask
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    occupied = np.argwhere(mask)
    for r0, c0 in occupied:
        if visited[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        visited[r0, c0] = True
        pixels = []
        while stack:
            r, c = stack.pop()
            pixels.append((r, c))
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if visited[nr, nc] or not mask[nr, nc]:
                    continue
                visited[nr, nc] = True
                stack.append((nr, nc))
        if len(pixels) < min_size:
            continue
        rr = np.fromiter((p[0] for p in pixels), dtype=np.int32)
        cc = np.fromiter((p[1] for p in pixels), dtype=np.int32)
        kept[rr, cc] = True
    return kept


def contiguous_runs(mask_row: Array) -> list:
    idx = np.flatnonzero(mask_row)
    if len(idx) == 0:
        return []
    runs = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def grid_to_xy(row: int, col: float, origin_xy: Array, resolution: float) -> Array:
    x = float(origin_xy[0] + (col + 0.5) * resolution)
    y = float(origin_xy[1] + (row + 0.5) * resolution)
    return np.array([x, y], dtype=np.float32)


def extract_edge_points(
    mask: Array,
    origin_xy: Array,
    resolution: float,
    row_step: int,
    min_width: float,
    max_width: float,
    max_jump: float,
) -> Tuple[Array, Array]:
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    left = []
    right = []
    prev_left = None
    prev_right = None
    rows = range(0, mask.shape[0], max(1, row_step))
    min_width_cells = max(1, int(np.floor(min_width / resolution)))
    max_width_cells = max(min_width_cells, int(np.ceil(max_width / resolution)))
    max_jump_cells = max_jump / resolution
    for row in rows:
        runs = contiguous_runs(mask[row])
        if not runs:
            continue
        left_col = None
        right_col = None
        if len(runs) >= 2:
            best_score = None
            best_pair = None
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    l_run = runs[i]
                    r_run = runs[j]
                    l_col = float(l_run[1])
                    r_col = float(r_run[0])
                    width_cells = r_col - l_col
                    if width_cells < min_width_cells or width_cells > max_width_cells:
                        continue
                    score = -(width_cells - (min_width_cells + max_width_cells) * 0.5) ** 2
                    if prev_left is not None:
                        score -= abs(l_col - prev_left)
                    if prev_right is not None:
                        score -= abs(r_col - prev_right)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pair = (l_col, r_col)
            if best_pair is not None:
                left_col, right_col = best_pair
        if left_col is None or right_col is None:
            cols = np.flatnonzero(mask[row])
            if len(cols) == 0:
                continue
            left_col = float(cols.min())
            right_col = float(cols.max())
            width_cells = right_col - left_col
            if width_cells < min_width_cells or width_cells > max_width_cells:
                continue
        if prev_left is not None and abs(left_col - prev_left) > max_jump_cells:
            continue
        if prev_right is not None and abs(right_col - prev_right) > max_jump_cells:
            continue
        left.append(grid_to_xy(row, left_col, origin_xy, resolution))
        right.append(grid_to_xy(row, right_col, origin_xy, resolution))
        prev_left = left_col
        prev_right = right_col
    left_arr = np.asarray(left, dtype=np.float32) if left else np.empty((0, 2), dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32) if right else np.empty((0, 2), dtype=np.float32)
    return left_arr, right_arr


def smooth_edge_points(points_xy: Array, window_size: int) -> Array:
    if len(points_xy) == 0:
        return points_xy
    if window_size < 2 or len(points_xy) < 3:
        return points_xy.astype(np.float32, copy=False)
    if window_size % 2 == 0:
        window_size += 1
    pts = points_xy[np.argsort(points_xy[:, 1])].astype(np.float32, copy=True)
    radius = window_size // 2
    padded_x = np.pad(pts[:, 0], (radius, radius), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    pts[:, 0] = np.convolve(padded_x, kernel, mode="valid")
    return pts


def select_main_cluster(
    edge_xy: Array,
    eps: float,
    min_samples: int,
    ref_point: Array = None,
) -> Array:
    if len(edge_xy) == 0:
        return edge_xy
    if len(edge_xy) < min_samples:
        return edge_xy
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(edge_xy)
    valid_labels = [lab for lab in sorted(set(labels)) if lab != -1]
    if not valid_labels:
        return edge_xy
    clusters = [(lab, edge_xy[labels == lab]) for lab in valid_labels]
    if ref_point is None or len(ref_point) == 0:
        clusters.sort(key=lambda item: len(item[1]), reverse=True)
        return clusters[0][1]
    ref = np.asarray(ref_point, dtype=np.float32)
    clusters.sort(key=lambda item: (np.linalg.norm(item[1][0] - ref), -len(item[1])))
    return clusters[0][1]


def is_valid_edge_segment(
    edge_xy: Array,
    min_points: int,
    min_length: float,
    max_x_range: float,
    max_x_std: float,
) -> bool:
    if len(edge_xy) < min_points:
        return False
    edge_xy = edge_xy[np.argsort(edge_xy[:, 1])]
    deltas = np.diff(edge_xy, axis=0)
    length = float(np.linalg.norm(deltas, axis=1).sum()) if len(deltas) > 0 else 0.0
    if length < min_length:
        return False
    x_range = float(edge_xy[:, 0].max() - edge_xy[:, 0].min())
    x_std = float(edge_xy[:, 0].std())
    if x_range > max_x_range:
        return False
    if x_std > max_x_std:
        return False
    return True


def merge_edge_tracks(
    track_xy: Array,
    edge_xy: Array,
    min_spacing: float,
    max_join_dist: float,
    max_heading_diff_deg: float,
) -> Array:
    if len(edge_xy) == 0:
        return track_xy
    edge_xy = edge_xy[np.argsort(edge_xy[:, 1])].astype(np.float32, copy=False)
    if len(track_xy) == 0:
        return edge_xy.copy()
    if len(track_xy) < 2 or len(edge_xy) < 2:
        return track_xy

    track_xy = track_xy[np.argsort(track_xy[:, 1])].astype(np.float32, copy=False)
    head_count = min(3, len(edge_xy))
    probe = edge_xy[:head_count]
    dists = np.linalg.norm(track_xy[None, :, :] - probe[:, None, :], axis=2)
    probe_nn = dists.min(axis=1)
    probe_idx = dists.argmin(axis=1)
    mean_join_dist = float(probe_nn.mean())
    if mean_join_dist > max_join_dist:
        return track_xy

    anchor_idx = int(np.median(probe_idx))
    if anchor_idx < 1:
        anchor_idx = 1
    if anchor_idx >= len(track_xy):
        anchor_idx = len(track_xy) - 1

    track_dir = track_xy[anchor_idx] - track_xy[anchor_idx - 1]
    edge_dir = edge_xy[1] - edge_xy[0]
    track_norm = float(np.linalg.norm(track_dir))
    edge_norm = float(np.linalg.norm(edge_dir))
    if track_norm < 1e-6 or edge_norm < 1e-6:
        return track_xy

    cosang = float(np.clip(np.dot(track_dir, edge_dir) / (track_norm * edge_norm), -1.0, 1.0))
    heading_diff_deg = float(np.degrees(np.arccos(cosang)))
    if heading_diff_deg > max_heading_diff_deg:
        return track_xy

    merged_prefix = track_xy[: anchor_idx + 1].astype(np.float32, copy=True).tolist()
    start_idx = 0
    anchor_point = np.asarray(merged_prefix[-1], dtype=np.float32)
    while start_idx < len(edge_xy) and np.linalg.norm(edge_xy[start_idx] - anchor_point) < min_spacing:
        start_idx += 1
    for pt in edge_xy[start_idx:]:
        p = np.asarray(pt, dtype=np.float32)
        if np.linalg.norm(p - np.asarray(merged_prefix[-1], dtype=np.float32)) >= min_spacing:
            merged_prefix.append(p.tolist())
    return np.asarray(merged_prefix, dtype=np.float32)


def save_mask(mask: Array, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.zeros((4, 4), dtype=np.uint8) if mask.size == 0 else mask.astype(np.uint8) * 255
    plt.imsave(path, image, cmap="gray", vmin=0, vmax=255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone road-edge extraction from all semantic point clouds")
    parser.add_argument("-c", "--config", default="config/outdoor_config.json", help="Config json path")
    parser.add_argument("-i", "--input", default=None, help="Input pickle path")
    parser.add_argument("-t", "--trajectory", default=None, help="Pose csv path")
    parser.add_argument("--road-class", type=int, default=None, help="Road semantic class id")
    parser.add_argument("--window", type=int, default=10, help="Sliding window size")
    parser.add_argument("--step", type=int, default=1, help="Process every N frames")
    parser.add_argument("--max-frames", type=int, default=9600, help="Maximum frames to read before processing")
    parser.add_argument("--box-forward", type=float, default=20.0, help="Forward crop range in local frame")
    parser.add_argument("--box-lateral", type=float, default=12.0, help="Lateral crop range in local frame")
    parser.add_argument("--box-z-min", type=float, default=-4.0, help="Minimum z in local frame")
    parser.add_argument("--box-z-max", type=float, default=2.0, help="Maximum z in local frame")
    parser.add_argument("--resolution", type=float, default=0.2, help="BEV grid resolution in meters")
    parser.add_argument("--min-points-per-cell", type=int, default=3, help="Minimum points per occupied BEV cell")
    parser.add_argument("--close-iter", type=int, default=2, help="Binary closing iterations")
    parser.add_argument("--min-component-cells", type=int, default=200, help="Minimum connected-component size in cells")
    parser.add_argument("--edge-row-step", type=int, default=2, help="Row step when sampling left/right edge points from the mask")
    parser.add_argument("--edge-min-width", type=float, default=2.5, help="Minimum road width for left/right edge selection")
    parser.add_argument("--edge-max-width", type=float, default=12.0, help="Maximum road width for left/right edge selection")
    parser.add_argument("--edge-max-jump", type=float, default=1.0, help="Maximum per-row lateral jump for edge continuity")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for local edge smoothing")
    parser.add_argument("--merge-min-spacing", type=float, default=0.3, help="Minimum spacing when appending points into merged edge tracks")
    parser.add_argument("--merge-max-join-dist", type=float, default=1.5, help="Maximum distance to join a new local edge segment to the merged track tail")
    parser.add_argument("--merge-max-heading-diff", type=float, default=35.0, help="Maximum heading difference in degrees to join a new local edge segment")
    parser.add_argument("--valid-min-points", type=int, default=8, help="Minimum point count for a local edge segment to be considered valid")
    parser.add_argument("--valid-min-length", type=float, default=3.0, help="Minimum polyline length for a local edge segment to be considered valid")
    parser.add_argument("--valid-max-x-range", type=float, default=8.0, help="Maximum lateral x range allowed for a local edge segment")
    parser.add_argument("--valid-max-x-std", type=float, default=2.5, help="Maximum lateral x standard deviation allowed for a local edge segment")
    parser.add_argument("--cluster-eps", type=float, default=0.8, help="DBSCAN eps for local edge clustering")
    parser.add_argument("--cluster-min-samples", type=int, default=4, help="DBSCAN min_samples for local edge clustering")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_path = args.input or os.path.join(config["save_folder"], f"{config['mode']}.pkl")
    pose_path = args.trajectory or os.path.join(config["save_folder"], "pose.csv")
    road_class = args.road_class if args.road_class is not None else config.get("road_class", 13)
    output_dir = args.output_dir or os.path.join(config["save_folder"], "road_demo")
    os.makedirs(output_dir, exist_ok=True)

    poses = np.loadtxt(pose_path, delimiter=",").astype(np.float32)
    road_frames = load_road_frames(input_path, road_class, min(args.max_frames, len(poses)))
    if not road_frames:
        raise ValueError(f"No frames were loaded from {input_path}")
    process_count = min(len(road_frames), len(poses))
    if process_count < args.window:
        raise ValueError(
            f"Loaded {process_count} frames, fewer than window={args.window}; reduce --window or provide more data"
        )

    last_local_points = np.empty((0, 3), dtype=np.float32)
    last_aligned_points = np.empty((0, 3), dtype=np.float32)
    last_mask = np.zeros((0, 0), dtype=bool)
    last_origin = np.zeros(2, dtype=np.float32)
    last_counts = np.zeros((0, 0), dtype=np.uint16)
    last_left_edge = np.empty((0, 2), dtype=np.float32)
    last_right_edge = np.empty((0, 2), dtype=np.float32)
    last_left_edge_smooth = np.empty((0, 2), dtype=np.float32)
    last_right_edge_smooth = np.empty((0, 2), dtype=np.float32)
    merged_left_track = np.empty((0, 2), dtype=np.float32)
    merged_right_track = np.empty((0, 2), dtype=np.float32)
    processed_frame = None
    heading_yaw = None

    window: Deque[Array] = deque(maxlen=args.window)
    for frame_idx in range(process_count):
        window.append(road_frames[frame_idx])
        if len(window) < args.window or frame_idx % args.step != 0:
            continue

        stacked_parts = [part for part in window if len(part) > 0]
        if not stacked_parts:
            continue

        pose = poses[frame_idx]
        heading_yaw = quat_to_yaw(pose[3:7])
        roads_world = np.vstack(stacked_parts)
        roads_local = pcd_trans(roads_world, pose[:3], pose[3:7], True)
        roads_local = crop_local_box(roads_local, args.box_forward, args.box_lateral, args.box_z_min, args.box_z_max)
        aligned_xy = rotate_xy(roads_local[:, :2], heading_yaw)
        aligned_local = np.column_stack((aligned_xy, roads_local[:, 2])) if len(roads_local) > 0 else np.empty((0, 3), dtype=np.float32)
        road_mask, origin_xy, counts = rasterize_points(aligned_xy, args.resolution, args.min_points_per_cell)
        road_mask = binary_close(road_mask, args.close_iter)
        road_mask = keep_large_components(road_mask, args.min_component_cells)
        left_edge, right_edge = extract_edge_points(
            road_mask,
            origin_xy,
            args.resolution,
            args.edge_row_step,
            args.edge_min_width,
            args.edge_max_width,
            args.edge_max_jump,
        )
        left_edge_smooth = smooth_edge_points(left_edge, args.smooth_window)
        right_edge_smooth = smooth_edge_points(right_edge, args.smooth_window)
        left_edge_smooth = select_main_cluster(left_edge_smooth, args.cluster_eps, args.cluster_min_samples)
        right_edge_smooth = select_main_cluster(right_edge_smooth, args.cluster_eps, args.cluster_min_samples)
        left_edge_smooth = left_edge_smooth[np.argsort(left_edge_smooth[:, 1])] if len(left_edge_smooth) > 0 else left_edge_smooth
        right_edge_smooth = right_edge_smooth[np.argsort(right_edge_smooth[:, 1])] if len(right_edge_smooth) > 0 else right_edge_smooth
        left_is_valid = is_valid_edge_segment(
            left_edge_smooth,
            args.valid_min_points,
            args.valid_min_length,
            args.valid_max_x_range,
            args.valid_max_x_std,
        )
        right_is_valid = is_valid_edge_segment(
            right_edge_smooth,
            args.valid_min_points,
            args.valid_min_length,
            args.valid_max_x_range,
            args.valid_max_x_std,
        )
        if not left_is_valid:
            left_edge_smooth = np.empty((0, 2), dtype=np.float32)
        if not right_is_valid:
            right_edge_smooth = np.empty((0, 2), dtype=np.float32)

        left_edge_world = aligned_edge_to_world(left_edge_smooth, heading_yaw, pose)
        right_edge_world = aligned_edge_to_world(right_edge_smooth, heading_yaw, pose)

        last_local_points = roads_local
        last_aligned_points = aligned_local
        last_mask = road_mask
        last_left_edge = left_edge
        last_right_edge = right_edge
        last_left_edge_smooth = left_edge_world
        last_right_edge_smooth = right_edge_world
        merged_left_track = merge_edge_tracks(
            merged_left_track,
            left_edge_world,
            args.merge_min_spacing,
            args.merge_max_join_dist,
            args.merge_max_heading_diff,
        )
        merged_right_track = merge_edge_tracks(
            merged_right_track,
            right_edge_world,
            args.merge_min_spacing,
            args.merge_max_join_dist,
            args.merge_max_heading_diff,
        )
        last_origin = origin_xy
        last_counts = counts
        processed_frame = frame_idx

    np.save(os.path.join(output_dir, "road_local_points.npy"), last_local_points)
    np.save(os.path.join(output_dir, "road_aligned_points.npy"), last_aligned_points)
    np.save(os.path.join(output_dir, "road_mask.npy"), last_mask.astype(np.uint8))
    np.save(os.path.join(output_dir, "road_counts.npy"), last_counts)
    np.save(os.path.join(output_dir, "left_edge_points.npy"), last_left_edge)
    np.save(os.path.join(output_dir, "right_edge_points.npy"), last_right_edge)
    np.save(os.path.join(output_dir, "left_edge_smooth.npy"), last_left_edge_smooth)
    np.save(os.path.join(output_dir, "right_edge_smooth.npy"), last_right_edge_smooth)
    np.save(os.path.join(output_dir, "left_edge_merged.npy"), merged_left_track)
    np.save(os.path.join(output_dir, "right_edge_merged.npy"), merged_right_track)
    save_mask(last_mask, os.path.join(output_dir, "road_mask.png"))

    with open(os.path.join(output_dir, "road_grid.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": os.path.abspath(input_path),
                "pose": os.path.abspath(pose_path),
                "processed_frame": processed_frame,
                "heading_yaw_rad": None if heading_yaw is None else float(heading_yaw),
                "heading_yaw_deg": None if heading_yaw is None else float(np.degrees(heading_yaw)),
                "rotation_applied": True,
                "road_class": int(road_class),
                "window": int(args.window),
                "step": int(args.step),
                "max_frames": int(args.max_frames),
                "box": {
                    "forward": float(args.box_forward),
                    "lateral": float(args.box_lateral),
                    "z_min": float(args.box_z_min),
                    "z_max": float(args.box_z_max),
                },
                "resolution": float(args.resolution),
                "min_points_per_cell": int(args.min_points_per_cell),
                "close_iter": int(args.close_iter),
                "min_component_cells": int(args.min_component_cells),
                "edge_row_step": int(args.edge_row_step),
                "edge_min_width": float(args.edge_min_width),
                "edge_max_width": float(args.edge_max_width),
                "edge_max_jump": float(args.edge_max_jump),
                "smooth_window": int(args.smooth_window),
                "merge_min_spacing": float(args.merge_min_spacing),
                "merge_max_join_dist": float(args.merge_max_join_dist),
                "merge_max_heading_diff": float(args.merge_max_heading_diff),
                "valid_min_points": int(args.valid_min_points),
                "valid_min_length": float(args.valid_min_length),
                "valid_max_x_range": float(args.valid_max_x_range),
                "valid_max_x_std": float(args.valid_max_x_std),
                "cluster_eps": float(args.cluster_eps),
                "cluster_min_samples": int(args.cluster_min_samples),
                "local_point_count": int(len(last_local_points)),
                "left_edge_count": int(len(last_left_edge)),
                "right_edge_count": int(len(last_right_edge)),
                "left_edge_smooth_count": int(len(last_left_edge_smooth)),
                "right_edge_smooth_count": int(len(last_right_edge_smooth)),
                "left_edge_merged_count": int(len(merged_left_track)),
                "right_edge_merged_count": int(len(merged_right_track)),
                "grid_shape": list(last_mask.shape),
                "origin_xy": last_origin.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved road local points to {os.path.join(output_dir, 'road_local_points.npy')}")
    print(f"Saved aligned road points to {os.path.join(output_dir, 'road_aligned_points.npy')}")
    print(f"Saved left edge points to {os.path.join(output_dir, 'left_edge_points.npy')}")
    print(f"Saved right edge points to {os.path.join(output_dir, 'right_edge_points.npy')}")
    print(f"Saved smoothed left edge to {os.path.join(output_dir, 'left_edge_smooth.npy')}")
    print(f"Saved smoothed right edge to {os.path.join(output_dir, 'right_edge_smooth.npy')}")
    print(f"Saved merged left edge to {os.path.join(output_dir, 'left_edge_merged.npy')}")
    print(f"Saved merged right edge to {os.path.join(output_dir, 'right_edge_merged.npy')}")
    print(f"Saved road mask to {os.path.join(output_dir, 'road_mask.png')}")
    print(f"Saved road grid metadata to {os.path.join(output_dir, 'road_grid.json')}")


if __name__ == "__main__":
    main()
