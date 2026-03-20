#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


Array = np.ndarray
Point2 = Tuple[float, float]


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


def quat_to_rot(q: Sequence[float]) -> Array:
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def world_to_local(points_xyz: Array, pose: Array) -> Array:
    t = pose[:3].astype(np.float32)
    r = quat_to_rot(pose[3:7])
    return (points_xyz - t) @ r


def local_to_world(points_xyz: Array, pose: Array) -> Array:
    t = pose[:3].astype(np.float32)
    r = quat_to_rot(pose[3:7])
    return points_xyz @ r.T + t


def contiguous_runs(mask: Array) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs: List[Tuple[int, int]] = []
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


def edge_candidates_from_hist(xs: Array, bin_size: float, min_run_bins: int, quantile: float) -> List[Dict[str, float]]:
    if len(xs) == 0:
        return []
    xmin = float(xs.min())
    xmax = float(xs.max())
    nbins = max(1, int(np.ceil((xmax - xmin) / bin_size)) + 1)
    hist, edges = np.histogram(xs, bins=nbins, range=(xmin, xmax + bin_size))
    runs = contiguous_runs(hist > 0)
    candidates: List[Dict[str, float]] = []
    for start, end in runs:
        if end - start + 1 < min_run_bins:
            continue
        lo = float(edges[start])
        hi = float(edges[end + 1])
        cluster = xs[(xs >= lo) & (xs <= hi)]
        if len(cluster) == 0:
            continue
        candidates.append(
            {
                "x": float(np.quantile(cluster, quantile)),
                "lo": lo,
                "hi": hi,
                "support": float(len(cluster)),
                "span": hi - lo,
            }
        )
    candidates.sort(key=lambda item: item["x"])
    return candidates


def choose_edge_pair(
    left_candidates: List[Dict[str, float]],
    right_candidates: List[Dict[str, float]],
    prev_left: Optional[Point2],
    prev_right: Optional[Point2],
    prev_width: Optional[float],
    min_width: float,
    max_width: float,
    max_edge_jump: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    best_pair: Optional[Tuple[float, float, float]] = None
    best_score: Optional[float] = None
    for left in left_candidates:
        for right in right_candidates:
            left_x = left["x"]
            right_x = right["x"]
            width = right_x - left_x
            if width < min_width or width > max_width:
                continue
            if prev_left is not None and abs(left_x - prev_left[0]) > max_edge_jump * 2.0:
                continue
            if prev_right is not None and abs(right_x - prev_right[0]) > max_edge_jump * 2.0:
                continue
            score = left["support"] + right["support"]
            score -= 0.2 * (left["span"] + right["span"])
            if prev_left is not None:
                score -= 4.0 * abs(left_x - prev_left[0])
            if prev_right is not None:
                score -= 4.0 * abs(right_x - prev_right[0])
            if prev_width is not None:
                score -= 2.0 * abs(width - prev_width)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (left_x, right_x, width)
    if best_pair is None:
        return None, None, prev_width
    return best_pair


def extract_edges_from_local(
    local_xyz: Array,
    forward_range: Tuple[float, float],
    lateral_limit: float,
    z_range: Tuple[float, float],
    slice_step: float,
    min_slice_points: int,
    edge_bin_size: float,
    min_run_bins: int,
    left_quantile: float,
    right_quantile: float,
    min_width: float,
    max_width: float,
    max_edge_jump: float,
) -> Tuple[List[Point2], List[Point2]]:
    x = local_xyz[:, 0]
    y = local_xyz[:, 1]
    z = local_xyz[:, 2]
    roi = (
        (y >= forward_range[0])
        & (y <= forward_range[1])
        & (np.abs(x) <= lateral_limit)
        & (z >= z_range[0])
        & (z <= z_range[1])
    )
    pts = local_xyz[roi]
    if len(pts) == 0:
        return [], []

    left_points: List[Point2] = []
    right_points: List[Point2] = []
    y_edges = np.arange(forward_range[0], forward_range[1] + slice_step, slice_step, dtype=np.float32)
    prev_left = None
    prev_right = None
    prev_width = None

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        sl = pts[(pts[:, 1] >= y0) & (pts[:, 1] < y1)]
        if len(sl) < min_slice_points:
            continue
        xs = sl[:, 0]
        yc = float(sl[:, 1].mean())
        left_candidates = edge_candidates_from_hist(xs, edge_bin_size, min_run_bins, left_quantile)
        right_candidates = edge_candidates_from_hist(xs, edge_bin_size, min_run_bins, right_quantile)
        left_x, right_x, width = choose_edge_pair(
            left_candidates=left_candidates,
            right_candidates=right_candidates,
            prev_left=prev_left,
            prev_right=prev_right,
            prev_width=prev_width,
            min_width=min_width,
            max_width=max_width,
            max_edge_jump=max_edge_jump,
        )
        if left_x is None or right_x is None:
            continue

        if prev_left is not None and abs(left_x - prev_left[0]) > max_edge_jump:
            left_x = None
        if prev_right is not None and abs(right_x - prev_right[0]) > max_edge_jump:
            right_x = None

        if left_x is not None:
            lp = (left_x, yc)
            left_points.append(lp)
            prev_left = lp
        if right_x is not None:
            rp = (right_x, yc)
            right_points.append(rp)
            prev_right = rp
        if left_x is not None and right_x is not None:
            prev_width = width

    return left_points, right_points


def rdp(points: Sequence[Point2], epsilon: float) -> List[Point2]:
    if len(points) <= 2:
        return list(points)
    pts = np.asarray(points, dtype=np.float64)
    start = pts[0]
    end = pts[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        dists = np.linalg.norm(pts - start, axis=1)
    else:
        vecs = pts - start
        dists = np.abs(line[0] * vecs[:, 1] - line[1] * vecs[:, 0]) / norm
    idx = int(np.argmax(dists))
    if float(dists[idx]) <= epsilon:
        return [tuple(pts[0]), tuple(pts[-1])]
    left = rdp([tuple(p) for p in pts[: idx + 1]], epsilon)
    right = rdp([tuple(p) for p in pts[idx:]], epsilon)
    return left[:-1] + right


def to_world_polyline(local_points: List[Point2], pose: Array) -> Array:
    if not local_points:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.array([[x, y, 0.0] for x, y in local_points], dtype=np.float32)
    return local_to_world(arr, pose)


def dedupe_polyline(points_xy: Array, min_spacing: float) -> Array:
    if len(points_xy) <= 1:
        return points_xy.astype(np.float32, copy=False)
    kept = [points_xy[0]]
    for pt in points_xy[1:]:
        if np.linalg.norm(pt - kept[-1]) >= min_spacing:
            kept.append(pt)
    return np.asarray(kept, dtype=np.float32)


def merge_world_segments(segments: List[Array], merge_dist: float, smooth_window: int) -> Array:
    if not segments:
        return np.empty((0, 3), dtype=np.float32)
    track_xy: List[Array] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        seg_xy = dedupe_polyline(segment[:, :2].astype(np.float32, copy=False), max(merge_dist * 0.5, 1e-3))
        if len(seg_xy) < 2:
            continue
        if not track_xy:
            track_xy.extend(seg_xy)
            continue
        track_arr = np.asarray(track_xy, dtype=np.float32)
        dists = np.linalg.norm(seg_xy[:, None, :] - track_arr[None, :, :], axis=2)
        nearest = dists.min(axis=1)
        overlap = nearest <= merge_dist
        start = int(np.max(np.flatnonzero(overlap)) + 1) if np.any(overlap) else 0
        for pt in seg_xy[start:]:
            if np.linalg.norm(pt - track_xy[-1]) >= max(merge_dist * 0.5, 1e-3):
                track_xy.append(pt)
    if not track_xy:
        return np.empty((0, 3), dtype=np.float32)
    merged_xy = np.asarray(track_xy, dtype=np.float32)
    if smooth_window >= 3 and len(merged_xy) >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        radius = smooth_window // 2
        padded = np.pad(merged_xy, ((radius, radius), (0, 0)), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        merged_xy = np.stack(
            [
                np.convolve(padded[:, 0], kernel, mode="valid"),
                np.convolve(padded[:, 1], kernel, mode="valid"),
            ],
            axis=1,
        )
    merged = np.zeros((len(merged_xy), 3), dtype=np.float32)
    merged[:, :2] = merged_xy
    return merged


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_preview(path: str, left_edges: List[Array], right_edges: List[Array], poses: Array) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 8))
    if len(poses) > 0:
        plt.plot(poses[:, 0], poses[:, 1], color="black", linewidth=1, label="trajectory")
    for i, edge in enumerate(left_edges):
        if len(edge) >= 2:
            plt.plot(edge[:, 0], edge[:, 1], color="tab:blue", linewidth=1.5, alpha=0.9, label="left_edge" if i == 0 else None)
    for i, edge in enumerate(right_edges):
        if len(edge) >= 2:
            plt.plot(edge[:, 0], edge[:, 1], color="tab:red", linewidth=1.5, alpha=0.9, label="right_edge" if i == 0 else None)
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search road left/right edges with a sliding local window")
    parser.add_argument("--input", default="result/outdoor/outdoor.pkl", help="Semantic outdoor.pkl path")
    parser.add_argument("--pose", default="result/outdoor/pose.csv", help="Pose csv path")
    parser.add_argument("--road-class", type=int, default=13, help="Road class id")
    parser.add_argument("--window", type=int, default=10, help="Sliding window size in frames")
    parser.add_argument("--step", type=int, default=3, help="Process every N frames")
    parser.add_argument("--forward-min", type=float, default=0.0, help="Local forward ROI min y")
    parser.add_argument("--forward-max", type=float, default=15.0, help="Local forward ROI max y")
    parser.add_argument("--lateral-limit", type=float, default=12.0, help="Local lateral ROI abs(x) limit")
    parser.add_argument("--z-min", type=float, default=-4.0, help="Local z ROI min")
    parser.add_argument("--z-max", type=float, default=2.0, help="Local z ROI max")
    parser.add_argument("--slice-step", type=float, default=0.5, help="Forward slice size in meters")
    parser.add_argument("--min-slice-points", type=int, default=30, help="Minimum points in one slice")
    parser.add_argument("--edge-bin-size", type=float, default=0.2, help="Histogram bin size for lateral edge search")
    parser.add_argument("--min-run-bins", type=int, default=2, help="Minimum consecutive occupied x bins for a stable edge")
    parser.add_argument("--left-quantile", type=float, default=0.15, help="Quantile inside the left stable cluster")
    parser.add_argument("--right-quantile", type=float, default=0.85, help="Quantile inside the right stable cluster")
    parser.add_argument("--min-width", type=float, default=2.5, help="Minimum valid road width in one slice")
    parser.add_argument("--max-width", type=float, default=20.0, help="Maximum valid road width in one slice")
    parser.add_argument("--max-edge-jump", type=float, default=1.5, help="Maximum lateral jump allowed between adjacent slices")
    parser.add_argument("--simplify", type=float, default=0.3, help="RDP epsilon in meters")
    parser.add_argument("--merge-dist", type=float, default=0.75, help="Merge radius for stitching local edge segments")
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd moving-average window for merged global edges")
    parser.add_argument("--output-dir", default="demo_output/road_edge", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    poses = np.loadtxt(args.pose, delimiter=",").astype(np.float32)
    frame_iter = iter_pickled_arrays(args.input)
    window: Deque[Array] = deque(maxlen=args.window)
    left_world_edges: List[Array] = []
    right_world_edges: List[Array] = []
    frame_records: List[Dict] = []

    processed = 0
    for frame_idx, frame in enumerate(frame_iter):
        road = frame[frame[:, 3].astype(np.int32) == args.road_class][:, :3]
        window.append(road)
        if len(window) < args.window:
            continue
        if frame_idx >= len(poses):
            break
        if frame_idx % args.step != 0:
            continue
        if sum(len(x) for x in window) == 0:
            continue

        pose = poses[frame_idx]
        stacked = np.vstack([x for x in window if len(x) > 0])
        local = world_to_local(stacked, pose)
        left_local, right_local = extract_edges_from_local(
            local_xyz=local,
            forward_range=(args.forward_min, args.forward_max),
            lateral_limit=args.lateral_limit,
            z_range=(args.z_min, args.z_max),
            slice_step=args.slice_step,
            min_slice_points=args.min_slice_points,
            edge_bin_size=args.edge_bin_size,
            min_run_bins=args.min_run_bins,
            left_quantile=args.left_quantile,
            right_quantile=args.right_quantile,
            min_width=args.min_width,
            max_width=args.max_width,
            max_edge_jump=args.max_edge_jump,
        )

        if len(left_local) >= 2:
            left_local = rdp(left_local, args.simplify)
            left_world = to_world_polyline(left_local, pose)
            left_world_edges.append(left_world)
        else:
            left_world = np.empty((0, 3), dtype=np.float32)

        if len(right_local) >= 2:
            right_local = rdp(right_local, args.simplify)
            right_world = to_world_polyline(right_local, pose)
            right_world_edges.append(right_world)
        else:
            right_world = np.empty((0, 3), dtype=np.float32)

        frame_records.append(
            {
                "frame": frame_idx,
                "pose": pose.tolist(),
                "left_edge": left_world[:, :2].tolist(),
                "right_edge": right_world[:, :2].tolist(),
                "window_points": int(len(stacked)),
            }
        )
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} local windows at frame {frame_idx}")

    left_points = merge_world_segments(left_world_edges, args.merge_dist, args.smooth_window)
    right_points = merge_world_segments(right_world_edges, args.merge_dist, args.smooth_window)

    np.save(os.path.join(args.output_dir, "left_edge_points.npy"), left_points)
    np.save(os.path.join(args.output_dir, "right_edge_points.npy"), right_points)
    save_json(
        os.path.join(args.output_dir, "road_edges.json"),
        {
            "input": os.path.abspath(args.input),
            "pose": os.path.abspath(args.pose),
            "road_class": args.road_class,
            "window": args.window,
            "step": args.step,
            "forward_range": [args.forward_min, args.forward_max],
            "lateral_limit": args.lateral_limit,
            "slice_step": args.slice_step,
            "merge_dist": args.merge_dist,
            "smooth_window": args.smooth_window,
            "global_left_edge": left_points[:, :2].tolist(),
            "global_right_edge": right_points[:, :2].tolist(),
            "frames": frame_records,
        },
    )
    save_preview(
        os.path.join(args.output_dir, "road_edges_preview.png"),
        [left_points] if len(left_points) >= 2 else [],
        [right_points] if len(right_points) >= 2 else [],
        poses,
    )

    print(f"Saved left edge points to {os.path.join(args.output_dir, 'left_edge_points.npy')}")
    print(f"Saved right edge points to {os.path.join(args.output_dir, 'right_edge_points.npy')}")
    print(f"Saved edge polylines to {os.path.join(args.output_dir, 'road_edges.json')}")
    print(f"Saved preview to {os.path.join(args.output_dir, 'road_edges_preview.png')}")


if __name__ == "__main__":
    main()
