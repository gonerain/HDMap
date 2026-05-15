#!/usr/bin/python3
"""Concatenate `outdoor.pkl` into a single semantic point cloud written
as a colored PCD (RGBA per class via the Mapillary Vistas cmap).

Optional flags let you trim before writing:
  --keep-classes 13 15 23     keep only road/sidewalk/crosswalk
  --voxel-size 0.1            voxel-grid downsample to ~0.1 m
  --max-points 5_000_000      hard cap (random subsample)

Output is the standard PCL-PCD format and opens in `pcl_viewer`,
CloudCompare, MeshLab, or as a `sensor_msgs/PointCloud2` in rviz once
loaded with `rosrun pcl_ros pcd_to_pointcloud`.

Must run inside the rospytorch container (uses pclpy via util.save_nppc).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


# Inlined Mapillary Vistas label colormap (copied from predict.py) so this
# tool runs even without torch / detectron2 / pclpy installed.
MAPILLARY_COLORMAP = np.asarray([
    [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
    [180, 165, 180], [102, 102, 156], [102, 102, 156], [128, 64, 255],
    [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
    [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
    [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
    [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128],
    [255, 255, 255], [64, 170, 64], [128, 64, 64], [70, 130, 180],
    [255, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
    [255, 255, 128], [250, 0, 30], [0, 0, 0], [220, 220, 220],
    [170, 170, 170], [222, 40, 40], [100, 170, 30], [40, 40, 40],
    [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170],
    [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 142],
    [250, 170, 30], [192, 192, 192], [220, 220, 0], [180, 165, 180],
    [119, 11, 32], [0, 0, 142], [0, 60, 100], [0, 0, 142],
    [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
    [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32],
    [0, 0, 0], [0, 0, 0],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default="result/outdoor_full/outdoor.pkl")
    parser.add_argument("--output", default="result/outdoor_full/semantic.pcd")
    parser.add_argument("--cmap", default="mapillary")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel-grid downsample edge length in meters (0 disables).",
    )
    parser.add_argument(
        "--keep-classes",
        type=int,
        nargs="+",
        default=None,
        help="If set, keep only these class ids.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Hard cap (random subsample). 0 disables.",
    )
    parser.add_argument(
        "--xyz-only",
        action="store_true",
        help="Write a plain XYZ PCD (no colors). Useful for tools that don't read RGBA.",
    )
    parser.add_argument(
        "--single-frame",
        type=int,
        default=None,
        help=(
            "1-based frame index. If set, dump only that frame's points "
            "(no cross-frame accumulation). Avoids the 'thick road/sidewalk' "
            "artifact caused by IE pose drift + lidar noise compounding "
            "across frames."
        ),
    )
    parser.add_argument(
        "--per-frame-dir",
        default=None,
        help=(
            "If set, write one PCD per frame into this directory "
            "(<dir>/frame_NNNNNN.pcd) instead of a single concatenated PCD. "
            "Mutually exclusive with --single-frame."
        ),
    )
    return parser.parse_args()


def save_pcd_xyzrgb_binary(points_xyzc, color_lut, fname):
    """Write a binary PCD file with x/y/z (float32) + rgb (PCL packed-float).

    PCL encodes RGB as a single float whose bits are an aligned uint32
    (R<<16) | (G<<8) | B. CloudCompare, pcl_viewer and rviz all decode
    this format correctly. No pclpy dependency, so it runs even when the
    docker image's laspy chain is broken.
    """
    pts = np.asarray(points_xyzc, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"expected (N, 4) [x, y, z, class] array, got {pts.shape}")

    classes = pts[:, 3].astype(np.int64)
    table = np.asarray(color_lut, dtype=np.uint32)
    safe = np.clip(classes, 0, len(table) - 1)
    rgb = table[safe]                                                # (N, 3) in 0..255
    rgb_int = ((rgb[:, 0] & 0xFF) << 16) | ((rgb[:, 1] & 0xFF) << 8) | (rgb[:, 2] & 0xFF)
    rgb_int = rgb_int.astype(np.uint32)
    # bit-cast to float32 — required by PCL's "rgb" field convention
    rgb_float = rgb_int.view(np.float32)

    n = len(pts)
    header = (
        "# .PCD v0.7 - HDMap semantic export\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )

    out = np.empty((n, 4), dtype=np.float32)
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 1]
    out[:, 2] = pts[:, 2]
    out[:, 3] = rgb_float
    with open(fname, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes(order="C"))


def save_pcd_xyz_intensity_binary(points_xyzc, fname):
    """Plain XYZ + intensity (class id) PCD, no color lookup."""
    pts = np.asarray(points_xyzc, dtype=np.float32)
    n = len(pts)
    header = (
        "# .PCD v0.7 - HDMap semantic export (intensity = class id)\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    out = np.empty((n, 4), dtype=np.float32)
    out[:, :3] = pts[:, :3]
    out[:, 3] = pts[:, 3]
    with open(fname, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes(order="C"))


def voxel_downsample_first_hit(points_xyzc, voxel):
    """Keep the first point per voxel — fast and produces a clean dense
    output without the floating-mean artifacts of true VG averaging.
    """
    v = float(voxel)
    if v <= 0.0:
        return points_xyzc
    keys = np.floor(points_xyzc[:, :3] / v).astype(np.int64)
    offset = -keys.min(axis=0)
    keys += offset
    rng = keys.max(axis=0) + 1
    encoded = (keys[:, 0] * rng[1] + keys[:, 1]) * rng[2] + keys[:, 2]
    sort_order = np.argsort(encoded)
    encoded_sorted = encoded[sort_order]
    first_mask = np.concatenate([[True], np.diff(encoded_sorted) != 0])
    keep_idx = sort_order[first_mask]
    return points_xyzc[keep_idx]


def main():
    args = parse_args()
    color_classes = None
    if args.cmap and not args.xyz_only:
        # Inline copy of predict.create_mapillary_vistas_label_colormap
        # to avoid the pclpy/laspy/requests import chain that's broken in
        # this docker image.
        if args.cmap == "mapillary":
            color_classes = MAPILLARY_COLORMAP
        else:
            sys.path.insert(0, ".")
            from predict import get_colors
            color_classes = get_colors(args.cmap)

    if args.single_frame is not None and args.per_frame_dir:
        raise SystemExit("--single-frame and --per-frame-dir are mutually exclusive")

    print(f"loading {args.pkl} ...")
    arrs = []
    with open(args.pkl, "rb") as f:
        while True:
            try:
                arrs.append(pickle.load(f))
            except EOFError:
                break
    if not arrs:
        raise RuntimeError(f"empty pkl: {args.pkl}")
    print(f"  loaded {len(arrs)} frames")

    def _process_and_write(points_xyzc, out_path):
        before = len(points_xyzc)
        if args.keep_classes:
            keep = np.in1d(points_xyzc[:, 3].astype(np.int64), args.keep_classes)
            points_xyzc = points_xyzc[keep]
        if args.voxel_size > 0.0:
            points_xyzc = voxel_downsample_first_hit(points_xyzc, args.voxel_size)
        if args.max_points > 0 and len(points_xyzc) > args.max_points:
            idx = np.random.default_rng(0).choice(len(points_xyzc), size=args.max_points, replace=False)
            points_xyzc = points_xyzc[idx]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.xyz_only:
            save_pcd_xyz_intensity_binary(points_xyzc, str(out_path))
        else:
            save_pcd_xyzrgb_binary(points_xyzc, color_classes, str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"wrote {out_path}  ({before:,} -> {len(points_xyzc):,} points, {size_mb:.1f} MB)")
        return points_xyzc

    if args.per_frame_dir:
        out_dir = Path(args.per_frame_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kept_total = 0
        for i, arr in enumerate(arrs, start=1):
            if len(arr) == 0:
                continue
            pts = np.asarray(arr, dtype=np.float32)
            out_path = out_dir / f"frame_{i:06d}.pcd"
            kept = _process_and_write(pts, out_path)
            kept_total += len(kept)
        print(f"\nper-frame export complete: {len(arrs)} frames, {kept_total:,} points total in {out_dir}")
        return

    if args.single_frame is not None:
        idx = int(args.single_frame) - 1
        if not (0 <= idx < len(arrs)):
            raise SystemExit(f"--single-frame {args.single_frame} out of range [1, {len(arrs)}]")
        if len(arrs[idx]) == 0:
            raise SystemExit(f"frame {args.single_frame} is empty in the pkl")
        points = np.asarray(arrs[idx], dtype=np.float32)
        print(f"  single-frame mode: frame {args.single_frame} -> {len(points):,} points (no cross-frame stacking)")
    else:
        points = np.vstack(arrs).astype(np.float32, copy=False)
        print(f"  combined: {len(points):,} points across all frames")

    output = Path(args.output)
    points = _process_and_write(points, output)

    if not args.xyz_only:
        ids, counts = np.unique(points[:, 3].astype(np.int64), return_counts=True)
        order = np.argsort(-counts)
        print("top classes:")
        for i in order[:10]:
            print(f"  class {int(ids[i]):3d}: {int(counts[i]):,}")


if __name__ == "__main__":
    main()
