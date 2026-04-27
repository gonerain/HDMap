#!/usr/bin/python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.pkl_frame_loader import load_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Export one frame or all frames from outdoor.pkl to ASCII PCD.")
    parser.add_argument(
        "-i",
        "--input",
        default="result/outdoor/outdoor.pkl",
        help="Path to semantic pkl, e.g. result/outdoor/outdoor.pkl",
    )
    parser.add_argument(
        "-f",
        "--frame-index",
        type=int,
        default=0,
        help="0-based frame index to export. Ignored when --all-frames is set.",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --frame-index as 1-based.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="demo_output/pkl_to_pcd/frame_000000.pcd",
        help="Output PCD path for single-frame export, or output directory for --all-frames.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Export every frame in the pkl to one PCD per frame.",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge all frames into one full PCD.",
    )
    parser.add_argument(
        "--drop-semantic",
        action="store_true",
        help="Write xyz only. By default writes xyz plus semantic as intensity.",
    )
    return parser.parse_args()


def count_stream_frames(input_path):
    frame_count = 0
    with Path(input_path).open("rb") as f:
        while True:
            try:
                _ = pickle.load(f)
            except EOFError:
                break
            frame_count += 1
    return frame_count


def ensure_frame_array(frame):
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"frame must have shape (N, >=3), got {arr.shape}")
    return arr


def write_ascii_pcd(points, output_path, drop_semantic=False):
    points = ensure_frame_array(points)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = points[:, :3]
    semantic = None if drop_semantic or points.shape[1] < 4 else points[:, 3]

    if semantic is None:
        header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {len(xyz)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(xyz)}",
            "DATA ascii",
        ]
        body = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in xyz]
    else:
        header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {len(xyz)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(xyz)}",
            "DATA ascii",
        ]
        body = [f"{x:.6f} {y:.6f} {z:.6f} {c:.6f}" for (x, y, z), c in zip(xyz, semantic)]

    output_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return output_path


def export_single_frame(args):
    frame_index = args.frame_index - 1 if args.one_based else args.frame_index
    frame, _resolved_mode = load_frame(args.input, frame_index, mode="auto")
    output_path = write_ascii_pcd(frame, args.output, drop_semantic=args.drop_semantic)
    print(f"saved frame {frame_index} to {output_path}")


def export_all_frames(args):
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = count_stream_frames(input_path)
    for frame_index in range(total):
        frame, _resolved_mode = load_frame(input_path, frame_index, mode="auto")
        output_path = output_dir / f"frame_{frame_index:06d}.pcd"
        write_ascii_pcd(frame, output_path, drop_semantic=args.drop_semantic)
    print(f"saved {total} frames to {output_dir}")


def export_merged_frames(args):
    input_path = Path(args.input)
    total = count_stream_frames(input_path)
    merged = []
    for frame_index in range(total):
        frame, _resolved_mode = load_frame(input_path, frame_index, mode="auto")
        arr = ensure_frame_array(frame)
        if len(arr) != 0:
            merged.append(arr)
    if merged:
        merged_points = np.vstack(merged)
    else:
        merged_points = np.zeros((0, 4), dtype=np.float64)
    output_path = write_ascii_pcd(merged_points, args.output, drop_semantic=args.drop_semantic)
    print(f"saved merged {total} frames ({len(merged_points)} points) to {output_path}")


def main():
    args = parse_args()
    if args.all_frames and args.merge_all:
        raise SystemExit("--all-frames and --merge-all are mutually exclusive")
    if args.merge_all:
        export_merged_frames(args)
        return
    if args.all_frames:
        export_all_frames(args)
        return
    export_single_frame(args)


if __name__ == "__main__":
    main()
