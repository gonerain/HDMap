#!/usr/bin/python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection import render_outdoor_projection_overlay
from src.projection import summarize_aligned_range


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render sidewalk semantic point-cloud overlays on raw images for a frame range."
    )
    parser.add_argument("-i", "--input", default="result/outdoor/outdoor.pkl", help="Path to outdoor.pkl.")
    parser.add_argument("-c", "--config", default="config/outdoor_config.json", help="Projection config path.")
    parser.add_argument("--trajectory", default="result/outdoor/pose.csv", help="Path to pose.csv.")
    parser.add_argument("--images-dir", default="result/outdoor/originpics", help="Directory containing raw images.")
    parser.add_argument("--frame-start", type=int, default=0, help="0-based start frame.")
    parser.add_argument("--frame-end", type=int, default=2000, help="0-based inclusive end frame.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step.")
    parser.add_argument(
        "--output-dir",
        default="demo_output/sidewalk_projection_0000_2000",
        help="Directory used to store rendered overlay images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_start = int(args.frame_start)
    frame_end = int(args.frame_end)
    frame_step = max(int(args.frame_step), 1)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if frame_start < 0:
        raise ValueError(f"frame_start must be >= 0, got {frame_start}")
    if frame_end < frame_start:
        raise ValueError(f"frame_end must be >= frame_start, got {frame_end} < {frame_start}")

    aligned = summarize_aligned_range(args.input, args.trajectory, args.images_dir, one_based_images=True)
    if frame_end >= aligned["aligned_frames"]:
        raise IndexError(
            f"frame_end {frame_end} exceeds aligned frame range {aligned['aligned_frames'] - 1}"
        )

    frame_reports = []
    frame_indices = list(range(frame_start, frame_end + 1, frame_step))
    for idx, frame_index in enumerate(frame_indices, start=1):
        output_path = output_dir / f"frame_{frame_index:04d}_sidewalk_overlay.png"
        rendered = render_outdoor_projection_overlay(
            args.input,
            frame_index,
            config_path=args.config,
            trajectory_path=args.trajectory,
            images_dir=args.images_dir,
            output_path=output_path,
            semantic_filter="sidewalk",
        )
        projection = rendered["projection"]
        frame_reports.append(
            {
                "frame_index": int(frame_index),
                "image_path": rendered["image_path"],
                "output_path": rendered["output_path"],
                "original_point_count": rendered["original_point_count"],
                "filtered_point_count": rendered["filtered_point_count"],
                "points_in_bounds": int(projection["in_bounds_mask"].sum()),
            }
        )
        if idx % 100 == 0 or idx == len(frame_indices):
            print(f"[{idx}/{len(frame_indices)}] rendered frame {frame_index}", file=sys.stderr)

    report = {
        "input": str(Path(args.input)),
        "config": str(Path(args.config)),
        "trajectory": str(Path(args.trajectory)),
        "images_dir": str(Path(args.images_dir)),
        "semantic_filter": "sidewalk",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
        "output_dir": str(output_dir),
        "aligned_range": aligned,
        "frame_count": len(frame_reports),
        "frames": frame_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path), "frame_count": len(frame_reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
