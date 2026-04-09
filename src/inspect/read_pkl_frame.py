#!/usr/bin/python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.demo_paths import demo_output_path
from src.io.pkl_frame_loader import load_frame
from src.io.pkl_frame_loader import summarize_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Load any logical frame from an indoor/outdoor semantic pkl.")
    parser.add_argument(
        "-i",
        "--input",
        default="result/outdoor/outdoor.pkl",
        help="Path to the semantic pkl, e.g. result/outdoor/outdoor.pkl",
    )
    parser.add_argument(
        "-f",
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to load. Defaults to 0-based indexing.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "outdoor", "indoor"],
        default="auto",
        help="How the pkl is organized. auto will inspect the first object.",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --frame-index as 1-based.",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Also save a JSON summary under demo_output/pkl_reader/.",
    )
    parser.add_argument(
        "--print-points",
        type=int,
        default=5,
        help="How many leading points to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_index = args.frame_index - 1 if args.one_based else args.frame_index
    frame, resolved_mode = load_frame(args.input, frame_index, mode=args.mode)
    summary = summarize_frame(frame)
    summary["input"] = str(Path(args.input))
    summary["resolved_mode"] = resolved_mode
    summary["frame_index"] = int(frame_index)
    summary["preview_points"] = frame[: max(int(args.print_points), 0)].tolist()

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_summary:
        output_path = demo_output_path("pkl_reader", f"frame_{frame_index:06d}_summary.json")
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved summary to {output_path}")


if __name__ == "__main__":
    main()
