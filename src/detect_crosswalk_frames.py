#!/usr/bin/python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import genpy
import numpy as np
from cv_bridge import CvBridge
from rosbag import Bag

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predict


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read images from rosbag, detect crosswalk frames, and record matched frame indices."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/outdoor_config.json",
        help="Config file path.",
    )
    parser.add_argument("-b", "--bag", default=None, help="ROS bag path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Defaults to <save_folder>/crosswalk_frames.json",
    )
    parser.add_argument(
        "--fastforward",
        default=None,
        type=float,
        help="Start reading from this many seconds after bag start.",
    )
    parser.add_argument(
        "--duration",
        default=None,
        type=float,
        help="How many seconds to read. Use -1 for all remaining messages.",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Run detection on every N-th frame. Defaults to 1.",
    )
    parser.add_argument(
        "--crosswalk-classes",
        default=None,
        nargs="+",
        type=int,
        help="Segmentation class ids for crosswalk. Defaults to config value or 23.",
    )
    parser.add_argument(
        "--min-pixels",
        default=500,
        type=int,
        help="Minimum crosswalk pixels to mark a frame as containing crosswalk.",
    )
    parser.add_argument(
        "--min-ratio",
        default=0.0005,
        type=float,
        help="Minimum crosswalk pixel ratio to mark a frame as containing crosswalk.",
    )
    return parser.parse_args()


def get_time_window(bag, fastforward, duration):
    start_sec = bag.get_start_time() + fastforward
    start_time = genpy.Time.from_sec(start_sec)
    if duration == -1:
        return start_time, None
    end_time = genpy.Time.from_sec(start_sec + duration)
    return start_time, end_time


def build_image_decoder(is_compressed):
    bridge = CvBridge()
    if is_compressed:
        return bridge.compressed_imgmsg_to_cv2
    return bridge.imgmsg_to_cv2


def resolve_crosswalk_classes(args, config):
    if args.crosswalk_classes is not None:
        return list(dict.fromkeys(int(v) for v in args.crosswalk_classes))

    classes = []
    if config.get("crosswalk_class") is not None:
        classes.append(int(config["crosswalk_class"]))
    if config.get("crosswalk_plain_class") is not None:
        classes.append(int(config["crosswalk_plain_class"]))
    if not classes:
        classes.append(23)
    return list(dict.fromkeys(classes))


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.step <= 0:
        raise ValueError("--step must be a positive integer")

    bag_path = args.bag or config["bag_file"]
    fastforward = args.fastforward if args.fastforward is not None else config["start_time"]
    duration = args.duration if args.duration is not None else config["play_time"]
    crosswalk_classes = resolve_crosswalk_classes(args, config)
    output_path = args.output or os.path.join(config["save_folder"], "crosswalk_frames.json")
    image_topic = config["camera_topic"]
    decode_image = build_image_decoder(config.get("image_compressed", False))

    intrinsic = np.asarray(config.get("intrinsic"), dtype=np.float32)
    distortion = np.asarray(config.get("distortion_matrix"), dtype=np.float32)
    use_undistort = intrinsic.size == 9 and distortion.size >= 4

    predictor = getattr(predict, config["predict_func"])(config["model_config"], config["model_file"])

    matches = []
    total_frames = 0
    target_classes = np.asarray(crosswalk_classes, dtype=np.int32)

    with Bag(bag_path) as bag:
        start_time, end_time = get_time_window(bag, fastforward, duration)
        for _, msg, stamp in bag.read_messages(
            topics=[image_topic], start_time=start_time, end_time=end_time
        ):
            frame_index = total_frames + 1
            if (frame_index - 1) % args.step != 0:
                total_frames += 1
                continue

            img = decode_image(msg)
            if use_undistort:
                img = cv2.undistort(img, intrinsic, distortion)

            seg = predictor(img)
            crosswalk_mask = np.isin(seg, target_classes)
            crosswalk_pixels = int(np.count_nonzero(crosswalk_mask))
            total_pixels = int(seg.size)
            crosswalk_ratio = float(crosswalk_pixels / max(total_pixels, 1))
            class_pixel_counts = {
                str(cls): int(np.count_nonzero(seg == cls))
                for cls in crosswalk_classes
            }

            if crosswalk_pixels >= args.min_pixels and crosswalk_ratio >= args.min_ratio:
                matches.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": stamp.to_sec(),
                        "crosswalk_pixels": crosswalk_pixels,
                        "crosswalk_ratio": crosswalk_ratio,
                        "class_pixel_counts": class_pixel_counts,
                    }
                )

            total_frames += 1
            print(
                f"frame={frame_index} crosswalk_pixels={crosswalk_pixels} "
                f"crosswalk_ratio={crosswalk_ratio:.6f} classes={class_pixel_counts}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "bag_file": bag_path,
        "camera_topic": image_topic,
        "crosswalk_classes": [int(v) for v in crosswalk_classes],
        "min_pixels": int(args.min_pixels),
        "min_ratio": float(args.min_ratio),
        "step": int(args.step),
        "total_frames": int(total_frames),
        "evaluated_frames": int((total_frames + args.step - 1) // args.step),
        "matched_frames": matches,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"matched {len(matches)} / {total_frames} frames")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
