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
        description="Read images from rosbag, detect sidewalk frames, and record matched frame indices."
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
        help="Output JSON path. Defaults to <save_folder>/sidewalk_frames.json",
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
        "--sidewalk-class",
        default=None,
        type=int,
        help="Segmentation class id for sidewalk. Defaults to config value or 15.",
    )
    parser.add_argument(
        "--min-pixels",
        default=1000,
        type=int,
        help="Minimum sidewalk pixels to mark a frame as containing sidewalk.",
    )
    parser.add_argument(
        "--min-ratio",
        default=0.001,
        type=float,
        help="Minimum sidewalk pixel ratio to mark a frame as containing sidewalk.",
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


def main():
    args = parse_args()
    config = load_config(args.config)

    bag_path = args.bag or config["bag_file"]
    fastforward = args.fastforward if args.fastforward is not None else config["start_time"]
    duration = args.duration if args.duration is not None else config["play_time"]
    sidewalk_class = (
        args.sidewalk_class
        if args.sidewalk_class is not None
        else config.get("sidewalk_class", 15)
    )
    output_path = args.output or os.path.join(config["save_folder"], "sidewalk_frames.json")
    image_topic = config["camera_topic"]
    decode_image = build_image_decoder(config.get("image_compressed", False))

    intrinsic = np.asarray(config.get("intrinsic"), dtype=np.float32)
    distortion = np.asarray(config.get("distortion_matrix"), dtype=np.float32)
    use_undistort = intrinsic.size == 9 and distortion.size >= 4

    predictor = getattr(predict, config["predict_func"])(config["model_config"], config["model_file"])

    matches = []
    total_frames = 0

    with Bag(bag_path) as bag:
        start_time, end_time = get_time_window(bag, fastforward, duration)
        for _, msg, stamp in bag.read_messages(
            topics=[image_topic], start_time=start_time, end_time=end_time
        ):
            img = decode_image(msg)
            if use_undistort:
                img = cv2.undistort(img, intrinsic, distortion)

            seg = predictor(img)
            sidewalk_pixels = int(np.count_nonzero(seg == sidewalk_class))
            total_pixels = int(seg.size)
            sidewalk_ratio = float(sidewalk_pixels / max(total_pixels, 1))

            frame_index = total_frames + 1
            if sidewalk_pixels >= args.min_pixels and sidewalk_ratio >= args.min_ratio:
                matches.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": stamp.to_sec(),
                        "sidewalk_pixels": sidewalk_pixels,
                        "sidewalk_ratio": sidewalk_ratio,
                    }
                )

            total_frames += 1
            print(
                f"frame={frame_index} sidewalk_pixels={sidewalk_pixels} "
                f"sidewalk_ratio={sidewalk_ratio:.6f}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "bag_file": bag_path,
        "camera_topic": image_topic,
        "sidewalk_class": int(sidewalk_class),
        "min_pixels": int(args.min_pixels),
        "min_ratio": float(args.min_ratio),
        "total_frames": int(total_frames),
        "matched_frames": matches,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"matched {len(matches)} / {total_frames} frames")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
