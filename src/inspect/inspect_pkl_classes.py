#!/usr/bin/python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_label_names(path):
    if path is None:
        return {}
    with Path(path).open("r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "labels" in payload and isinstance(payload["labels"], list):
            return {idx: str(name) for idx, name in enumerate(payload["labels"])}
        if "classes" in payload and isinstance(payload["classes"], list):
            return {idx: str(name) for idx, name in enumerate(payload["classes"])}
        names = {}
        for key, value in payload.items():
            try:
                names[int(key)] = str(value)
            except Exception:
                continue
        return names

    if isinstance(payload, list):
        return {idx: str(name) for idx, name in enumerate(payload)}

    return {}


def iter_outdoor_frames(input_path):
    try:
        with Path(input_path).open("rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"failed to unpickle {input_path}: missing module {exc.name!r}. "
            "This file may not be a plain semantic point-cloud pkl."
        ) from exc


def iter_indoor_frames(input_path):
    try:
        with Path(input_path).open("rb") as f:
            payload = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"failed to unpickle {input_path}: missing module {exc.name!r}. "
            "This file may not be a plain semantic point-cloud pkl."
        ) from exc
    if isinstance(payload, list):
        for frame in payload:
            yield frame
        return
    raise ValueError(f"indoor pkl should contain a list, got {type(payload).__name__}")


def summarize_frame(frame):
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D frame array, got shape {arr.shape}")
    summary = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "point_count": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]) if arr.shape[0] >= 0 else 0,
    }
    if arr.shape[0] == 0 or arr.shape[1] < 3:
        summary["xyz_min"] = None
        summary["xyz_max"] = None
    else:
        xyz = arr[:, :3].astype(np.float64, copy=False)
        summary["xyz_min"] = xyz.min(axis=0)
        summary["xyz_max"] = xyz.max(axis=0)
    if arr.shape[0] == 0 or arr.shape[1] < 4:
        summary["class_ids"] = np.zeros((0,), dtype=np.int64)
    else:
        summary["class_ids"] = np.rint(arr[:, 3]).astype(np.int64, copy=False)
    return summary


def format_vec3(vec):
    if vec is None:
        return "n/a"
    return "[" + ", ".join(f"{float(v):.3f}" for v in vec) + "]"


def main():
    parser = argparse.ArgumentParser(description="Inspect semantic pkl frames and print class statistics.")
    parser.add_argument("-i", "--input", required=True, help="Path to semantic pkl file")
    parser.add_argument("-m", "--mode", choices=["outdoor", "indoor"], required=True, help="How the pkl is organized")
    parser.add_argument("--labels", default=None, help="Optional JSON file mapping class ids to names")
    parser.add_argument("--max-frames", type=int, default=None, help="Only inspect the first N logical frames")
    parser.add_argument("--frame-samples", type=int, default=5, help="Print details for the first N inspected frames")
    args = parser.parse_args()

    label_names = load_label_names(args.labels)
    frame_iter = iter_outdoor_frames(args.input) if args.mode == "outdoor" else iter_indoor_frames(args.input)

    total_frames = 0
    total_points = 0
    feature_dims = set()
    dtypes = set()
    empty_frames = 0
    class_hist = {}
    global_xyz_min = None
    global_xyz_max = None
    sampled_frames = []

    for frame_idx, frame in enumerate(frame_iter):
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        info = summarize_frame(frame)
        total_frames += 1
        total_points += info["point_count"]
        feature_dims.add(info["feature_dim"])
        dtypes.add(info["dtype"])
        if info["point_count"] == 0:
            empty_frames += 1

        if info["xyz_min"] is not None:
            if global_xyz_min is None:
                global_xyz_min = info["xyz_min"].copy()
                global_xyz_max = info["xyz_max"].copy()
            else:
                global_xyz_min = np.minimum(global_xyz_min, info["xyz_min"])
                global_xyz_max = np.maximum(global_xyz_max, info["xyz_max"])

        unique_ids, counts = np.unique(info["class_ids"], return_counts=True)
        for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
            class_hist[class_id] = class_hist.get(class_id, 0) + int(count)

        if len(sampled_frames) < max(int(args.frame_samples), 0):
            sampled_frames.append({
                "frame_index": frame_idx,
                "shape": info["shape"],
                "dtype": info["dtype"],
                "point_count": info["point_count"],
                "xyz_min": info["xyz_min"],
                "xyz_max": info["xyz_max"],
                "class_count": len(unique_ids),
                "classes": unique_ids.tolist(),
            })

    print(f"input: {args.input}")
    print(f"mode: {args.mode}")
    print(f"frames_inspected: {total_frames}")
    print(f"points_total: {total_points}")
    print(f"points_per_frame_avg: {((total_points / total_frames) if total_frames else 0.0):.2f}")
    print(f"empty_frames: {empty_frames}")
    print(f"feature_dims: {sorted(feature_dims)}")
    print(f"dtypes: {sorted(dtypes)}")
    print(f"xyz_min_global: {format_vec3(global_xyz_min)}")
    print(f"xyz_max_global: {format_vec3(global_xyz_max)}")
    print("")

    print("class_histogram:")
    if not class_hist:
        print("  <none>")
    else:
        for class_id, count in sorted(class_hist.items(), key=lambda item: (-item[1], item[0])):
            ratio = (count / total_points) if total_points else 0.0
            name = label_names.get(class_id)
            suffix = f" ({name})" if name is not None else ""
            print(f"  class {class_id:>3}: {count:>12} pts  {ratio:>7.3%}{suffix}")
    print("")

    print("sampled_frames:")
    if not sampled_frames:
        print("  <none>")
    else:
        for item in sampled_frames:
            print(
                f"  frame {item['frame_index']:>6}: shape={item['shape']}, dtype={item['dtype']}, "
                f"points={item['point_count']}, classes={item['class_count']}"
            )
            print(f"    xyz_min={format_vec3(item['xyz_min'])}, xyz_max={format_vec3(item['xyz_max'])}")
            print(f"    class_ids={item['classes']}")


if __name__ == "__main__":
    main()
