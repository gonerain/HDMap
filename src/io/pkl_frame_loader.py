import pickle
from pathlib import Path

import numpy as np


def _load_indoor_payload(input_path):
    with Path(input_path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"indoor pkl should contain a list, got {type(payload).__name__}")
    return payload


def infer_pkl_mode(input_path):
    with Path(input_path).open("rb") as f:
        first_obj = pickle.load(f)
    return "indoor" if isinstance(first_obj, list) else "outdoor"


def iter_frames(input_path, mode="auto"):
    input_path = Path(input_path)
    if mode == "auto":
        mode = infer_pkl_mode(input_path)

    if mode == "indoor":
        for frame in _load_indoor_payload(input_path):
            yield np.asarray(frame)
        return

    if mode != "outdoor":
        raise ValueError(f"unsupported mode: {mode}")

    with input_path.open("rb") as f:
        while True:
            try:
                yield np.asarray(pickle.load(f))
            except EOFError:
                break


def load_frame(input_path, frame_index, mode="auto"):
    if frame_index < 0:
        raise IndexError(f"frame_index must be >= 0, got {frame_index}")

    input_path = Path(input_path)
    if mode == "auto":
        with input_path.open("rb") as f:
            first_obj = pickle.load(f)
            if isinstance(first_obj, list):
                if frame_index >= len(first_obj):
                    raise IndexError(f"frame {frame_index} exceeds indoor pkl length {len(first_obj)}")
                return np.asarray(first_obj[frame_index]), "indoor"

            if frame_index == 0:
                return np.asarray(first_obj), "outdoor"

            current_index = 0
            while True:
                try:
                    frame = pickle.load(f)
                except EOFError as exc:
                    raise IndexError(f"frame {frame_index} exceeds outdoor pkl length") from exc
                current_index += 1
                if current_index == frame_index:
                    return np.asarray(frame), "outdoor"

    if mode == "indoor":
        frames = _load_indoor_payload(input_path)
        if frame_index >= len(frames):
            raise IndexError(f"frame {frame_index} exceeds indoor pkl length {len(frames)}")
        return np.asarray(frames[frame_index]), "indoor"

    if mode != "outdoor":
        raise ValueError(f"unsupported mode: {mode}")

    with input_path.open("rb") as f:
        current_index = 0
        while True:
            try:
                frame = pickle.load(f)
            except EOFError as exc:
                raise IndexError(f"frame {frame_index} exceeds outdoor pkl length") from exc
            if current_index == frame_index:
                return np.asarray(frame), "outdoor"
            current_index += 1


def summarize_frame(frame):
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D frame array, got shape {arr.shape}")

    summary = {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "point_count": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]),
    }

    if arr.shape[0] == 0:
        summary["xyz_min"] = None
        summary["xyz_max"] = None
    else:
        xyz = arr[:, : min(3, arr.shape[1])].astype(np.float64, copy=False)
        summary["xyz_min"] = xyz.min(axis=0).tolist()
        summary["xyz_max"] = xyz.max(axis=0).tolist()

    if arr.shape[1] >= 4 and arr.shape[0] != 0:
        class_ids, counts = np.unique(np.rint(arr[:, 3]).astype(np.int64), return_counts=True)
        summary["class_histogram"] = [
            {"class_id": int(class_id), "count": int(count)}
            for class_id, count in zip(class_ids.tolist(), counts.tolist())
        ]
    else:
        summary["class_histogram"] = []

    return summary
