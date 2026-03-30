#!/usr/bin/python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def as_xy(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"expected at least 2 values, got {value}")
    return arr[:2]


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) == 0:
        return vectors.reshape(0, 2)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    out = np.zeros_like(vectors, dtype=np.float32)
    out[valid] = vectors[valid] / norms[valid]
    if np.any(~valid):
        fill = np.array([1.0, 0.0], dtype=np.float32)
        for idx in np.where(~valid)[0]:
            out[idx] = out[idx - 1] if idx > 0 else fill
    return out


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values.copy()
    window = max(int(window), 1)
    if window == 1 or len(values) == 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    pad_shape = [(radius, radius)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad_shape, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if values.ndim == 1:
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)
    out = np.empty_like(values, dtype=np.float32)
    for dim in range(values.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def median_filter_1d(values, window):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return values.copy()
    window = max(int(window), 1)
    if window == 1 or len(values) == 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        out[idx] = np.median(padded[idx:idx + window])
    return out


def extract_midpoint(edge):
    if edge is None:
        return None
    p1 = as_xy(edge["p1"])
    p2 = as_xy(edge["p2"])
    return 0.5 * (p1 + p2)


def parse_record(record):
    left_edge = record.get("left_edge")
    right_edge = record.get("right_edge")
    m_left = extract_midpoint(left_edge)
    m_right = extract_midpoint(right_edge)
    if m_left is None and m_right is None:
        return None

    dirc = as_xy(record["dirc"])
    t_hint = normalize_rows(dirc[None, :])[0]
    n_hint = np.array([-t_hint[1], t_hint[0]], dtype=np.float32)
    centroid = as_xy(record.get("centroid", [0.0, 0.0]))
    road_z = float(record.get("road_z", 0.0))

    if m_left is not None and m_right is not None:
        c = 0.5 * (m_left + m_right)
        w = float(np.linalg.norm(m_left - m_right))
        # Prefer the left-right span to estimate the local road normal.
        # This is more stable than using motion direction alone on sharp turns.
        n = normalize_rows((m_left - m_right)[None, :])[0]
        if float(np.dot(n, n_hint)) < 0.0:
            n = -n
        t = np.array([-n[1], n[0]], dtype=np.float32)
        if float(np.dot(t, t_hint)) < 0.0:
            t = -t
    elif m_left is not None:
        c = m_left.copy()
        w = np.nan
        t = t_hint
        n = n_hint
    else:
        c = m_right.copy()
        w = np.nan
        t = t_hint
        n = n_hint

    return {
        "index": int(record.get("index", -1)),
        "centroid": centroid.astype(np.float32),
        "road_z": road_z,
        "m_left": None if m_left is None else m_left.astype(np.float32),
        "m_right": None if m_right is None else m_right.astype(np.float32),
        "c": c.astype(np.float32),
        "t": t.astype(np.float32),
        "n": n,
        "w": w,
    }


def fill_missing_edges(samples, fallback_width):
    filled = []
    for sample in samples:
        current = dict(sample)
        width = current["w"]
        if not np.isfinite(width):
            width = fallback_width
        if current["m_left"] is None and current["m_right"] is not None:
            current["m_left"] = current["m_right"] + width * current["n"]
        if current["m_right"] is None and current["m_left"] is not None:
            current["m_right"] = current["m_left"] - width * current["n"]
        if current["m_left"] is None or current["m_right"] is None:
            continue
        current["c"] = 0.5 * (current["m_left"] + current["m_right"])
        current["w"] = float(np.linalg.norm(current["m_left"] - current["m_right"]))
        filled.append(current)
    return filled


def filter_samples(records, args):
    parsed = []
    stats = {
        "total_records": len(records),
        "parse_rejected": 0,
        "side_rejected": 0,
        "width_rejected": 0,
        "centroid_rejected": 0,
        "backtrack_rejected": 0,
        "side_flipped": False,
    }

    for record in records:
        try:
            sample = parse_record(record)
        except Exception:
            stats["parse_rejected"] += 1
            continue
        if sample is None:
            stats["parse_rejected"] += 1
            continue
        parsed.append(sample)

    width_values = np.asarray([sample["w"] for sample in parsed if np.isfinite(sample["w"])], dtype=np.float32)
    fallback_width = float(np.median(width_values)) if len(width_values) != 0 else float(args.default_width)
    parsed = fill_missing_edges(parsed, fallback_width)
    if len(parsed) == 0:
        stats["valid_records"] = 0
        stats["width_median"] = fallback_width
        return [], stats

    normal_side_ok = 0
    flipped_side_ok = 0
    for sample in parsed:
        left_signed = float(np.dot(sample["m_left"] - sample["c"], sample["n"]))
        right_signed = float(np.dot(sample["m_right"] - sample["c"], sample["n"]))
        if left_signed > 0.0 and right_signed < 0.0:
            normal_side_ok += 1
        if left_signed < 0.0 and right_signed > 0.0:
            flipped_side_ok += 1

    stats["side_flipped"] = flipped_side_ok > normal_side_ok

    legal = []
    for sample in parsed:
        left_signed = float(np.dot(sample["m_left"] - sample["c"], sample["n"]))
        right_signed = float(np.dot(sample["m_right"] - sample["c"], sample["n"]))
        if stats["side_flipped"]:
            side_ok = left_signed < 0.0 and right_signed > 0.0
        else:
            side_ok = left_signed > 0.0 and right_signed < 0.0
        if not side_ok:
            stats["side_rejected"] += 1
            continue
        legal.append(sample)

    if len(legal) == 0:
        stats["valid_records"] = 0
        stats["width_median"] = fallback_width
        return [], stats

    legal = sorted(legal, key=lambda item: item["index"])
    widths = np.asarray([sample["w"] for sample in legal], dtype=np.float32)
    width_median = float(np.median(widths))
    width_ref = median_filter_1d(widths, args.width_window)
    width_delta = np.abs(widths - width_ref)
    width_dev_limit = max(float(args.width_dev), float(np.median(width_delta) * 3.0))

    filtered = []
    for sample, width_ref_i in zip(legal, width_ref):
        width = float(sample["w"])
        if width < args.width_min or width > args.width_max:
            stats["width_rejected"] += 1
            continue
        if args.width_dev > 0.0 and abs(width - float(width_ref_i)) > width_dev_limit:
            stats["width_rejected"] += 1
            continue
        if args.centroid_thresh > 0.0:
            centroid_offset = float(np.linalg.norm(sample["centroid"] - sample["c"]))
            if centroid_offset > args.centroid_thresh:
                stats["centroid_rejected"] += 1
                continue
        filtered.append(sample)

    filtered = filter_backtracking_samples(filtered, args, stats)

    stats["valid_records"] = len(filtered)
    stats["width_median"] = width_median
    return filtered, stats


def filter_backtracking_samples(samples, args, stats):
    if len(samples) <= 1 or args.max_backtrack <= 0.0:
        return samples

    kept = [samples[0]]
    for sample in samples[1:]:
        prev = kept[-1]
        delta = sample["c"] - prev["c"]
        ref_t = normalize_rows((prev["t"] + sample["t"])[None, :])[0]
        forward = float(np.dot(delta, ref_t))
        if forward < -float(args.max_backtrack):
            stats["backtrack_rejected"] += 1
            continue
        kept.append(sample)
    return kept


def compute_station(samples):
    ordered = sorted(samples, key=lambda item: item["index"])
    if len(ordered) == 0:
        return ordered, np.zeros((0,), dtype=np.float32)
    s = np.zeros((len(ordered),), dtype=np.float32)
    for idx in range(1, len(ordered)):
        delta_s = float(np.linalg.norm(ordered[idx]["c"] - ordered[idx - 1]["c"]))
        s[idx] = s[idx - 1] + delta_s
    return ordered, s


def estimate_center_tangents(center, fallback_tangent):
    center = np.asarray(center, dtype=np.float32)
    fallback_tangent = normalize_rows(fallback_tangent)
    if len(center) == 0:
        return center.reshape(0, 2)
    if len(center) == 1:
        return fallback_tangent.copy()

    tangent = np.zeros_like(center, dtype=np.float32)
    tangent[0] = center[1] - center[0]
    tangent[-1] = center[-1] - center[-2]
    if len(center) > 2:
        tangent[1:-1] = center[2:] - center[:-2]

    tangent = normalize_rows(tangent)
    for idx in range(len(tangent)):
        if float(np.dot(tangent[idx], fallback_tangent[idx])) < 0.0:
            tangent[idx] = -tangent[idx]
    return tangent


def smooth_samples(samples, s_values, args):
    center = np.asarray([sample["c"] for sample in samples], dtype=np.float32)
    tangent = np.asarray([sample["t"] for sample in samples], dtype=np.float32)
    widths = np.asarray([sample["w"] for sample in samples], dtype=np.float32)
    road_z = np.asarray([sample["road_z"] for sample in samples], dtype=np.float32)

    center_smooth = moving_average(center, args.center_window)
    tangent_geom = estimate_center_tangents(center_smooth, tangent)
    tangent_smooth = normalize_rows(moving_average(tangent_geom, args.dir_window))
    width_smooth = moving_average(median_filter_1d(widths, args.width_window), args.width_window)
    road_z_smooth = moving_average(road_z, args.center_window)

    fused = []
    for idx, sample in enumerate(samples):
        center_i = center_smooth[idx]
        tangent_i = tangent_smooth[idx]
        normal_i = np.array([-tangent_i[1], tangent_i[0]], dtype=np.float32)
        width_i = float(width_smooth[idx])
        if width_i < 1e-6:
            width_i = float(sample["w"])
        left_i = center_i + 0.5 * width_i * normal_i
        right_i = center_i - 0.5 * width_i * normal_i
        fused.append({
            "index": sample["index"],
            "s": float(s_values[idx]),
            "center": [float(center_i[0]), float(center_i[1]), float(road_z_smooth[idx])],
            "left_edge": [float(left_i[0]), float(left_i[1]), float(road_z_smooth[idx])],
            "right_edge": [float(right_i[0]), float(right_i[1]), float(road_z_smooth[idx])],
            "t": [float(tangent_i[0]), float(tangent_i[1])],
            "w": width_i,
        })
    return fused


def save_preview(output_path, records, fused):
    if plt is None:
        print("skip preview: matplotlib is not available")
        return

    preview_path = output_path.with_suffix('.png')
    fig, ax = plt.subplots(figsize=(10, 10))

    raw_left = []
    raw_right = []
    raw_centers = []
    for record in records:
        try:
            parsed = parse_record(record)
        except Exception:
            continue
        if parsed is None:
            continue
        if parsed["m_left"] is not None:
            raw_left.append(parsed["m_left"])
        if parsed["m_right"] is not None:
            raw_right.append(parsed["m_right"])
        raw_centers.append(parsed["c"])

    if raw_left:
        raw_left = np.asarray(raw_left, dtype=np.float32)
        ax.scatter(raw_left[:, 0], raw_left[:, 1], s=16, c="tab:blue", alpha=0.35, label="raw left midpoints")
    if raw_right:
        raw_right = np.asarray(raw_right, dtype=np.float32)
        ax.scatter(raw_right[:, 0], raw_right[:, 1], s=16, c="tab:red", alpha=0.35, label="raw right midpoints")
    if raw_centers:
        raw_centers = np.asarray(raw_centers, dtype=np.float32)
        ax.scatter(raw_centers[:, 0], raw_centers[:, 1], s=12, c="0.55", alpha=0.3, label="raw centers")

    left_edge = fused.get("left_edge") or []
    right_edge = fused.get("right_edge") or []
    center_line = fused.get("center_line") or []

    if left_edge:
        left_edge = np.asarray(left_edge, dtype=np.float32)
        ax.plot(left_edge[:, 0], left_edge[:, 1], color="tab:blue", linewidth=2.5, label="fused left edge")
    if right_edge:
        right_edge = np.asarray(right_edge, dtype=np.float32)
        ax.plot(right_edge[:, 0], right_edge[:, 1], color="tab:red", linewidth=2.5, label="fused right edge")
    if center_line:
        center_line = np.asarray(center_line, dtype=np.float32)
        ax.plot(center_line[:, 0], center_line[:, 1], color="tab:green", linewidth=2.0, linestyle="--", label="fused center line")

    ax.set_title(
        f"Road Edge Fusion Preview\nvalid={fused['meta']['valid_records']} / {fused['meta']['total_records']}, "
        f"side_flipped={fused['meta'].get('side_flipped', False)}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)
    print(f"saved preview image to {preview_path}")


def build_output(records, args):
    filtered, stats = filter_samples(records, args)
    ordered, s_values = compute_station(filtered)
    fused_samples = smooth_samples(ordered, s_values, args) if len(ordered) != 0 else []

    left_edge = [sample["left_edge"] for sample in fused_samples]
    right_edge = [sample["right_edge"] for sample in fused_samples]
    center_line = [sample["center"] for sample in fused_samples]

    return {
        "meta": {
            "input": str(args.input),
            "total_records": stats["total_records"],
            "valid_records": stats["valid_records"],
            "parse_rejected": stats["parse_rejected"],
            "side_rejected": stats["side_rejected"],
            "width_rejected": stats["width_rejected"],
            "centroid_rejected": stats["centroid_rejected"],
            "backtrack_rejected": stats["backtrack_rejected"],
            "side_flipped": stats["side_flipped"],
            "width_median": stats["width_median"],
            "params": {
                "width_min": args.width_min,
                "width_max": args.width_max,
                "width_dev": args.width_dev,
                "max_backtrack": args.max_backtrack,
                "default_width": args.default_width,
                "centroid_thresh": args.centroid_thresh,
                "dir_window": args.dir_window,
                "center_window": args.center_window,
                "edge_window": args.edge_window,
                "width_window": args.width_window,
            },
        },
        "left_edge": left_edge,
        "right_edge": right_edge,
        "center_line": center_line,
        "samples": fused_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse per-frame road edge records into a smooth road-edge track.")
    parser.add_argument("-i", "--input", required=True, help="Path to road_edge_records.json")
    parser.add_argument("-o", "--output", default=None, help="Path to fused output json")
    parser.add_argument("--width-min", type=float, default=1.5, help="Minimum legal road width in meters")
    parser.add_argument("--width-max", type=float, default=20.0, help="Maximum legal road width in meters")
    parser.add_argument("--width-dev", type=float, default=0.0, help="Allowed deviation from local median width in meters, <=0 disables")
    parser.add_argument("--max-backtrack", type=float, default=0.3, help="Reject samples that move backward along local tangent by more than this many meters")
    parser.add_argument("--default-width", type=float, default=4.0, help="Fallback width used to compensate missing sides")
    parser.add_argument("--centroid-thresh", type=float, default=0.0, help="Optional centroid-vs-pseudo-center threshold, 0 disables")
    parser.add_argument("--dir-window", type=int, default=9, help="Moving-average window for direction smoothing")
    parser.add_argument("--center-window", type=int, default=5, help="Moving-average window for center smoothing")
    parser.add_argument("--edge-window", type=int, default=5, help="Moving-average window for edge smoothing")
    parser.add_argument("--width-window", type=int, default=8, help="Median-plus-average window for width smoothing")
    parser.add_argument("--preview", dest="preview", action="store_true", help="Save a PNG preview next to the fused output json")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Disable PNG preview generation")
    parser.set_defaults(preview=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_fused.json")

    with input_path.open("r") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"expected a list of records in {input_path}, got {type(records).__name__}")

    fused = build_output(records, args)
    with output_path.open("w") as f:
        json.dump(fused, f, indent=2)

    if args.preview:
        save_preview(output_path, records, fused)

    print(f"saved fused road edges to {output_path}")
    print(
        "valid / total = "
        f"{fused['meta']['valid_records']} / {fused['meta']['total_records']}, "
        f"width_median={fused['meta']['width_median']:.3f}"
    )


if __name__ == "__main__":
    main()
