#!/usr/bin/python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
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


def make_debug_entry(sample, reason=None, forward=None):
    entry = {
        "index": int(sample["index"]),
        "center": np.asarray(sample["c"], dtype=np.float32).copy(),
        "left": None if sample["m_left"] is None else np.asarray(sample["m_left"], dtype=np.float32).copy(),
        "right": None if sample["m_right"] is None else np.asarray(sample["m_right"], dtype=np.float32).copy(),
    }
    if reason is not None:
        entry["reason"] = str(reason)
    if forward is not None:
        entry["forward"] = float(forward)
    return entry


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
    debug = {
        "single_side_missing": [],
        "left_right_swap": [],
        "local_backtracking_outlier": [],
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
        if (sample["m_left"] is None) != (sample["m_right"] is None):
            debug["single_side_missing"].append(make_debug_entry(sample))

    width_values = np.asarray([sample["w"] for sample in parsed if np.isfinite(sample["w"])], dtype=np.float32)
    fallback_width = float(np.median(width_values)) if len(width_values) != 0 else float(args.default_width)
    parsed = fill_missing_edges(parsed, fallback_width)
    if len(parsed) == 0:
        stats["valid_records"] = 0
        stats["width_median"] = fallback_width
        return [], stats, debug

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
            debug["left_right_swap"].append(make_debug_entry(sample))
            continue
        legal.append(sample)

    if len(legal) == 0:
        stats["valid_records"] = 0
        stats["width_median"] = fallback_width
        return [], stats, debug

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

    filtered = filter_backtracking_samples(filtered, args, stats, debug)

    stats["valid_records"] = len(filtered)
    stats["width_median"] = width_median
    return filtered, stats, debug


def filter_backtracking_samples(samples, args, stats, debug):
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
            debug["local_backtracking_outlier"].append(
                make_debug_entry(sample, reason=f"forward={forward:.3f}", forward=forward)
            )
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


def fit_polynomial_curve(s_values, points_xy, degree):
    s_values = np.asarray(s_values, dtype=np.float32).reshape(-1)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(s_values) != len(points_xy):
        raise ValueError(f"s_values and points_xy must have the same length, got {len(s_values)} and {len(points_xy)}")
    if len(points_xy) == 0:
        return points_xy.reshape(0, 2)
    if len(points_xy) == 1:
        return points_xy.copy()

    degree = max(1, min(int(degree), len(points_xy) - 1))
    s_min = float(np.min(s_values))
    s_max = float(np.max(s_values))
    s_span = max(s_max - s_min, 1e-6)
    s_norm = ((s_values - s_min) / s_span) * 2.0 - 1.0

    fitted = np.empty_like(points_xy, dtype=np.float32)
    for dim in range(points_xy.shape[1]):
        coeffs = np.polyfit(s_norm, points_xy[:, dim], deg=degree)
        fitted[:, dim] = np.polyval(coeffs, s_norm).astype(np.float32)
    return fitted


def fit_edges_least_squares(samples, s_values, args):
    if len(samples) == 0:
        return []

    left_raw = np.asarray([sample["m_left"] for sample in samples], dtype=np.float32)
    right_raw = np.asarray([sample["m_right"] for sample in samples], dtype=np.float32)
    road_z = np.asarray([sample["road_z"] for sample in samples], dtype=np.float32)

    left_fit = fit_polynomial_curve(s_values, left_raw, args.ls_degree)
    right_fit = fit_polynomial_curve(s_values, right_raw, args.ls_degree)
    center_fit = 0.5 * (left_fit + right_fit)
    tangent_fit = estimate_center_tangents(center_fit, np.asarray([sample["t"] for sample in samples], dtype=np.float32))
    road_z_fit = moving_average(road_z, args.center_window)

    fused = []
    for idx, sample in enumerate(samples):
        left_i = left_fit[idx]
        right_i = right_fit[idx]
        center_i = center_fit[idx]
        tangent_i = tangent_fit[idx]
        width_i = float(np.linalg.norm(left_i - right_i))
        fused.append({
            "index": sample["index"],
            "s": float(s_values[idx]),
            "center": [float(center_i[0]), float(center_i[1]), float(road_z_fit[idx])],
            "left_edge": [float(left_i[0]), float(left_i[1]), float(road_z_fit[idx])],
            "right_edge": [float(right_i[0]), float(right_i[1]), float(road_z_fit[idx])],
            "t": [float(tangent_i[0]), float(tangent_i[1])],
            "w": width_i,
        })
    return fused


def plot_raw_records(ax, records):
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


def plot_fused(ax, fused):
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


def subset_records_by_index(records, focus_indices, radius=8):
    if not focus_indices:
        return []
    focus_indices = np.asarray(sorted(set(int(idx) for idx in focus_indices)), dtype=np.int32)
    subset = []
    for record in records:
        index = int(record.get("index", -10**9))
        if np.any(np.abs(focus_indices - index) <= radius):
            subset.append(record)
    return subset


def subset_fused_by_index(fused, focus_indices, radius=8):
    if not focus_indices:
        return {
            "left_edge": [],
            "right_edge": [],
            "center_line": [],
            "samples": [],
            "meta": fused.get("meta", {}),
        }

    focus_indices = np.asarray(sorted(set(int(idx) for idx in focus_indices)), dtype=np.int32)
    kept_samples = []
    for sample in fused.get("samples", []):
        index = int(sample.get("index", -10**9))
        if np.any(np.abs(focus_indices - index) <= radius):
            kept_samples.append(sample)

    return {
        "left_edge": [sample["left_edge"] for sample in kept_samples],
        "right_edge": [sample["right_edge"] for sample in kept_samples],
        "center_line": [sample["center"] for sample in kept_samples],
        "samples": kept_samples,
        "meta": fused.get("meta", {}),
    }


def select_focus_entries(entries, group_gap=5):
    if not entries:
        return []

    ordered = sorted(entries, key=lambda item: int(item["index"]))
    groups = [[ordered[0]]]
    for entry in ordered[1:]:
        if int(entry["index"]) - int(groups[-1][-1]["index"]) <= group_gap:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    def group_score(group):
        severity = 0.0
        for entry in group:
            if "forward" in entry:
                severity = max(severity, abs(float(entry["forward"])))
        return (len(group), severity, -abs(float(group[len(group) // 2]["index"])))

    return max(groups, key=group_score)


def set_focus_limits(ax, entries, raw_records, fused):
    points = []
    for entry in entries:
        points.append(np.asarray(entry["center"], dtype=np.float32))
        if entry.get("left") is not None:
            points.append(np.asarray(entry["left"], dtype=np.float32))
        if entry.get("right") is not None:
            points.append(np.asarray(entry["right"], dtype=np.float32))

    for record in raw_records:
        try:
            parsed = parse_record(record)
        except Exception:
            continue
        if parsed is None:
            continue
        points.append(np.asarray(parsed["c"], dtype=np.float32))
        if parsed["m_left"] is not None:
            points.append(np.asarray(parsed["m_left"], dtype=np.float32))
        if parsed["m_right"] is not None:
            points.append(np.asarray(parsed["m_right"], dtype=np.float32))

    for sample in fused.get("samples", []):
        points.append(np.asarray(sample["center"][:2], dtype=np.float32))
        points.append(np.asarray(sample["left_edge"][:2], dtype=np.float32))
        points.append(np.asarray(sample["right_edge"][:2], dtype=np.float32))

    if not points:
        return

    pts = np.asarray(points, dtype=np.float32)
    xy_min = np.min(pts, axis=0)
    xy_max = np.max(pts, axis=0)
    span = np.maximum(xy_max - xy_min, 1.0)
    pad = np.maximum(span * 0.25, 1.5)
    ax.set_xlim(float(xy_min[0] - pad[0]), float(xy_max[0] + pad[0]))
    ax.set_ylim(float(xy_min[1] - pad[1]), float(xy_max[1] + pad[1]))


def annotate_debug_entries(ax, entries, color, label):
    if not entries:
        return
    centers = np.asarray([entry["center"] for entry in entries], dtype=np.float32)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=72,
        facecolors="none",
        edgecolors=color,
        linewidths=1.8,
        label=label,
        zorder=5,
    )
    for entry in entries:
        center = entry["center"]
        ax.text(center[0], center[1], str(entry["index"]), fontsize=7, color=color)
        left = entry.get("left")
        right = entry.get("right")
        if left is not None:
            ax.plot([center[0], left[0]], [center[1], left[1]], color=color, linewidth=1.0, alpha=0.9)
            ax.scatter(left[0], left[1], s=24, c=color, alpha=0.9)
        if right is not None:
            ax.plot([center[0], right[0]], [center[1], right[1]], color=color, linewidth=1.0, alpha=0.9)
            ax.scatter(right[0], right[1], s=24, c=color, alpha=0.9)


def save_debug_preview(debug_dir, stem, title, suffix, records, fused, entries, color):
    focus_entries = select_focus_entries(entries)
    focus_indices = [entry["index"] for entry in focus_entries]
    records_focus = subset_records_by_index(records, focus_indices, radius=8)
    fused_focus = subset_fused_by_index(fused, focus_indices, radius=8)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_raw_records(ax, records_focus)
    plot_fused(ax, fused_focus)
    annotate_debug_entries(ax, focus_entries, color=color, label=title)
    set_focus_limits(ax, focus_entries, records_focus, fused_focus)
    ax.set_title(f"{title}\ncount={len(focus_entries)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = debug_dir / f"{stem}_{suffix}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"saved debug image to {output_path}")


def resolve_debug_dir(output_path):
    candidates = [
        Path("demo_output"),
        output_path.parent / f"{output_path.stem}_demo_output",
        Path("/tmp/demo_output"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            with probe.open("w") as f:
                f.write("ok")
            probe.unlink()
            if candidate != Path("demo_output"):
                print(f"demo_output is not writable, fallback to {candidate}")
            return candidate
        except OSError:
            continue
    raise OSError("no writable debug output directory available")


def save_preview(output_path, records, fused, debug):
    if plt is None:
        print("skip preview: matplotlib is not available")
        return

    preview_path = output_path.with_suffix('.png')
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_raw_records(ax, records)
    plot_fused(ax, fused)

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

    debug_dir = resolve_debug_dir(output_path)
    stem = output_path.stem
    save_debug_preview(
        debug_dir,
        stem,
        "single-side missing",
        "single_side_missing",
        records,
        fused,
        debug["single_side_missing"],
        color="goldenrod",
    )
    save_debug_preview(
        debug_dir,
        stem,
        "left-right swap",
        "left_right_swap",
        records,
        fused,
        debug["left_right_swap"],
        color="magenta",
    )
    save_debug_preview(
        debug_dir,
        stem,
        "local backtracking outlier",
        "local_backtracking_outlier",
        records,
        fused,
        debug["local_backtracking_outlier"],
        color="black",
    )


def build_output(records, args):
    filtered, stats, debug = filter_samples(records, args)
    ordered, s_values = compute_station(filtered)
    if len(ordered) == 0:
        fused_samples = []
    elif args.method == "least_squares_edges":
        fused_samples = fit_edges_least_squares(ordered, s_values, args)
    else:
        fused_samples = smooth_samples(ordered, s_values, args)

    left_edge = [sample["left_edge"] for sample in fused_samples]
    right_edge = [sample["right_edge"] for sample in fused_samples]
    center_line = [sample["center"] for sample in fused_samples]

    output = {
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
            "method": args.method,
            "debug_counts": {
                "single_side_missing": len(debug["single_side_missing"]),
                "left_right_swap": len(debug["left_right_swap"]),
                "local_backtracking_outlier": len(debug["local_backtracking_outlier"]),
            },
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
                "ls_degree": args.ls_degree,
            },
        },
        "left_edge": left_edge,
        "right_edge": right_edge,
        "center_line": center_line,
        "samples": fused_samples,
    }
    return output, debug


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse per-frame road edge records into a smooth road-edge track.")
    parser.add_argument("-i", "--input", required=True, help="Path to road_edge_records.json")
    parser.add_argument("-o", "--output", default=None, help="Path to fused output json")
    parser.add_argument(
        "--method",
        choices=("moving_average", "least_squares_edges"),
        default="moving_average",
        help="Fusion method. least_squares_edges fits left/right edges independently with polynomial least squares.",
    )
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
    parser.add_argument("--ls-degree", type=int, default=3, help="Polynomial degree for least_squares_edges.")
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

    fused, debug = build_output(records, args)
    with output_path.open("w") as f:
        json.dump(fused, f, indent=2)

    if args.preview:
        save_preview(output_path, records, fused, debug)

    print(f"saved fused road edges to {output_path}")
    print(
        "valid / total = "
        f"{fused['meta']['valid_records']} / {fused['meta']['total_records']}, "
        f"width_median={fused['meta']['width_median']:.3f}"
    )


if __name__ == "__main__":
    main()
