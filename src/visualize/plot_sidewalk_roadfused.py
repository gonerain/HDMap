#!/usr/bin/python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for plotting") from exc


def as_xy_array(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim == 1:
        return arr.reshape(1, -1)[:, :2]
    return arr[:, :2]


def load_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}, got {type(data).__name__}")
    return data


def collect_polyline(records, key):
    rows = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        rows.append(np.asarray(value, dtype=np.float32)[:2])
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(rows)


def collect_side_field(records, side_key, field):
    items = []
    for record in records:
        side = record.get(side_key)
        if not isinstance(side, dict):
            continue
        value = side.get(field)
        if value is None:
            continue
        arr = as_xy_array(value)
        if len(arr) != 0:
            items.append((int(record["index"]), arr))
    return items


def plot_records(records, output_path, show_outline=False, show_anchors=True, show_indices=False):
    fig, ax = plt.subplots(figsize=(12, 12))

    road_center = collect_polyline(records, "road_center")
    left_road = collect_polyline(records, "left_road_edge")
    right_road = collect_polyline(records, "right_road_edge")

    if len(road_center) >= 2:
        ax.plot(road_center[:, 0], road_center[:, 1], "--", color="0.35", linewidth=2.0, label="road center")
    elif len(road_center) == 1:
        ax.scatter(road_center[:, 0], road_center[:, 1], s=52, color="0.35", label="road center")

    if len(left_road) >= 2:
        ax.plot(left_road[:, 0], left_road[:, 1], color="tab:blue", linewidth=2.2, label="road left edge")
    elif len(left_road) == 1:
        ax.scatter(left_road[:, 0], left_road[:, 1], s=52, color="tab:blue", label="road left edge")

    if len(right_road) >= 2:
        ax.plot(right_road[:, 0], right_road[:, 1], color="tab:red", linewidth=2.2, label="road right edge")
    elif len(right_road) == 1:
        ax.scatter(right_road[:, 0], right_road[:, 1], s=52, color="tab:red", label="road right edge")

    if len(records) == 1:
        record = records[0]
        center = np.asarray(record["road_center"], dtype=np.float32)[:2]
        tangent = np.asarray(record.get("road_tangent", [1.0, 0.0]), dtype=np.float32)[:2]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
            preview_len = max(float(record.get("road_width", 4.0)) * 0.7, 2.0)
            p1 = center - 0.5 * preview_len * tangent
            p2 = center + 0.5 * preview_len * tangent
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linestyle="--",
                color="0.45",
                linewidth=1.8,
                alpha=0.9,
                label="road tangent preview",
            )

    left_inner = collect_side_field(records, "left_sidewalk", "inner_polyline")
    right_inner = collect_side_field(records, "right_sidewalk", "inner_polyline")
    left_anchor = collect_side_field(records, "left_sidewalk", "anchor")
    right_anchor = collect_side_field(records, "right_sidewalk", "anchor")
    left_outline = collect_side_field(records, "left_sidewalk", "outline")
    right_outline = collect_side_field(records, "right_sidewalk", "outline")

    for _, polyline in left_inner:
        ax.plot(polyline[:, 0], polyline[:, 1], color="tab:cyan", linewidth=2.0, alpha=0.9)
    for _, polyline in right_inner:
        ax.plot(polyline[:, 0], polyline[:, 1], color="tab:orange", linewidth=2.0, alpha=0.9)

    if left_inner:
        ax.plot([], [], color="tab:cyan", linewidth=2.0, label="left sidewalk inner polyline")
    if right_inner:
        ax.plot([], [], color="tab:orange", linewidth=2.0, label="right sidewalk inner polyline")

    if show_outline:
        for _, outline in left_outline:
            ax.plot(outline[:, 0], outline[:, 1], color="tab:blue", linewidth=1.0, alpha=0.25)
        for _, outline in right_outline:
            ax.plot(outline[:, 0], outline[:, 1], color="tab:red", linewidth=1.0, alpha=0.25)
        if left_outline:
            ax.plot([], [], color="tab:blue", linewidth=1.0, alpha=0.4, label="left sidewalk outline")
        if right_outline:
            ax.plot([], [], color="tab:red", linewidth=1.0, alpha=0.4, label="right sidewalk outline")

    if show_anchors:
        if left_anchor:
            pts = np.vstack([arr for _, arr in left_anchor])
            ax.scatter(pts[:, 0], pts[:, 1], s=28, color="navy", alpha=0.9, label="left anchors")
        if right_anchor:
            pts = np.vstack([arr for _, arr in right_anchor])
            ax.scatter(pts[:, 0], pts[:, 1], s=28, color="darkorange", alpha=0.9, label="right anchors")

    if show_indices:
        for record in records:
            center = np.asarray(record["road_center"], dtype=np.float32)[:2]
            ax.text(center[0], center[1], str(record["index"]), fontsize=7, color="0.2")

    ax.set_title(f"Sidewalk Road-Fused Preview\nrecords={len(records)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sidewalk_roadfused records with road references.")
    parser.add_argument(
        "-i",
        "--input",
        default="demo_output/sidewalk_roadfused/sidewalk_roadfused_records.json",
        help="Path to sidewalk_roadfused_records.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input_stem>.png",
    )
    parser.add_argument(
        "--show-outline",
        action="store_true",
        help="Also draw sidewalk outlines with low alpha.",
    )
    parser.add_argument(
        "--hide-anchors",
        action="store_true",
        help="Do not draw anchor points.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Annotate road centers with record indices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")

    records = load_records(input_path)
    plot_records(
        records,
        output_path,
        show_outline=args.show_outline,
        show_anchors=not args.hide_anchors,
        show_indices=args.show_indices,
    )
    print(f"saved preview image to {output_path}")


if __name__ == "__main__":
    main()
