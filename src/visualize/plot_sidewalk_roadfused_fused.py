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
    if isinstance(data, list):
        raise ValueError(
            f"{path} looks like raw sidewalk_roadfused records (list). "
            "Use src/visualize/plot_sidewalk_roadfused.py for that file, or pass *_fused.json to this script."
        )
    if not isinstance(data, dict):
        raise ValueError(f"expected a fused dict in {path}, got {type(data).__name__}")
    return data


def load_raw_records_for_fused(input_path, fused_data):
    meta = fused_data.get("meta", {})
    raw_name = meta.get("input")
    if not raw_name:
        return []
    raw_path = Path(raw_name)
    if not raw_path.is_absolute():
        raw_path = input_path.parent / raw_path
    if not raw_path.exists():
        return []
    with open(raw_path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def collect_polyline(records, key):
    rows = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        arr = as_xy_array(value)
        if len(arr) == 1:
            rows.append(arr[0])
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(rows)


def dedupe_polyline(points_xy, tol=1e-4):
    points_xy = as_xy_array(points_xy)
    if len(points_xy) == 0:
        return points_xy
    dedup = [points_xy[0]]
    for point in points_xy[1:]:
        if np.linalg.norm(point - dedup[-1]) > tol:
            dedup.append(point)
    return np.asarray(dedup, dtype=np.float32)


def collect_track_source_indices(data):
    indices = set()
    for side_key in ("left_sidewalks", "right_sidewalks"):
        for item in data.get(side_key, []):
            for value in item.get("source_indices", []):
                indices.add(int(value))
    return indices


def plot_records(
    data,
    output_path,
    raw_records=None,
    show_road=True,
    show_anchors=True,
    show_inner=True,
    show_history_inner=False,
    show_indices=False,
):
    fig, ax = plt.subplots(figsize=(12, 12))

    raw_records = raw_records or []
    if show_road and raw_records:
        source_indices = collect_track_source_indices(data)
        if source_indices:
            raw_records = [record for record in raw_records if int(record.get("index", -1)) in source_indices]

        road_center = dedupe_polyline(collect_polyline(raw_records, "road_center"))
        left_road = dedupe_polyline(collect_polyline(raw_records, "left_road_edge"))
        right_road = dedupe_polyline(collect_polyline(raw_records, "right_road_edge"))

        if len(road_center) >= 2:
            ax.plot(road_center[:, 0], road_center[:, 1], "--", color="0.4", linewidth=1.8, alpha=0.9, label="road center")
        elif len(road_center) == 1:
            ax.scatter(road_center[:, 0], road_center[:, 1], s=36, color="0.4", alpha=0.9, label="road center")
        if len(left_road) >= 2:
            ax.plot(left_road[:, 0], left_road[:, 1], color="tab:blue", linewidth=1.8, alpha=0.45, label="road left edge")
        elif len(left_road) == 1:
            ax.scatter(left_road[:, 0], left_road[:, 1], s=36, color="tab:blue", alpha=0.55, label="road left edge")
        if len(right_road) >= 2:
            ax.plot(right_road[:, 0], right_road[:, 1], color="tab:red", linewidth=1.8, alpha=0.45, label="road right edge")
        elif len(right_road) == 1:
            ax.scatter(right_road[:, 0], right_road[:, 1], s=36, color="tab:red", alpha=0.55, label="road right edge")

    def plot_side(items, outline_color, anchor_color, inner_color, label_prefix):
        if not items:
            return
        for item in items:
            outline = as_xy_array(item.get("outline", []))
            if len(outline) >= 2:
                ax.plot(outline[:, 0], outline[:, 1], color=outline_color, linewidth=2.2, alpha=0.9)
            elif len(outline) == 1:
                ax.scatter(outline[:, 0], outline[:, 1], s=48, color=outline_color, alpha=0.9)

            if show_anchors:
                anchors = as_xy_array(item.get("anchors", []))
                if len(anchors) != 0:
                    ax.scatter(anchors[:, 0], anchors[:, 1], s=24, color=anchor_color, alpha=0.85)

            if show_inner:
                merged_polyline = as_xy_array(item.get("polyline", []))
                if len(merged_polyline) >= 2:
                    ax.plot(merged_polyline[:, 0], merged_polyline[:, 1], color=inner_color, linewidth=2.4, alpha=0.95)
                if show_history_inner:
                    for polyline in item.get("inner_polylines", []):
                        arr = as_xy_array(polyline)
                        if len(arr) >= 2:
                            ax.plot(arr[:, 0], arr[:, 1], color=inner_color, linewidth=1.0, alpha=0.25)

            if show_indices:
                centroid = as_xy_array(item.get("centroid", []))
                if len(centroid) == 1:
                    source_indices = item.get("source_indices", [])
                    text = ",".join(str(v) for v in source_indices[:5])
                    if len(source_indices) > 5:
                        text += "..."
                    ax.text(centroid[0, 0], centroid[0, 1], text, fontsize=7, color="0.2")

        ax.plot([], [], color=outline_color, linewidth=2.2, label=f"{label_prefix} fused outline")
        if show_anchors:
            ax.scatter([], [], s=24, color=anchor_color, label=f"{label_prefix} anchors")
        if show_inner:
            ax.plot([], [], color=inner_color, linewidth=1.6, label=f"{label_prefix} inner polylines")

    plot_side(data.get("left_sidewalks", []), "tab:blue", "navy", "tab:cyan", "left")
    plot_side(data.get("right_sidewalks", []), "tab:red", "darkorange", "tab:orange", "right")

    ax.set_title(
        "Sidewalk Road-Fused Contours\n"
        f"left={len(data.get('left_sidewalks', []))} right={len(data.get('right_sidewalks', []))}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fused sidewalk road-fused contours.")
    parser.add_argument(
        "-i",
        "--input",
        default="demo_output/sidewalk_roadfused/sidewalk_roadfused_records_fused.json",
        help="Path to sidewalk_roadfused_records_fused.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input_stem>.png",
    )
    parser.add_argument(
        "--hide-anchors",
        action="store_true",
        help="Do not draw anchor points.",
    )
    parser.add_argument(
        "--hide-road",
        action="store_true",
        help="Do not draw road center/edge references from the raw records.",
    )
    parser.add_argument(
        "--hide-inner",
        action="store_true",
        help="Do not draw fused polylines.",
    )
    parser.add_argument(
        "--show-history-inner",
        action="store_true",
        help="Also draw source inner polylines with low alpha.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Annotate fused contours with source record indices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")

    data = load_records(input_path)
    raw_records = load_raw_records_for_fused(input_path, data)
    plot_records(
        data,
        output_path,
        raw_records=raw_records,
        show_road=not args.hide_road,
        show_anchors=not args.hide_anchors,
        show_inner=not args.hide_inner,
        show_history_inner=args.show_history_inner,
        show_indices=args.show_indices,
    )
    print(f"saved fused preview image to {output_path}")


if __name__ == "__main__":
    main()
