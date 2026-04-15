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
    if not isinstance(data, dict):
        raise ValueError(f"expected dict-like fused sidewalk json in {path}, got {type(data).__name__}")
    if "sidewalks" not in data:
        raise ValueError(f"{path} is not a fused sidewalk file with 'sidewalks' key")
    return data


def plot_records(data, output_path, show_history=False, show_indices=False):
    items = data.get("sidewalks", [])
    fig, ax = plt.subplots(figsize=(12, 12))

    outline_color = "goldenrod"
    centroid_color = "tab:red"
    history_color = "tab:blue"

    for idx, item in enumerate(items):
        outline = as_xy_array(item.get("outline", []))
        if len(outline) >= 2:
            ax.plot(
                outline[:, 0],
                outline[:, 1],
                color=outline_color,
                linewidth=2.2,
                alpha=0.9,
                label="fused sidewalk outline" if idx == 0 else None,
            )
        elif len(outline) == 1:
            ax.scatter(
                outline[:, 0],
                outline[:, 1],
                s=36,
                color=outline_color,
                alpha=0.9,
                label="fused sidewalk outline" if idx == 0 else None,
            )

        centroid = as_xy_array(item.get("centroid", []))
        if len(centroid) == 1:
            ax.scatter(
                centroid[:, 0],
                centroid[:, 1],
                s=32,
                color=centroid_color,
                alpha=0.8,
                label="centroid" if idx == 0 else None,
            )
            if show_indices:
                ax.text(
                    centroid[0, 0],
                    centroid[0, 1],
                    f"{item.get('id', idx)}",
                    fontsize=8,
                    color="0.2",
                )

        if show_history:
            for hist in item.get("outlines", []):
                hist_xy = as_xy_array(hist)
                if len(hist_xy) >= 2:
                    ax.plot(
                        hist_xy[:, 0],
                        hist_xy[:, 1],
                        color=history_color,
                        linewidth=0.8,
                        alpha=0.15,
                        label="source outlines" if idx == 0 else None,
                    )
                    history_color = history_color

    ax.set_title(f"Fused Sidewalk Overview\ncount={len(items)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fused sidewalk outlines.")
    parser.add_argument(
        "-i",
        "--input",
        default="demo_output/sidewalk/sidewalk_records_fused.json",
        help="Path to sidewalk_records_fused.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input_stem>.png",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Also draw source outlines with low alpha.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Annotate fused sidewalk ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    data = load_records(input_path)
    plot_records(
        data,
        output_path,
        show_history=args.show_history,
        show_indices=args.show_indices,
    )
    print(f"saved fused sidewalk overview to {output_path}")


if __name__ == "__main__":
    main()
