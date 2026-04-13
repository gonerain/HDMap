#!/usr/bin/python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.map_geometry import filter_geometries_by_distance
from src.map_geometry import load_map_geometries
from src.projection import draw_projected_polylines
from src.projection import load_frame_projection_context
from src.projection import project_world_polyline_to_image
from src.projection import save_output_image


def render_map_overlay(map_items, context):
    projected_items = []
    for item in map_items:
        projected_items.append(
            {
                "name": item["name"],
                "projected": project_world_polyline_to_image(item["points_xyz"], context),
                "color": item["color"],
                "thickness": item["thickness"],
            }
        )
    overlay = draw_projected_polylines(context["image"], projected_items)
    return overlay, projected_items


def render_frame_report(geometry_bundle, frame_index, config_path, trajectory_path, images_dir, max_distance, output_image):
    items = geometry_bundle["items"]
    context = load_frame_projection_context(
        frame_index,
        config_path=config_path,
        trajectory_path=trajectory_path,
        images_dir=images_dir,
    )
    items = filter_geometries_by_distance(items, context["lidar_to_world"][:3, 3], max_distance)
    overlay, projected_items = render_map_overlay(items, context)
    output_path = save_output_image(output_image, overlay)
    return {
        "frame_index": int(frame_index),
        "image_path": context["image_path"],
        "output_path": str(output_path),
        "drawn_items": [
            {
                "name": item["name"],
                "point_count": int(len(item["projected"]["pixels"])),
                "visible_points": int(np.count_nonzero(item["projected"]["visible_mask"])),
                "visible_pixel_bbox": projected_item_visible_bbox(item["projected"]),
            }
            for item in projected_items
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Render fused map polylines on the origin image.")
    parser.add_argument("-i", "--input", required=True, help="Path to fused map json.")
    parser.add_argument("-f", "--frame-index", type=int, default=0, help="0-based frame index.")
    parser.add_argument("--frame-start", type=int, default=None, help="0-based start frame for batch rendering.")
    parser.add_argument("--frame-end", type=int, default=None, help="0-based inclusive end frame for batch rendering.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step for batch rendering.")
    parser.add_argument("-c", "--config", default="config/outdoor_config.json", help="Projection config path.")
    parser.add_argument("--trajectory", default="result/outdoor/pose.csv", help="Path to pose.csv.")
    parser.add_argument("--images-dir", default="result/outdoor/originpics", help="Directory containing saved origin images.")
    parser.add_argument("-o", "--output-image", default="demo_output/map_overlay.png", help="Overlay output path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch rendering. Defaults to sibling folder of --output-image.")
    parser.add_argument("--max-distance", type=float, default=80.0, help="Keep only map points within this XY distance of the current pose. <=0 disables.")
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_bundle = load_map_geometries(args.input)
    report = {
        "input": str(Path(args.input)),
        "source_type": geometry_bundle["source_type"],
    }

    if args.frame_start is None and args.frame_end is None:
        frame_report = render_frame_report(
            geometry_bundle,
            args.frame_index,
            config_path=args.config,
            trajectory_path=args.trajectory,
            images_dir=args.images_dir,
            max_distance=args.max_distance,
            output_image=args.output_image,
        )
        report.update(frame_report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    frame_start = args.frame_index if args.frame_start is None else int(args.frame_start)
    frame_end = args.frame_index if args.frame_end is None else int(args.frame_end)
    frame_step = max(int(args.frame_step), 1)
    if frame_end < frame_start:
        raise ValueError(f"frame_end must be >= frame_start, got {frame_end} < {frame_start}")

    output_image = Path(args.output_image)
    output_dir = Path(args.output_dir) if args.output_dir else output_image.with_suffix("")
    frame_reports = []
    for frame_index in range(frame_start, frame_end + 1, frame_step):
        frame_output = output_dir / f"map_overlay_frame_{frame_index:03d}.png"
        frame_reports.append(
            render_frame_report(
                geometry_bundle,
                frame_index,
                config_path=args.config,
                trajectory_path=args.trajectory,
                images_dir=args.images_dir,
                max_distance=args.max_distance,
                output_image=frame_output,
            )
        )
    report.update(
        {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_step": frame_step,
            "output_dir": str(output_dir),
            "frames": frame_reports,
        }
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def projected_item_visible_bbox(projected):
    pixels = np.asarray(projected["pixels"], dtype=np.float64)
    visible_mask = np.asarray(projected["visible_mask"], dtype=bool)
    visible_pixels = pixels[visible_mask]
    if len(visible_pixels) == 0:
        return None
    return {
        "u_min": float(visible_pixels[:, 0].min()),
        "u_max": float(visible_pixels[:, 0].max()),
        "v_min": float(visible_pixels[:, 1].min()),
        "v_max": float(visible_pixels[:, 1].max()),
    }


if __name__ == "__main__":
    main()
