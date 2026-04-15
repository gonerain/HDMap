#!/usr/bin/python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect one saved frame with the segmentation model and matching semantic point-cloud stats."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/outdoor_config.json",
        help="Config file path.",
    )
    parser.add_argument(
        "-f",
        "--frame-index",
        required=True,
        type=int,
        help="1-based saved frame index, matching originpics/000001.png style naming.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <save_folder>/inspect_seg/frame_XXXXXX",
    )
    parser.add_argument(
        "--alpha",
        default=0.45,
        type=float,
        help="Overlay blend weight for the segmentation color mask.",
    )
    parser.add_argument(
        "--topk",
        default=12,
        type=int,
        help="How many top classes to print for mask pixels and semantic points.",
    )
    parser.add_argument(
        "--skip-pkl",
        action="store_true",
        help="Skip reading the semantic point-cloud pkl.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Only highlight these class ids in extra outputs, such as 13 15 24.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def load_mapillary_metadata():
    class_path = ROOT / "imseg" / "mask2former" / "class.json"
    if not class_path.exists():
        return {}, None
    with class_path.open("r") as f:
        payload = json.load(f)
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return {}, None

    label_names = {}
    colors = np.zeros((max(len(labels), 1), 3), dtype=np.uint8)
    for idx, item in enumerate(labels):
        if not isinstance(item, dict):
            continue
        label_names[idx] = str(item.get("readable") or item.get("name") or idx)
        color = item.get("color")
        if isinstance(color, list) and len(color) == 3:
            colors[idx] = np.asarray(color, dtype=np.uint8)
    return label_names, colors


def load_label_names(cmap_name):
    if cmap_name == "mapillary":
        label_names, _ = load_mapillary_metadata()
        return label_names
    return {}


def load_colors(config):
    cmap_name = config["cmap"]
    if cmap_name == "mapillary":
        _, colors = load_mapillary_metadata()
        if colors is not None:
            return colors

    try:
        import predict
    except Exception:
        return np.zeros((256, 3), dtype=np.uint8)
    return np.asarray(predict.get_colors(cmap_name), dtype=np.uint8)


def run_predictor(config, image):
    import predict

    predictor = getattr(predict, config["predict_func"])(config["model_config"], config["model_file"])
    return np.asarray(predictor(image), dtype=np.int32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_output_dir(requested_output_dir, save_folder, frame_index):
    if requested_output_dir is not None:
        path = Path(requested_output_dir)
        ensure_dir(path)
        return path

    candidates = [
        Path(save_folder) / "inspect_seg" / f"frame_{frame_index:06d}",
        Path("/tmp/hdmap_inspect_seg") / f"frame_{frame_index:06d}",
    ]
    last_error = None
    for path in candidates:
        try:
            ensure_dir(path)
            return path
        except OSError as exc:
            last_error = exc
            continue
    raise last_error or RuntimeError("failed to create any output directory")


def colorize_mask(mask, colors):
    colors = np.asarray(colors, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.int64)
    if colors.shape[0] == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)
    clipped = np.clip(mask, 0, colors.shape[0] - 1)
    return colors[clipped]


def colorize_mask_subset(mask, colors, class_ids):
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if not class_ids:
        return out
    class_ids = [int(v) for v in class_ids]
    valid = np.isin(mask, np.asarray(class_ids, dtype=np.int64))
    if not np.any(valid):
        return out
    colors = np.asarray(colors, dtype=np.uint8)
    clipped = np.clip(mask[valid], 0, colors.shape[0] - 1)
    out[valid] = colors[clipped]
    return out


def summarize_ids(ids):
    ids = np.asarray(ids)
    if ids.size == 0:
        return []
    uniq, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts)
    return [(int(uniq[i]), int(counts[i])) for i in order]


def format_hist(title, hist, total, label_names, topk):
    lines = [title]
    if not hist:
        lines.append("  <none>")
        return "\n".join(lines)
    for class_id, count in hist[:topk]:
        ratio = count / max(int(total), 1)
        name = label_names.get(class_id)
        suffix = f" ({name})" if name else ""
        lines.append(f"  class {class_id:>3}: {count:>10}  {ratio:>7.3%}{suffix}")
    return "\n".join(lines)


def format_class_list(class_ids, label_names):
    if not class_ids:
        return "<none>"
    items = []
    for class_id in class_ids:
        name = label_names.get(int(class_id))
        items.append(f"{int(class_id)} ({name})" if name else str(int(class_id)))
    return ", ".join(items)


def load_semantic_frame_points(pkl_path, frame_index):
    with open(pkl_path, "rb") as f:
        for idx in range(1, frame_index + 1):
            try:
                frame = pickle.load(f)
            except EOFError as exc:
                raise IndexError(f"frame {frame_index} exceeds pkl length") from exc
            if idx == frame_index:
                return np.asarray(frame)
    raise IndexError(f"frame {frame_index} exceeds pkl length")


def main():
    args = parse_args()
    if args.frame_index <= 0:
        raise ValueError("--frame-index must be >= 1")

    config = load_config(args.config)
    save_folder = config["save_folder"]
    origin_path = Path(save_folder) / "originpics" / f"{args.frame_index:06d}.png"
    saved_seg_path = Path(save_folder) / "sempics" / f"{args.frame_index:06d}.png"
    pkl_path = Path(save_folder) / ("indoor.pkl" if config["mode"] == "indoor" else "outdoor.pkl")
    output_dir = choose_output_dir(args.output_dir, save_folder, args.frame_index)

    image = cv2.imread(str(origin_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read origin image: {origin_path}")

    colors = load_colors(config)
    label_names = load_label_names(config["cmap"])
    mask_source = None
    predictor_error = None

    try:
        mask = run_predictor(config, image)
        mask_source = "predictor"
    except Exception as exc:
        predictor_error = str(exc)
        saved_seg = cv2.imread(str(saved_seg_path), cv2.IMREAD_UNCHANGED)
        if saved_seg is None:
            raise RuntimeError(
                f"predictor failed ({predictor_error}) and no saved seg image exists at {saved_seg_path}"
            ) from exc
        mask = np.asarray(saved_seg, dtype=np.int32)
        mask_source = "saved_sempics"

    mask_color = colorize_mask(mask, colors)
    overlay = cv2.addWeighted(image, 1.0 - float(args.alpha), mask_color, float(args.alpha), 0.0)
    focus_mask_color = None
    focus_overlay = None
    focus_hist = []
    focus_pixel_count = 0
    if args.classes:
        focus_mask_color = colorize_mask_subset(mask, colors, args.classes)
        focus_overlay = cv2.addWeighted(image, 1.0 - float(args.alpha), focus_mask_color, float(args.alpha), 0.0)
        focus_ids = mask[np.isin(mask, np.asarray(args.classes, dtype=np.int32))]
        focus_hist = summarize_ids(focus_ids)
        focus_pixel_count = int(focus_ids.size)

    cv2.imwrite(str(output_dir / "origin.png"), image)
    cv2.imwrite(str(output_dir / "mask_color.png"), mask_color)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)
    if focus_mask_color is not None:
        cv2.imwrite(str(output_dir / "focus_mask_color.png"), focus_mask_color)
        cv2.imwrite(str(output_dir / "focus_overlay.png"), focus_overlay)
    if saved_seg_path.exists():
        saved_seg = cv2.imread(str(saved_seg_path), cv2.IMREAD_UNCHANGED)
        if saved_seg is not None:
            cv2.imwrite(str(output_dir / "saved_seg_raw.png"), saved_seg)

    mask_hist = summarize_ids(mask.reshape(-1))
    print(f"frame_index: {args.frame_index}")
    print(f"origin: {origin_path}")
    print(f"output_dir: {output_dir}")
    print(f"mask_source: {mask_source}")
    if predictor_error is not None:
        print(f"predictor_error: {predictor_error}")
    print(f"mask_shape: {tuple(mask.shape)}")
    print(format_hist("mask_pixel_histogram:", mask_hist, mask.size, label_names, args.topk))
    if args.classes:
        print("")
        print(f"focus_classes: {format_class_list(args.classes, label_names)}")
        print(format_hist("focus_mask_pixel_histogram:", focus_hist, focus_pixel_count, label_names, args.topk))

    summary = {
        "frame_index": int(args.frame_index),
        "origin_path": str(origin_path),
        "saved_seg_path": str(saved_seg_path) if saved_seg_path.exists() else None,
        "output_dir": str(output_dir),
        "mask_source": mask_source,
        "predictor_error": predictor_error,
        "focus_classes": [int(v) for v in (args.classes or [])],
        "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "mask_classes": [
            {
                "class_id": int(class_id),
                "count": int(count),
                "ratio": float(count / max(int(mask.size), 1)),
                "label": label_names.get(class_id),
            }
            for class_id, count in mask_hist
        ],
    }
    if args.classes:
        summary["focus_mask_classes"] = [
            {
                "class_id": int(class_id),
                "count": int(count),
                "ratio": float(count / max(focus_pixel_count, 1)),
                "label": label_names.get(class_id),
            }
            for class_id, count in focus_hist
        ]

    if not args.skip_pkl and pkl_path.exists():
        try:
            sem_frame = load_semantic_frame_points(pkl_path, args.frame_index)
            point_hist = []
            focus_point_hist = []
            focus_point_count = 0
            xyz_min = None
            xyz_max = None
            if sem_frame.ndim == 2 and sem_frame.shape[0] != 0 and sem_frame.shape[1] >= 4:
                point_ids = np.rint(sem_frame[:, 3]).astype(np.int64, copy=False)
                point_hist = summarize_ids(point_ids)
                if args.classes:
                    focus_point_ids = point_ids[np.isin(point_ids, np.asarray(args.classes, dtype=np.int64))]
                    focus_point_hist = summarize_ids(focus_point_ids)
                    focus_point_count = int(focus_point_ids.size)
                xyz = sem_frame[:, :3].astype(np.float64, copy=False)
                xyz_min = xyz.min(axis=0).tolist()
                xyz_max = xyz.max(axis=0).tolist()
            print("")
            print(f"semantic_pcd_shape: {tuple(sem_frame.shape)}")
            if xyz_min is not None:
                print(f"semantic_pcd_xyz_min: {[round(float(v), 3) for v in xyz_min]}")
                print(f"semantic_pcd_xyz_max: {[round(float(v), 3) for v in xyz_max]}")
            print(format_hist("semantic_point_histogram:", point_hist, len(sem_frame), label_names, args.topk))
            if args.classes:
                print("")
                print(format_hist("focus_semantic_point_histogram:", focus_point_hist, focus_point_count, label_names, args.topk))

            summary["semantic_pcd"] = {
                "path": str(pkl_path),
                "shape": [int(v) for v in sem_frame.shape],
                "xyz_min": xyz_min,
                "xyz_max": xyz_max,
                "classes": [
                    {
                        "class_id": int(class_id),
                        "count": int(count),
                        "ratio": float(count / max(int(len(sem_frame)), 1)),
                        "label": label_names.get(class_id),
                    }
                    for class_id, count in point_hist
                ],
            }
            if args.classes:
                summary["semantic_pcd"]["focus_classes"] = [
                    {
                        "class_id": int(class_id),
                        "count": int(count),
                        "ratio": float(count / max(focus_point_count, 1)),
                        "label": label_names.get(class_id),
                    }
                    for class_id, count in focus_point_hist
                ]
        except Exception as exc:
            print("")
            print(f"semantic_pcd_error: {exc}")
            summary["semantic_pcd_error"] = str(exc)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
