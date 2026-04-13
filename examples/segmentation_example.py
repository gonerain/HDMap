#!/usr/bin/python3
import argparse
from pathlib import Path

import cv2
import numpy as np

from predict import get_colors
from predict import get_predict_func_detectron
from predict import get_predict_func_mmsegmentation


def build_predictor(backend, config_path, model_path):
    if backend == "detectron2":
        return get_predict_func_detectron(config_path, model_path)
    if backend == "mmseg":
        return get_predict_func_mmsegmentation(config_path, model_path)
    raise ValueError(f"unsupported backend: {backend}")


def colorize_mask(mask, cmap):
    colors = np.asarray(get_colors(cmap), dtype=np.uint8)
    max_label = int(mask.max()) if mask.size != 0 else 0
    if max_label >= len(colors):
        pad = np.zeros((max_label + 1 - len(colors), 3), dtype=np.uint8)
        colors = np.vstack((colors, pad))
    return colors[mask]


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal semantic segmentation example.")
    parser.add_argument("--backend", choices=("detectron2", "mmseg"), default="detectron2")
    parser.add_argument("--config", required=True, help="Model config file path.")
    parser.add_argument("--model", required=True, help="Model weight file path.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default="demo_output/segmentation_example", help="Where to save outputs.")
    parser.add_argument("--cmap", choices=("mapillary", "ade20k", "cityscapes"), default="mapillary")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend weight.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    predictor = build_predictor(args.backend, args.config, args.model)
    mask = np.asarray(predictor(image), dtype=np.uint8)
    color_mask = colorize_mask(mask, args.cmap)
    overlay = cv2.addWeighted(image, 1.0 - float(args.alpha), color_mask, float(args.alpha), 0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"failed to write mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"failed to write overlay: {overlay_path}")

    print("Segmentation example finished.")
    print(f"input_image: {image_path}")
    print(f"mask_output: {mask_path}")
    print(f"overlay_output: {overlay_path}")


if __name__ == "__main__":
    main()
