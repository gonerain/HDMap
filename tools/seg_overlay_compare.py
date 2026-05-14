"""Side-by-side semantic-overlay comparison between two outdoor_livox_ie runs.

For each sampled frame index, loads originpics/NNNNNN.png + sempics/NNNNNN.png
from both run dirs, blends the semantic class image (colored via mapillary
cmap) on top of the original, and writes baseline | optimized side-by-side.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import get_colors  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", required=True, help="Baseline run dir (has originpics/ + sempics/).")
    p.add_argument("--run-b", required=True, help="Optimized run dir.")
    p.add_argument("--label-a", default="baseline")
    p.add_argument("--label-b", default="optimized")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frames", default="1,10,20,30,40,50",
                   help="Comma list of frame indices to export.")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Semantic overlay weight (0=image only, 1=color only).")
    p.add_argument("--highlight-class", type=int, default=15,
                   help="Class to outline brightly (sidewalk=15). -1 disables.")
    p.add_argument("--cmap", default="mapillary")
    return p.parse_args()


def colorize(class_img, colors):
    h, w = class_img.shape
    rgb = colors[class_img.flatten()].reshape(h, w, 3).astype(np.uint8)
    return rgb[:, :, ::-1]  # mapillary cmap returns RGB, we want BGR for cv2


def overlay(orig_bgr, sem_class, colors, alpha, highlight_cls):
    sem_bgr = colorize(sem_class, colors)
    blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, sem_bgr, alpha, 0.0)
    if highlight_cls >= 0:
        mask = (sem_class == highlight_cls).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)
    return blended


def label_panel(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def load_pair(run_dir, fi):
    op = Path(run_dir) / "originpics" / f"{fi:06d}.png"
    sp = Path(run_dir) / "sempics" / f"{fi:06d}.png"
    if not op.exists() or not sp.exists():
        return None
    orig = cv2.imread(str(op), cv2.IMREAD_COLOR)
    sem = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    return orig, sem


def class_stats(sem, cls):
    return int((sem == cls).sum())


def main():
    args = parse_args()
    colors = np.asarray(get_colors(args.cmap), dtype=np.uint8)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(s) for s in args.frames.split(",") if s.strip()]
    written = 0
    for fi in indices:
        a = load_pair(args.run_a, fi)
        b = load_pair(args.run_b, fi)
        if a is None or b is None:
            print(f"frame {fi}: missing in {args.run_a if a is None else args.run_b}, skip")
            continue
        orig_a, sem_a = a
        orig_b, sem_b = b
        if orig_a.shape != orig_b.shape:
            print(f"frame {fi}: shape mismatch {orig_a.shape} vs {orig_b.shape}, skip")
            continue
        h, w = orig_a.shape[:2]
        ov_a = overlay(orig_a, sem_a, colors, args.alpha, args.highlight_class)
        ov_b = overlay(orig_b, sem_b, colors, args.alpha, args.highlight_class)

        sw_a = class_stats(sem_a, args.highlight_class)
        sw_b = class_stats(sem_b, args.highlight_class)
        delta = 100.0 * (sw_b - sw_a) / max(sw_a, 1)
        title_a = f"{args.label_a}  frame {fi}  sw_px={sw_a:,}"
        title_b = f"{args.label_b}  sw_px={sw_b:,}  ({delta:+.1f}%)"

        panel_a = label_panel(ov_a, title_a)
        panel_b = label_panel(ov_b, title_b)
        gap = np.full((h, 6, 3), 60, dtype=np.uint8)
        combined = np.hstack([panel_a, gap, panel_b])

        out_path = out_dir / f"compare_{fi:06d}.png"
        cv2.imwrite(str(out_path), combined)
        written += 1
        print(f"wrote {out_path} (sidewalk: {sw_a:,} -> {sw_b:,}, {delta:+.1f}%)")
    print(f"\n{written}/{len(indices)} frames written to {out_dir}")


if __name__ == "__main__":
    main()
