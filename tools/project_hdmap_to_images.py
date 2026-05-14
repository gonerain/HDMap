"""Project HDMap sidewalk polygon outline onto camera images.

For every Nth frame, transform the polygon vertices (ENU world coords)
into camera space via IE pose + extrinsics, project to pixels, draw the
closed contour. Edges/points beyond --max-depth are dropped so the
visualization stays meaningful (far points get arbitrarily distorted).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.geometry import (  # noqa: E402
    cam_extrinsics_from_config,
    normalize_camera_model,
    pose_to_cam_world,
    project_world_to_image,
)
from core.pkl_io import load_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--records", required=True,
                   help="Path to sidewalk_records_*.json (must have meta + sidewalks[i].outline + outline_z).")
    p.add_argument("--pose", required=True)
    p.add_argument("--originpics", required=True)
    p.add_argument("--pkl", default=None,
                   help="outdoor.pkl path; if set, overlay sidewalk LiDAR points for each frame.")
    p.add_argument("--sempics", default=None,
                   help="sempics dir; if set, overlay sidewalk segmentation mask as semi-transparent tint.")
    p.add_argument("--sidewalk-class", type=int, default=15)
    p.add_argument("--seg-alpha", type=float, default=0.35,
                   help="Blend alpha for segmentation overlay (0=invisible, 1=opaque).")
    p.add_argument("--seg-color-bgr", type=int, nargs=3, default=[255, 100, 0],
                   help="BGR tint for segmentation overlay (default blue).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--max-cam-depth-m", type=float, default=15.0,
                   help="Drop polygon vertices beyond this camera depth.")
    p.add_argument("--min-cam-depth-m", type=float, default=0.5)
    p.add_argument("--camera-model", default=None,
                   choices=[None, "pinhole", "equidistant"])
    p.add_argument("--flatten-z", action="store_true",
                   help="Replace each vertex's z with sidewalk_z (median ground "
                        "height of the polygon). Fixes the case where per-vertex "
                        "z was picked from non-ground LiDAR (railings, trees).")
    return p.parse_args()


def project(points_world, R_cw, t_cw, K, D, model):
    """Local 2-arg wrapper kept for the tool's legacy call sites."""
    uv, _valid, z = project_world_to_image(points_world, R_cw, t_cw, K, D, model)
    return uv, z


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    K = np.asarray(cfg["intrinsic"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(cfg["distortion_matrix"], dtype=np.float64).reshape(-1, 1)
    model = normalize_camera_model(cfg.get("camera_model"))
    if args.camera_model is not None:
        model = args.camera_model.lower()

    R_cb, t_cb = cam_extrinsics_from_config(cfg)

    records = json.load(open(args.records))
    sidewalks = records.get("sidewalks", [])
    if not sidewalks:
        raise SystemExit("no sidewalks in records")

    # Build world XYZ list per sidewalk and assign a distinct color.
    palette = [
        (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0),
        (255, 0, 255), (255, 255, 0), (255, 255, 255), (128, 0, 255),
    ]
    polys = []
    for k, sw in enumerate(sidewalks):
        outline = np.asarray(sw["outline"], dtype=np.float64)  # (N,2) ENU XY
        z = sw.get("outline_z")
        if z is None or len(z) != len(outline):
            z = [float(sw.get("sidewalk_z", 0.0))] * len(outline)
        if args.flatten_z:
            flat = float(sw.get("sidewalk_z", float(np.median(z))))
            z = [flat] * len(outline)
            print(f"  poly id={sw.get('id', k)}: flattened z -> {flat:.3f} m")
        xyz = np.column_stack([outline, np.asarray(z, dtype=np.float64)])
        polys.append({
            "xyz": xyz,
            "color": palette[k % len(palette)],
            "id": sw.get("id", k),
        })
    print(f"loaded {len(polys)} polygon(s), total vertices = {sum(len(p['xyz']) for p in polys)}")

    poses = np.loadtxt(args.pose, delimiter=",")
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    n_total = len(poses)
    start = max(1, int(args.start))
    end = int(args.end) if args.end is not None else n_total
    end = min(end, n_total)

    orig_dir = Path(args.originpics)
    sem_dir = Path(args.sempics) if args.sempics else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_pkl = None
    if args.pkl:
        frames_pkl = load_frames(args.pkl)
        print(f"loaded {len(frames_pkl)} pkl frames")

    fids = list(range(start, end + 1, int(args.stride)))
    n_drawn = 0
    for fid in fids:
        img = cv2.imread(str(orig_dir / f"{fid:06d}.png"))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        R_cw, t_cw = pose_to_cam_world(poses[fid - 1], R_cb, t_cb)

        # 1) Segmentation overlay (semi-transparent tint).
        if sem_dir is not None:
            sem = cv2.imread(str(sem_dir / f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
            if sem is not None:
                if sem.ndim == 3:
                    sem = sem[:, :, 0]
                sw_mask = (sem == int(args.sidewalk_class))
                if sw_mask.any():
                    tint = np.zeros_like(img)
                    tint[sw_mask] = tuple(int(c) for c in args.seg_color_bgr)
                    alpha = float(args.seg_alpha)
                    img[sw_mask] = (img[sw_mask].astype(np.float32) * (1 - alpha)
                                    + tint[sw_mask].astype(np.float32) * alpha
                                    ).astype(np.uint8)

        # 2) Per-frame LiDAR sidewalk points (small green dots).
        if frames_pkl is not None and (fid - 1) < len(frames_pkl):
            arr = frames_pkl[fid - 1]
            if arr is not None and len(arr) > 0:
                mask_sw = arr[:, 3].astype(np.int64) == int(args.sidewalk_class)
                if mask_sw.any():
                    pts_world = arr[mask_sw, :3]
                    uv_l, z_l = project(pts_world, R_cw, t_cw, K, D, model)
                    ok = (
                        np.isfinite(uv_l[:, 0])
                        & (z_l >= float(args.min_cam_depth_m))
                        & (z_l <= float(args.max_cam_depth_m))
                        & (uv_l[:, 0] >= 0) & (uv_l[:, 0] < img_w)
                        & (uv_l[:, 1] >= 0) & (uv_l[:, 1] < img_h)
                    )
                    uv_l_int = np.round(uv_l[ok]).astype(np.int32)
                    for u, v in uv_l_int:
                        cv2.circle(img, (int(u), int(v)), 2, (0, 255, 0), -1, cv2.LINE_AA)

        # 3) HDMap polygon edges/vertices (existing path).
        for poly in polys:
            uv, z = project(poly["xyz"], R_cw, t_cw, K, D, model)
            valid = (
                np.isfinite(uv[:, 0])
                & (z >= float(args.min_cam_depth_m))
                & (z <= float(args.max_cam_depth_m))
            )
            # Draw edges only when BOTH endpoints are valid (avoid streaks
            # across the image).
            uv_int = np.round(uv).astype(np.int32)
            n = len(poly["xyz"])
            c = poly["color"]
            for i in range(n):
                j = (i + 1) % n
                if valid[i] and valid[j]:
                    pa = (int(uv_int[i, 0]), int(uv_int[i, 1]))
                    pb = (int(uv_int[j, 0]), int(uv_int[j, 1]))
                    cv2.line(img, pa, pb, c, 2, cv2.LINE_AA)
                if valid[i]:
                    u, v = int(uv_int[i, 0]), int(uv_int[i, 1])
                    if 0 <= u < img_w and 0 <= v < img_h:
                        cv2.circle(img, (u, v), 4, c, -1, cv2.LINE_AA)
        legend = []
        if sem_dir is not None: legend.append("blue=seg")
        if frames_pkl is not None: legend.append("green=LiDAR")
        legend.append("red=HDMap polygon")
        hdr = f"fid={fid}  max_depth={args.max_cam_depth_m}m  [{' '.join(legend)}]"
        cv2.putText(img, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"hdmap_proj_{fid:06d}.png"), img)
        n_drawn += 1
    print(f"wrote {n_drawn} images to {out_dir}")


if __name__ == "__main__":
    main()
