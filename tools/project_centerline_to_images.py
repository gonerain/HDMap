"""Project the final smoothed sidewalk centerline polyline back onto camera frames.

For each Nth frame: project every centerline node (ENU XY + a single
shared sidewalk z) into the camera, draw line segments between adjacent
nodes (open polyline, not a closed polygon) and a small dot at each
node. Optionally overlay sidewalk-class LiDAR points (green) and the
Mask2Former sidewalk mask (blue tint) so you can eyeball whether the
vector lies on top of the actual sidewalk in image space.

Use this to verify that the vectorize -> trunk-extract -> width-smooth
pipeline did not drift away from the LiDAR/segmentation evidence.
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
    p.add_argument("--centerline", required=True,
                   help="JSON with measurements[].p_xy (e.g. width_sw0_smooth.json)")
    p.add_argument("--pose", required=True)
    p.add_argument("--originpics", required=True)
    p.add_argument("--pkl", default=None,
                   help="outdoor.pkl; if set, overlay sidewalk-class LiDAR dots and "
                        "auto-derive sidewalk_z from the per-frame median if not given.")
    p.add_argument("--sempics", default=None)
    p.add_argument("--sidewalk-z", type=float, default=None,
                   help="Shared ground height (ENU m) to lift centerline nodes to. "
                        "Default: median of class=15 z in the pkl, else 0.")
    p.add_argument("--sidewalk-class", type=int, default=15)
    p.add_argument("--seg-alpha", type=float, default=0.30)
    p.add_argument("--seg-color-bgr", type=int, nargs=3, default=[255, 100, 0])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--max-cam-depth-m", type=float, default=20.0)
    p.add_argument("--min-cam-depth-m", type=float, default=0.5)
    p.add_argument("--camera-model", default=None, choices=[None, "pinhole", "equidistant"])
    p.add_argument("--node-radius", type=int, default=5)
    p.add_argument("--edge-thickness", type=int, default=3)
    p.add_argument("--edge-color-bgr", type=int, nargs=3, default=[0, 255, 255],
                   help="Default = bright yellow.")
    return p.parse_args()


def auto_sidewalk_z(pkl_path, sw_class):
    frames = load_frames(pkl_path)
    zs = []
    for arr in frames:
        if arr is None or len(arr) == 0:
            continue
        m = arr[:, 3].astype(np.int64) == int(sw_class)
        if m.any():
            zs.append(np.asarray(arr[m, 2], dtype=np.float64))
    if not zs:
        return None, frames
    z = float(np.median(np.concatenate(zs)))
    return z, frames


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    K = np.asarray(cfg["intrinsic"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(cfg["distortion_matrix"], dtype=np.float64).reshape(-1, 1)
    model = normalize_camera_model(cfg.get("camera_model"))
    if args.camera_model is not None:
        model = args.camera_model.lower()
    R_cb, t_cb = cam_extrinsics_from_config(cfg)

    cl = json.load(open(args.centerline))
    meas = cl["measurements"]
    P_xy = np.asarray([m["p_xy"] for m in meas], dtype=np.float64)
    T_xy = np.asarray([m["tangent"] for m in meas], dtype=np.float64)
    # Build the left/right boundary polylines the same way
    # tools/sidewalk_export_geodetic.py does: anchor at trunk, offset by
    # smoothed half-width along the in-plane normal. NaN widths (no
    # measurement at that node) drop their boundary point.
    w_sm = np.asarray(
        [m["w_smooth_m"] if m.get("w_smooth_m") is not None else np.nan
         for m in meas], dtype=np.float64)
    if np.isnan(w_sm).all():
        w_sm = np.asarray(
            [m["w_m"] if m.get("w_m") is not None else np.nan
             for m in meas], dtype=np.float64)
    half = w_sm / 2.0
    n_left = np.column_stack([-T_xy[:, 1], T_xy[:, 0]])
    L_xy = P_xy + n_left * half[:, None]
    R_xy = P_xy - n_left * half[:, None]
    boundary_valid = ~np.isnan(w_sm)
    print(f"centerline nodes: {len(P_xy)}  "
          f"(boundary-valid: {int(boundary_valid.sum())})")

    frames_pkl = None
    sw_z = args.sidewalk_z
    if args.pkl:
        if sw_z is None:
            sw_z, frames_pkl = auto_sidewalk_z(args.pkl, args.sidewalk_class)
            print(f"sidewalk_z auto = {sw_z:.3f} m (median of class={args.sidewalk_class} in pkl)")
        else:
            frames_pkl = load_frames(args.pkl)
    if sw_z is None:
        sw_z = 0.0
        print("sidewalk_z fallback = 0.0 (no pkl + no --sidewalk-z given)")
    z_col = np.full(len(P_xy), float(sw_z))
    nodes_xyz = np.column_stack([P_xy, z_col])
    L_xyz = np.column_stack([L_xy, z_col])
    R_xyz = np.column_stack([R_xy, z_col])

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

    center_c = tuple(int(c) for c in args.edge_color_bgr)  # yellow centerline
    left_c = (255, 255, 0)      # cyan  = left boundary
    right_c = (255, 0, 255)     # magenta = right boundary
    node_c = (0, 0, 255)        # red node markers
    lidar_c = (180, 180, 180)   # dim gray LiDAR dots (boundary colors pop more)

    fids = list(range(start, end + 1, int(args.stride)))
    n_drawn = 0
    for fid in fids:
        img = cv2.imread(str(orig_dir / f"{fid:06d}.png"))
        if img is None:
            continue
        H, W = img.shape[:2]
        R_cw, t_cw = pose_to_cam_world(poses[fid - 1], R_cb, t_cb)

        # 1) Optional segmentation tint.
        if sem_dir is not None:
            sem = cv2.imread(str(sem_dir / f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
            if sem is not None:
                if sem.ndim == 3:
                    sem = sem[:, :, 0]
                m = (sem == int(args.sidewalk_class))
                if m.any():
                    tint = np.zeros_like(img)
                    tint[m] = tuple(int(c) for c in args.seg_color_bgr)
                    a = float(args.seg_alpha)
                    img[m] = (img[m].astype(np.float32) * (1 - a)
                              + tint[m].astype(np.float32) * a).astype(np.uint8)

        # 2) Optional per-frame LiDAR sidewalk points.
        if frames_pkl is not None and (fid - 1) < len(frames_pkl):
            arr = frames_pkl[fid - 1]
            if arr is not None and len(arr) > 0:
                m = arr[:, 3].astype(np.int64) == int(args.sidewalk_class)
                if m.any():
                    uv, _, z = project_world_to_image(arr[m, :3], R_cw, t_cw, K, D, model)
                    ok = (np.isfinite(uv[:, 0])
                          & (z >= float(args.min_cam_depth_m))
                          & (z <= float(args.max_cam_depth_m))
                          & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                          & (uv[:, 1] >= 0) & (uv[:, 1] < H))
                    for u, v in np.round(uv[ok]).astype(np.int32):
                        cv2.circle(img, (int(u), int(v)), 2, lidar_c, -1, cv2.LINE_AA)

        def project_and_draw_polyline(world_xyz, color, thickness,
                                      node_mask=None, draw_nodes=False):
            """Project a polyline and draw open-edge segments. Returns
            (n_edges_drawn, total_segments)."""
            uv, _, z = project_world_to_image(world_xyz, R_cw, t_cw, K, D, model)
            valid = (np.isfinite(uv[:, 0])
                     & (z >= float(args.min_cam_depth_m))
                     & (z <= float(args.max_cam_depth_m)))
            if node_mask is not None:
                valid = valid & node_mask
            uv_safe = np.where(valid[:, None], uv, 0.0)
            uv_i = np.round(uv_safe).astype(np.int32)
            n_e = 0
            n_total = len(world_xyz) - 1
            for i in range(n_total):
                if valid[i] and valid[i + 1]:
                    pa = (int(uv_i[i, 0]), int(uv_i[i, 1]))
                    pb = (int(uv_i[i + 1, 0]), int(uv_i[i + 1, 1]))
                    cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
                    n_e += 1
            if draw_nodes:
                for i in range(len(world_xyz)):
                    if valid[i]:
                        u, v = int(uv_i[i, 0]), int(uv_i[i, 1])
                        if 0 <= u < W and 0 <= v < H:
                            cv2.circle(img, (u, v), args.node_radius,
                                       node_c, -1, cv2.LINE_AA)
            return n_e, n_total

        # 3a) Centerline (with node markers)
        nc, nt = project_and_draw_polyline(
            nodes_xyz, center_c, args.edge_thickness, draw_nodes=True)
        # 3b) Left + right boundary polylines (no node markers; thinner stroke)
        nL, _ = project_and_draw_polyline(
            L_xyz, left_c, max(1, args.edge_thickness - 1),
            node_mask=boundary_valid)
        nR, _ = project_and_draw_polyline(
            R_xyz, right_c, max(1, args.edge_thickness - 1),
            node_mask=boundary_valid)

        legend_parts = ["yellow=centerline", "cyan=L-bound", "magenta=R-bound",
                        "red=node"]
        if sem_dir is not None: legend_parts.append("blue=seg")
        if frames_pkl is not None: legend_parts.append("gray=LiDAR")
        hdr = (f"fid={fid}  C={nc}/{nt}  L={nL}/{nt}  R={nR}/{nt}  "
               f"z={sw_z:.2f}m  [{' '.join(legend_parts)}]")
        cv2.putText(img, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"hdmap_proj_{fid:06d}.png"), img)
        n_drawn += 1
    print(f"wrote {n_drawn} images to {out_dir}")


if __name__ == "__main__":
    main()
