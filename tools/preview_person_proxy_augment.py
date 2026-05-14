"""Side-by-side BEV comparison of sidewalk LiDAR points BEFORE vs AFTER the
ArUco-walker person-proxy augmentation.

Replays the same logic as core/sidewalk_process.py:_augment_with_person_proxy
per-frame, accumulates the virtual (promoted) points across all frames, and
renders one image with TWO panels:

  LEFT:  real class=15 LiDAR only (what's in outdoor.pkl)
  RIGHT: real (gray) + virtual (red) overlay -> what vectorize actually sees
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--aruco-json", required=True,
                   help="Output of tools/detect_aruco_walkers.py")
    p.add_argument("--output", required=True)
    p.add_argument("--sidewalk-class", type=int, default=15)
    p.add_argument("--person-class", type=int, default=19)
    p.add_argument("--walker-radius-m", type=float, default=0.8)
    p.add_argument("--sw-search-radius-m", type=float, default=2.0)
    p.add_argument("--min-sw-count", type=int, default=30)
    p.add_argument("--bev-res-m", type=float, default=0.1)
    p.add_argument("--margin-m", type=float, default=2.0)
    return p.parse_args()


def load_pkl(path):
    frames = []
    with open(path, "rb") as f:
        while True:
            try:
                frames.append(np.asarray(pickle.load(f), dtype=np.float64))
            except EOFError:
                break
    return frames


def augment_frame(arr, walkers_xy, sw_class, person_class,
                  walker_radius, sw_search_radius, min_sw_count):
    """Return (real_sidewalk_xy, virtual_sidewalk_xy) for one frame."""
    sw_mask = arr[:, 3] == sw_class
    pp_mask = arr[:, 3] == person_class
    sw_pts = arr[sw_mask]
    real_xy = sw_pts[:, :2]
    if not pp_mask.any() or len(sw_pts) < min_sw_count or len(walkers_xy) == 0:
        return real_xy, np.empty((0, 2))
    pp_pts = arr[pp_mask]
    walker_xy_a = np.asarray(walkers_xy, dtype=np.float64)
    diffs = pp_pts[:, None, :2] - walker_xy_a[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    near_walker = (d2.min(axis=1) <= walker_radius ** 2)
    if not near_walker.any():
        return real_xy, np.empty((0, 2))
    tree = cKDTree(sw_pts[:, :2])
    dist, _ = tree.query(pp_pts[near_walker, :2], k=1)
    near_sw = dist <= sw_search_radius
    if not near_sw.any():
        return real_xy, np.empty((0, 2))
    idx = np.where(near_walker)[0][near_sw]
    virtual_xy = pp_pts[idx, :2]
    return real_xy, virtual_xy


def world_to_canvas(pts_xy, x_min, y_min, res, h_px):
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    col = ((pts[:, 0] - x_min) / res).astype(np.int32)
    row = (h_px - 1 - (pts[:, 1] - y_min) / res).astype(np.int32)
    return np.column_stack([col, row])


def main():
    args = parse_args()
    frames = load_pkl(args.pkl)
    poses = np.loadtxt(args.pose, delimiter=",")
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    aruco = json.load(open(args.aruco_json))
    walker_map = {
        int(k): [tuple(map(float, v["world_xy"])) for v in lst]
        for k, lst in aruco.get("frames", {}).items()
    }

    real_all, virtual_all = [], []
    n_virtual_total = 0
    for fi, arr in enumerate(frames):
        if arr is None or len(arr) == 0:
            continue
        walkers = walker_map.get(fi, [])
        real_xy, virtual_xy = augment_frame(
            arr, walkers,
            int(args.sidewalk_class), int(args.person_class),
            float(args.walker_radius_m), float(args.sw_search_radius_m),
            int(args.min_sw_count),
        )
        if len(real_xy):
            real_all.append(real_xy)
        if len(virtual_xy):
            virtual_all.append(virtual_xy)
            n_virtual_total += len(virtual_xy)
    real_pts = np.vstack(real_all) if real_all else np.empty((0, 2))
    virtual_pts = np.vstack(virtual_all) if virtual_all else np.empty((0, 2))
    print(f"real sidewalk pts: {len(real_pts):,}")
    print(f"virtual sidewalk pts (after aruco proxy): {len(virtual_pts):,} "
          f"(+{100.0 * len(virtual_pts) / max(len(real_pts), 1):.1f}%)")

    # BEV canvas
    all_xy = np.vstack([p for p in [real_pts, virtual_pts, poses[:, :2]] if len(p)])
    margin = float(args.margin_m)
    res = float(args.bev_res_m)
    x_min = all_xy[:, 0].min() - margin
    y_min = all_xy[:, 1].min() - margin
    x_max = all_xy[:, 0].max() + margin
    y_max = all_xy[:, 1].max() + margin
    grid_w = int(np.ceil((x_max - x_min) / res))
    grid_h = int(np.ceil((y_max - y_min) / res))

    def make_panel(title, draw_real, draw_virtual):
        img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        if draw_real and len(real_pts):
            pix = world_to_canvas(real_pts, x_min, y_min, res, grid_h)
            ok = ((pix[:, 0] >= 0) & (pix[:, 0] < grid_w)
                  & (pix[:, 1] >= 0) & (pix[:, 1] < grid_h))
            img[pix[ok, 1], pix[ok, 0]] = (160, 160, 160)
        if draw_virtual and len(virtual_pts):
            pix = world_to_canvas(virtual_pts, x_min, y_min, res, grid_h)
            ok = ((pix[:, 0] >= 0) & (pix[:, 0] < grid_w)
                  & (pix[:, 1] >= 0) & (pix[:, 1] < grid_h))
            img[pix[ok, 1], pix[ok, 0]] = (0, 80, 255)
        # Trajectory
        traj = world_to_canvas(poses[:, :2], x_min, y_min, res, grid_h)
        ok = ((traj[:, 0] >= 0) & (traj[:, 0] < grid_w)
              & (traj[:, 1] >= 0) & (traj[:, 1] < grid_h))
        for u, v in traj[ok]:
            cv2.circle(img, (int(u), int(v)), 1, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    left = make_panel(f"BEFORE  real={len(real_pts):,}", True, False)
    right = make_panel(
        f"AFTER  real={len(real_pts):,} + virtual={len(virtual_pts):,}", True, True
    )
    gap = np.zeros((grid_h, 6, 3), dtype=np.uint8)
    gap[:, :] = (60, 60, 60)
    combined = np.hstack([left, gap, right])
    cv2.imwrite(args.output, combined)
    print(f"wrote {args.output} ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == "__main__":
    main()
