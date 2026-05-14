"""Pre-pass: detect ArUco markers in each originpic and project marker
centers to world ENU XY using the IE pose + camera intrinsics/extrinsics.

Output: JSON with shape
    {"meta": {...},
     "frames": {"<frame_idx_0_based>": [{"id": <int>, "world_xy": [x, y]}, ...], ...}}

Consumed by `core/sidewalk_process.py` (config key
`sidewalk_aruco_walker_map_json`) to gate the person-as-sidewalk-proxy
augmentation: only LiDAR person points whose XY is near an ArUco walker XY
in the same frame are promoted to virtual sidewalk points.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Same constants as the rest of the pipeline.
R_BASE_FROM_LIDAR = np.array(
    [[0.9063, 0.0, 0.4226], [0.0, 1.0, 0.0], [-0.4226, 0.0, 0.9063]],
    dtype=np.float64,
)
T_BASE_FROM_LIDAR_M = np.array([0.0315, 0.0, 0.1314], dtype=np.float64)
T_BASE_FROM_SPAN_M = np.array([-0.1854, 0.0, -0.242], dtype=np.float64)
R_LIDAR_FROM_BASE = R_BASE_FROM_LIDAR.T
T_FLU_LIDAR_FROM_SPAN = T_BASE_FROM_LIDAR_M - T_BASE_FROM_SPAN_M


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="outdoor_config JSON with intrinsics + extrinsics.")
    p.add_argument("--pose", required=True, help="pose.csv (per-frame).")
    p.add_argument("--originpics", required=True)
    p.add_argument("--output", required=True, help="JSON output.")
    p.add_argument("--dictionary", default="DICT_5X5_100",
                   choices=sorted(ARUCO_DICT_MAP.keys()))
    p.add_argument("--marker-length-m", type=float, default=0.085)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args()


def quat_to_rotmat(qx, qy, qz, qw):
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_world_from_cam(pose_row, R_bc, t_bc):
    """Return (R_wc, t_wc): world (ENU) <- camera."""
    tx, ty, tz, qx, qy, qz, qw = pose_row
    R_wb = quat_to_rotmat(qx, qy, qz, qw)
    t_wb = np.array([tx, ty, tz], dtype=np.float64)
    R_wc = R_wb @ R_bc
    t_wc = R_wb @ t_bc + t_wb
    return R_wc, t_wc


def estimate_marker_center_in_camera(corners_2d, marker_length, K, D, model):
    """Estimate marker 3D position in camera frame via solvePnP."""
    half = marker_length / 2.0
    obj_pts = np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float64,
    )
    img_pts = np.asarray(corners_2d, dtype=np.float64).reshape(-1, 2)
    if model == "equidistant":
        # Undistort image points to "pinhole" coordinates, then solvePnP with
        # identity-distortion pinhole.
        und = cv2.fisheye.undistortPoints(
            img_pts.reshape(-1, 1, 2), K, D
        ).reshape(-1, 2)
        und_pix = (und * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]]))
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, und_pix, K, np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D.reshape(-1, 1),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None
    return tvec.flatten()


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    K = np.asarray(cfg["intrinsic"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(cfg["distortion_matrix"], dtype=np.float64).reshape(-1, 1)
    model = str(cfg.get("camera_model", "pinhole")).lower()
    if model in {"fisheye", "equidistantcamera"}:
        model = "equidistant"

    R_lc = np.asarray(cfg["lidar_from_camera_rotation"], dtype=np.float64).reshape(3, 3)
    t_lc = np.asarray(cfg["lidar_from_camera_translation"], dtype=np.float64).reshape(3)
    R_cl = R_lc.T
    t_cl = -R_lc.T @ t_lc
    R_cb = R_cl @ R_LIDAR_FROM_BASE
    t_cb = t_cl - R_cb @ T_FLU_LIDAR_FROM_SPAN
    R_bc = R_cb.T
    t_bc = -R_cb.T @ t_cb

    poses = np.loadtxt(args.pose, delimiter=",")
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    orig_dir = Path(args.originpics)

    dict_id = ARUCO_DICT_MAP[args.dictionary]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    n_total = len(poses)
    start = max(1, int(args.start))
    end = int(args.end) if args.end is not None else n_total
    end = min(end, n_total)

    frames_out = {}
    total_markers = 0
    frames_with_markers = 0
    for fid in range(start, end + 1):
        img_path = orig_dir / f"{fid:06d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        R_wc, t_wc = pose_to_world_from_cam(poses[fid - 1], R_bc, t_bc)
        markers = []
        for marker_id, corner in zip(ids.flatten().tolist(), corners):
            tvec_cam = estimate_marker_center_in_camera(
                corner, float(args.marker_length_m), K, D, model)
            if tvec_cam is None:
                continue
            # Transform center from camera frame to world ENU
            center_world = R_wc @ tvec_cam + t_wc
            markers.append({
                "id": int(marker_id),
                "world_xy": [float(center_world[0]), float(center_world[1])],
                "world_z": float(center_world[2]),
                "cam_depth_m": float(tvec_cam[2]),
            })
        if markers:
            # Frame index in the JSON is 0-based to match
            # SidewalkEdgeProcess._frame_index_counter.
            frames_out[str(fid - 1)] = markers
            total_markers += len(markers)
            frames_with_markers += 1
        if fid % 100 == 0:
            print(f"  fid={fid}: {len(markers)} markers")

    out = {
        "meta": {
            "originpics": str(orig_dir),
            "pose": str(args.pose),
            "config": str(args.config),
            "dictionary": args.dictionary,
            "marker_length_m": float(args.marker_length_m),
            "frames_processed": int(end - start + 1),
            "frames_with_markers": int(frames_with_markers),
            "total_markers": int(total_markers),
        },
        "frames": frames_out,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}: {frames_with_markers}/{end-start+1} frames with markers, "
          f"{total_markers} total detections")


if __name__ == "__main__":
    main()
