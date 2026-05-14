"""Common geometry helpers: quaternion-rotation, camera projection,
ECEF/ENU/geodetic conversions, BEV canvas index.

Single source of truth -- imported by outdoor_livox_ie.py, sidewalk_process,
and per-tool scripts so the math stays consistent. WGS84 ellipsoid.
"""
import math

import cv2
import numpy as np

from core.extrinsics import (
    R_BASE_FROM_LIDAR,
    R_LIDAR_FROM_BASE,
    T_BASE_FROM_LIDAR_M,
    T_BASE_FROM_SPAN_M,
    T_FLU_LIDAR_FROM_SPAN,
)


# ---------- Rotations ----------

def quat_to_rotmat(qx, qy, qz, qw):
    """xyzw quaternion -> 3x3 rotation matrix (right-handed, active)."""
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


# ---------- Geodetic <-> ECEF <-> ENU (WGS84) ----------

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat_deg, lon_deg):
    """3x3 R such that enu = R @ (ecef - ecef_origin)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    return np.array(
        [[-sin_lon, cos_lon, 0.0],
         [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
         [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]],
        dtype=np.float64,
    )


def ecef_to_geodetic(ecef):
    """Zhu (1994) closed-form ECEF -> (lat_deg, lon_deg, height_m)."""
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    a, b = _WGS84_A, 6356752.3142
    e2 = 1.0 - (b / a) ** 2
    ep2 = (a / b) ** 2 - 1.0
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(theta) ** 3,
                     p - e2 * a * math.cos(theta) ** 3)
    lon = math.atan2(y, x)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    cos_lat = math.cos(lat)
    h = (p / cos_lat - N) if abs(cos_lat) > 1e-6 \
        else (abs(z) / abs(sin_lat) - N * (1.0 - e2))
    return math.degrees(lat), math.degrees(lon), h


def enu_to_geodetic_batch(pts_enu, ecef_origin, enu_from_ecef):
    """(N, 3) ENU -> (N, 3) lat/lon/h."""
    ecef_from_enu = enu_from_ecef.T
    pts_ecef = (ecef_from_enu @ np.asarray(pts_enu).T).T + ecef_origin
    out = np.zeros_like(pts_ecef)
    for i, e in enumerate(pts_ecef):
        out[i] = ecef_to_geodetic(e)
    return out


# ---------- Camera <-> world ----------

def cam_extrinsics_from_config(cfg):
    """Read lidar_from_camera_rotation/translation from a config dict, return
    R_cb, t_cb (camera <- base) using the project's fixed base/LiDAR/SPAN
    extrinsics."""
    R_lc = np.asarray(cfg["lidar_from_camera_rotation"], dtype=np.float64).reshape(3, 3)
    t_lc = np.asarray(cfg["lidar_from_camera_translation"], dtype=np.float64).reshape(3)
    R_cl = R_lc.T
    t_cl = -R_lc.T @ t_lc
    R_cb = R_cl @ R_LIDAR_FROM_BASE
    t_cb = t_cl - R_cb @ T_FLU_LIDAR_FROM_SPAN
    return R_cb, t_cb


def pose_to_cam_world(pose_row, R_cb, t_cb):
    """IE pose row (tx,ty,tz,qx,qy,qz,qw in ENU/body) -> (R_cw, t_cw)
    such that point_cam = R_cw @ point_world + t_cw."""
    tx, ty, tz, qx, qy, qz, qw = pose_row[:7]
    R_wb = quat_to_rotmat(qx, qy, qz, qw)
    t_wb = np.array([tx, ty, tz], dtype=np.float64)
    R_bw = R_wb.T
    t_bw = -R_bw @ t_wb
    R_cw = R_cb @ R_bw
    t_cw = R_cb @ t_bw + t_cb
    return R_cw, t_cw


def normalize_camera_model(model_str):
    """Normalize various aliases to 'equidistant' or 'pinhole'."""
    m = str(model_str or "pinhole").lower()
    if m in {"fisheye", "equidistantcamera", "equidistant"}:
        return "equidistant"
    return "pinhole"


def project_world_to_image(points_world, R_cw, t_cw, K, D, model,
                           img_w=None, img_h=None):
    """Project ENU points to image. Returns (uv, valid_mask, cam_depth).
    If img_w/img_h are given, `valid` includes in-bounds + positive depth;
    otherwise just positive depth + finite uv."""
    cam = (R_cw @ np.asarray(points_world).T).T + t_cw
    front = cam[:, 2] > 0.05
    pix = np.full((len(cam), 2), np.nan, dtype=np.float64)
    if front.any():
        pts = cam[front].reshape(-1, 1, 3)
        zero = np.zeros((3, 1), dtype=np.float64)
        if model == "equidistant":
            out, _ = cv2.fisheye.projectPoints(pts, zero, zero, K, D)
        else:
            out, _ = cv2.projectPoints(pts, zero, zero, K, D)
        pix[front] = out.reshape(-1, 2)
    valid = front & np.isfinite(pix[:, 0])
    if img_w is not None and img_h is not None:
        valid &= (pix[:, 0] >= 0) & (pix[:, 0] < img_w)
        valid &= (pix[:, 1] >= 0) & (pix[:, 1] < img_h)
    return pix, valid, cam[:, 2]


# ---------- BEV canvas ----------

def world_to_canvas(pts_xy, x_min, y_min, res, h_px):
    """ENU (x, y) -> canvas pixel (col, row). y+ is up in world, row decreases."""
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    col = ((pts[:, 0] - x_min) / res).astype(np.int32)
    row = (h_px - 1 - (pts[:, 1] - y_min) / res).astype(np.int32)
    return np.column_stack([col, row])
