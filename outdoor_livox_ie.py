#!/usr/bin/python3
import argparse
import bisect
import json
import math
import os
import pickle
import re
import signal
from collections import deque
from dataclasses import dataclass

import cv2
import genpy
import numpy as np
import rospy
from nav_msgs.msg import Path
from rosbag import Bag
from scipy.spatial.transform import Rotation as ScipyRotation
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_from_matrix

import predict
from util import bri
from util import get_colors
from util import get_i_pcd_msg
from util import get_rgba_pcd_msg
from util import save_nppc

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


STOP_REQUESTED = False
STOP_REASON = None
STOP_AFTER_MAX_FRAMES = False


def request_stop(reason):
    global STOP_REQUESTED
    global STOP_REASON
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        STOP_REASON = str(reason)
        print(f"stop requested: {STOP_REASON}")
        if rospy.core.is_initialized() and not rospy.is_shutdown():
            rospy.signal_shutdown(STOP_REASON)


def handle_termination_signal(signum, _frame):
    signal_name = signal.Signals(signum).name
    request_stop(signal_name)


def request_max_frames_stop(max_frames):
    global STOP_AFTER_MAX_FRAMES
    if not STOP_AFTER_MAX_FRAMES:
        STOP_AFTER_MAX_FRAMES = True
        request_stop(f"reach max frames {max_frames}")


def cmkdir(path):
    path = path.split("/")
    cur = ""
    for p in path:
        try:
            cur = "/".join([cur, p])
            os.mkdir(cur[1:])
        except Exception:
            pass


def dms_to_deg(deg_token: str, minute_token: str, second_token: str) -> float:
    deg = float(deg_token)
    minute = float(minute_token)
    second = float(second_token)
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


@dataclass
class IePose:
    timestamp: float
    latitude_deg: float
    longitude_deg: float
    height_m: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float


def load_ie_poses(ie_path):
    if not os.path.exists(ie_path):
        raise FileNotFoundError(f"IE txt not found: {ie_path}")
    poses = []
    with open(ie_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or not re.match(r"^[0-9]", line):
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            timestamp = float(parts[0])
            lat_deg = dms_to_deg(parts[3], parts[4], parts[5])
            lon_deg = dms_to_deg(parts[6], parts[7], parts[8])
            height_m = float(parts[9])
            roll_deg = float(parts[-4])
            pitch_deg = float(parts[-3])
            heading_deg = float(parts[-2])
        except (ValueError, TypeError):
            continue
        poses.append(
            IePose(
                timestamp=timestamp,
                latitude_deg=lat_deg,
                longitude_deg=lon_deg,
                height_m=height_m,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                heading_deg=heading_deg,
            )
        )
    poses.sort(key=lambda x: x.timestamp)
    return poses


class IePoseProvider:
    def __init__(self, poses):
        self.poses = poses
        # Pre-extract numpy arrays once so per-point batch interpolation
        # avoids re-walking the Python list on every lidar packet.
        self.timestamps = np.asarray([p.timestamp for p in poses], dtype=np.float64)
        self._lat = np.asarray([p.latitude_deg for p in poses], dtype=np.float64)
        self._lon = np.asarray([p.longitude_deg for p in poses], dtype=np.float64)
        self._h = np.asarray([p.height_m for p in poses], dtype=np.float64)
        self._roll = np.asarray([p.roll_deg for p in poses], dtype=np.float64)
        self._pitch = np.asarray([p.pitch_deg for p in poses], dtype=np.float64)
        self._heading = np.asarray([p.heading_deg for p in poses], dtype=np.float64)
        # Plain Python list version of timestamps for the scalar bisect path.
        self._timestamps_list = self.timestamps.tolist()

    def get_interpolated(self, timestamp):
        if not self.poses:
            return None
        right = bisect.bisect_right(self._timestamps_list, float(timestamp))
        left = max(0, min(len(self.poses) - 1, right - 1))
        lp = self.poses[left]
        if left + 1 >= len(self.poses):
            return lp
        rp = self.poses[left + 1]
        dt = float(rp.timestamp - lp.timestamp)
        if dt <= 1e-9:
            return lp
        alpha = float(np.clip((timestamp - lp.timestamp) / dt, 0.0, 1.0))
        return IePose(
            timestamp=float(timestamp),
            latitude_deg=float(lp.latitude_deg + (rp.latitude_deg - lp.latitude_deg) * alpha),
            longitude_deg=float(lp.longitude_deg + (rp.longitude_deg - lp.longitude_deg) * alpha),
            height_m=float(lp.height_m + (rp.height_m - lp.height_m) * alpha),
            roll_deg=lerp_angle_deg(lp.roll_deg, rp.roll_deg, alpha),
            pitch_deg=lerp_angle_deg(lp.pitch_deg, rp.pitch_deg, alpha),
            heading_deg=lerp_angle_deg(lp.heading_deg, rp.heading_deg, alpha),
        )

    def get_interpolated_batch(self, timestamps):
        """Vectorized version of get_interpolated for an (n,) array of times.

        Returns a dict of (n,) numpy arrays — same fields as IePose. Used by
        `undistort_livox_points_to_base` to avoid a per-point Python loop
        over `get_interpolated`, which was a primary cause of system stalls.
        """
        n_ie = len(self.poses)
        if n_ie == 0:
            return None
        ts = np.asarray(timestamps, dtype=np.float64)
        right = np.searchsorted(self.timestamps, ts, side="right")
        left = np.clip(right - 1, 0, n_ie - 1)
        right = np.clip(right, 0, n_ie - 1)
        t_left = self.timestamps[left]
        t_right = self.timestamps[right]
        dt = t_right - t_left
        alpha = np.where(dt > 1e-9, (ts - t_left) / dt, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        def _lerp(a, b):
            return a + alpha * (b - a)

        def _lerp_ang(a, b):
            d = (b - a + 180.0) % 360.0 - 180.0
            return a + d * alpha

        return {
            "timestamp": ts,
            "latitude_deg": _lerp(self._lat[left], self._lat[right]),
            "longitude_deg": _lerp(self._lon[left], self._lon[right]),
            "height_m": _lerp(self._h[left], self._h[right]),
            "roll_deg": _lerp_ang(self._roll[left], self._roll[right]),
            "pitch_deg": _lerp_ang(self._pitch[left], self._pitch[right]),
            "heading_deg": _lerp_ang(self._heading[left], self._heading[right]),
        }


def lerp_angle_deg(a_deg, b_deg, alpha):
    a = float(a_deg)
    b = float(b_deg)
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * float(alpha)


def _ecef_to_geodetic(ecef):
    """Zhu 1994 closed-form ECEF → (lat_deg, lon_deg, height_m)."""
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    a, b = 6378137.0, 6356752.3142
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
    h = (p / cos_lat - N) if abs(cos_lat) > 1e-6 else (abs(z) / abs(sin_lat) - N * (1.0 - e2))
    return math.degrees(lat), math.degrees(lon), h


def _euler_zyx_to_matrix(roll_deg, pitch_deg, yaw_deg):
    """ZYX active rotation matrix R_world_from_body from FAST-LIO2 Euler angles."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _r_enu_from_body_to_ie_angles(R):
    """Extract IE (roll, pitch, heading) from R_enu_from_body (3×3).

    Inverts rotation_matrix_from_ie using the identity:
        R_enu_from_body = R_enu_from_rfu @ R_rfu_from_flu
        R_rfu_from_enu  = Ry(-roll) @ Rx(-pitch) @ Rz(heading)

    scipy intrinsic 'zxy' as_euler([a,b,c]) means R = Ry(c) @ Rx(b) @ Rz(a),
    so the three angles map to [heading, -pitch, -roll].
    """
    from scipy.spatial.transform import Rotation as _Rot
    R_flu_from_rfu = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R_enu_from_rfu = R @ R_flu_from_rfu
    R_rfu_from_enu = R_enu_from_rfu.T
    angles = _Rot.from_matrix(R_rfu_from_enu).as_euler("zxy", degrees=True)
    heading = angles[0]
    pitch = -angles[1]
    roll = -angles[2]
    return float(roll), float(pitch), float(heading)


class ImuBuffer:
    """Time-sorted IMU gyro samples with vectorized rotation integration.

    Built once by reading the bag's IMU topic; consulted per-LiDAR-packet to
    compute per-point intra-scan rotation correction. Used by
    `undistort_livox_points_imu()`.

    Math: between consecutive 200Hz samples, vehicle yaw rate ≤ 30°/s so each
    inter-sample rotvec step is ≤ 0.15°. Vector-sum of small rotvecs is exact
    to ~1e-6 rad (Magnus / commutativity error negligible at small angles),
    so cumulative rotvec acts like a "rotation integral" we can index with
    linear interpolation.
    """

    def __init__(self):
        self._t = []
        self._gyro = []

    def add(self, timestamp_sec, gyro_xyz):
        self._t.append(float(timestamp_sec))
        self._gyro.append(np.asarray(gyro_xyz, dtype=np.float64))

    def finalize(self):
        self.t = np.asarray(self._t, dtype=np.float64)
        gyro = np.vstack(self._gyro) if self._gyro else np.zeros((0, 3))
        if len(self.t) >= 2:
            dt = np.diff(self.t)
            mean_g = 0.5 * (gyro[:-1] + gyro[1:])
            step = mean_g * dt[:, None]
            self._rotvec_cum = np.vstack([np.zeros(3), np.cumsum(step, axis=0)])
        else:
            self._rotvec_cum = np.zeros((len(self.t), 3))
        self._gyro = None  # release

    def _rotvec_at(self, t_targets):
        """Linear-interp cumulative rotvec at each t_target (numpy array)."""
        n = len(self.t)
        if n < 2:
            return np.zeros((len(t_targets), 3), dtype=np.float64)
        idx_right = np.searchsorted(self.t, t_targets, side="left")
        idx_right = np.clip(idx_right, 1, n - 1)
        idx_left = idx_right - 1
        t_l = self.t[idx_left]
        t_r = self.t[idx_right]
        alpha = np.clip((t_targets - t_l) / np.maximum(t_r - t_l, 1e-9), 0.0, 1.0)
        return self._rotvec_cum[idx_left] + alpha[:, None] * (
            self._rotvec_cum[idx_right] - self._rotvec_cum[idx_left])

    def rotation_between(self, t_from, t_targets):
        """Return (N, 3, 3) rotation matrices R(t_from → t_target).

        R = exp(rotvec_cum(t_target) - rotvec_cum(t_from)) under small-angle
        commutativity (valid at 200 Hz IMU rates).
        """
        if not hasattr(self, "t") or len(self.t) < 2:
            return np.tile(np.eye(3), (len(t_targets), 1, 1))
        rv_from = self._rotvec_at(np.array([float(t_from)]))[0]
        rv_to = self._rotvec_at(np.asarray(t_targets, dtype=np.float64))
        rv_delta = rv_to - rv_from
        return ScipyRotation.from_rotvec(rv_delta).as_matrix()


def deg2rad(v):
    return float(v) * math.pi / 180.0


def rotation_matrix_from_ie(roll_deg, pitch_deg, heading_deg):
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    neg_azimuth = -deg2rad(heading_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(neg_azimuth), math.sin(neg_azimuth)
    ry_passive_R = np.array([[cr, 0.0, -sr], [0.0, 1.0, 0.0], [sr, 0.0, cr]], dtype=np.float64)
    rx_passive_P = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]], dtype=np.float64)
    rz_passive_negA = np.array([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_rfu_from_enu = ry_passive_R @ rx_passive_P @ rz_passive_negA
    r_enu_from_rfu = r_rfu_from_enu.T
    r_rfu_from_flu = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return r_enu_from_rfu @ r_rfu_from_flu


def rotation_matrices_from_ie_batch(roll_deg, pitch_deg, heading_deg):
    """Vectorized form of rotation_matrix_from_ie. Inputs are (n,) arrays;
    returns an (n, 3, 3) array of R_enu_from_body matrices.
    """
    roll = np.deg2rad(np.asarray(roll_deg, dtype=np.float64))
    pitch = np.deg2rad(np.asarray(pitch_deg, dtype=np.float64))
    neg_az = -np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    n = roll.shape[0]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(neg_az), np.sin(neg_az)

    Ry = np.zeros((n, 3, 3), dtype=np.float64)
    Ry[:, 0, 0] = cr; Ry[:, 0, 2] = -sr
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = sr; Ry[:, 2, 2] = cr

    Rx = np.zeros((n, 3, 3), dtype=np.float64)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cp; Rx[:, 1, 2] = sp
    Rx[:, 2, 1] = -sp; Rx[:, 2, 2] = cp

    Rz = np.zeros((n, 3, 3), dtype=np.float64)
    Rz[:, 0, 0] = cz; Rz[:, 0, 1] = sz
    Rz[:, 1, 0] = -sz; Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1.0

    r_rfu_from_enu = Ry @ Rx @ Rz
    r_enu_from_rfu = r_rfu_from_enu.transpose(0, 2, 1)
    r_rfu_from_flu = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return r_enu_from_rfu @ r_rfu_from_flu


def pose_enu_from_geodetic_batch(lat_deg, lon_deg, h_m, ecef_origin, enu_from_ecef):
    """Vectorized geodetic_to_ecef + ENU rotation. Inputs are (n,) arrays;
    returns (n, 3) ENU positions.
    """
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(h_m, dtype=np.float64)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    sin_lon = np.sin(lon); cos_lon = np.cos(lon)
    n_curve = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n_curve + h) * cos_lat * cos_lon
    y = (n_curve + h) * cos_lat * sin_lon
    z = (n_curve * (1.0 - e2) + h) * sin_lat
    ecef = np.stack([x, y, z], axis=-1)
    # ENU = enu_from_ecef @ (ecef - origin); for row-vectors that's
    # (ecef - origin) @ enu_from_ecef.T
    return (ecef - ecef_origin[None, :]) @ enu_from_ecef.T


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat_deg, lon_deg):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [[-sin_lon, cos_lon, 0.0], [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]],
        dtype=np.float64,
    )


def pose_enu_from_ie(ie_pose, ecef_origin, enu_from_ecef):
    ecef = geodetic_to_ecef(ie_pose.latitude_deg, ie_pose.longitude_deg, ie_pose.height_m)
    return enu_from_ecef @ (ecef - ecef_origin)


def ie_pose_to_quaternion_xyzw(ie_pose):
    r = rotation_matrix_from_ie(ie_pose.roll_deg, ie_pose.pitch_deg, ie_pose.heading_deg)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    q = quaternion_from_matrix(T)
    return np.asarray(q, dtype=np.float64)


# base<-lidar and base<-span, copied from /mnt/ning_602/work/tracking
# (configs/lidar_config.yaml, ie_lidar_extrinsics block). base/body is FLU.
# Tracking calibrates span as RFU (per the "FLU->RFU" comment in the YAML
# and the SPAN convention used by NovAtel IE).
R_BASE_FROM_LIDAR = np.array([[0.9063, 0.0, 0.4226], [0.0, 1.0, 0.0], [-0.4226, 0.0, 0.9063]], dtype=np.float64)
T_BASE_FROM_LIDAR_M = np.array([0.0315, 0.0, 0.1314], dtype=np.float64)
R_BASE_FROM_SPAN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
T_BASE_FROM_SPAN_M = np.array([-0.1854, 0.0, -0.242], dtype=np.float64)
R_LIDAR_FROM_BASE = R_BASE_FROM_LIDAR.T

# Lidar position in the IE body frame (FLU axes, **origin at SPAN antenna**)
# = base->lidar position minus base->span position. This matches both the
# tracking project's chain `R_SPAN_FROM_LIDAR @ p + T_SPAN_FROM_LIDAR_M`
# followed by FLU<-RFU axis swap, and the convention used by NovAtel IE
# poses (the lat/lon/h reported by IE is the antenna position, so the
# body origin must also be the antenna for `p_world = R @ p_body + t_enu`
# to be self-consistent — see doc/coordinate_transforms.md).
T_FLU_LIDAR_FROM_SPAN = T_BASE_FROM_LIDAR_M - T_BASE_FROM_SPAN_M


def lidar_to_ie_body(points_lidar_xyz):
    """LiDAR-frame xyz -> IE body-frame (FLU, origin at SPAN antenna).

    Previously this hard-coded a `(x, -y, -z)` flip that assumed span was
    FRD. After the 2026-04-29 rotation fix the rotation was right but the
    body origin was at `base`, not `span`, leaving a constant ~30 cm
    offset in every one-way `lidar -> world ENU` (since `t_enu` derived
    from IE describes the SPAN antenna position). Tracking's body frame
    is span-centered; we now match it via the explicit
    `T_FLU_LIDAR_FROM_SPAN` translation.
    """
    pts = np.asarray(points_lidar_xyz, dtype=np.float64)
    return (R_BASE_FROM_LIDAR @ pts.T).T + T_FLU_LIDAR_FROM_SPAN[None, :]


def ie_body_to_lidar(points_body_flu):
    pts = np.asarray(points_body_flu, dtype=np.float64)
    return (R_LIDAR_FROM_BASE @ (pts - T_FLU_LIDAR_FROM_SPAN[None, :]).T).T


class RollingGroundPlane:
    """Sliding-window buffer of near-ground LiDAR points across recent frames.

    Each frame's near-ground points (camera_Z <= fit_max_depth, ground-class)
    are accumulated together with the per-frame (R_enu_from_lidar, t_enu_lidar)
    so the buffer can transform them into any later frame's LiDAR coordinates
    for a unified RANSAC plane fit. Per-frame fits on ~30-100 sidewalk points
    are too noisy and break on frames with no near coverage; pooling ~3000
    points across 30 frames gives a robust, stable plane.
    """

    def __init__(self, window=30, ransac_iter=200, inlier_thresh=0.05,
                 min_fit_points=50, rng_seed=0):
        from collections import deque as _deque
        self.window = int(window)
        self.ransac_iter = int(ransac_iter)
        self.inlier_thresh = float(inlier_thresh)
        self.min_fit_points = int(min_fit_points)
        self.history = _deque()
        self._rng = np.random.default_rng(rng_seed)

    def add(self, R_enu_from_lidar, t_enu_lidar, near_pts_lidar):
        if near_pts_lidar is None or len(near_pts_lidar) == 0:
            return
        self.history.append((
            np.asarray(R_enu_from_lidar, dtype=np.float64).copy(),
            np.asarray(t_enu_lidar, dtype=np.float64).copy(),
            np.asarray(near_pts_lidar, dtype=np.float64).copy(),
        ))
        while len(self.history) > self.window:
            self.history.popleft()

    def fit_in_target_lidar(self, R_enu_from_lidar_tgt, t_enu_lidar_tgt):
        if not self.history:
            return None
        R_l_from_enu = np.asarray(R_enu_from_lidar_tgt, dtype=np.float64).T
        t_tgt = np.asarray(t_enu_lidar_tgt, dtype=np.float64)
        chunks = []
        for R_old, t_old, pts_old in self.history:
            enu = (R_old @ pts_old.T).T + t_old
            pts_tgt = (R_l_from_enu @ (enu - t_tgt).T).T
            chunks.append(pts_tgt)
        pts = np.vstack(chunks)
        n = len(pts)
        if n < self.min_fit_points:
            return None
        best_inlier_count = 0
        best_inliers = None
        for _ in range(self.ransac_iter):
            idx = self._rng.choice(n, size=3, replace=False)
            sample = pts[idx]
            A_s = np.column_stack([sample[:, 0], sample[:, 1], np.ones(3)])
            try:
                coef = np.linalg.solve(A_s, sample[:, 2])
            except np.linalg.LinAlgError:
                continue
            a, b, c = coef
            pred = a * pts[:, 0] + b * pts[:, 1] + c
            inliers = np.abs(pts[:, 2] - pred) < self.inlier_thresh
            cnt = int(inliers.sum())
            if cnt > best_inlier_count:
                best_inlier_count = cnt
                best_inliers = inliers
        if best_inliers is None or best_inlier_count < self.min_fit_points // 2:
            return None
        in_pts = pts[best_inliers]
        A = np.column_stack([in_pts[:, 0], in_pts[:, 1], np.ones(len(in_pts))])
        coef, *_ = np.linalg.lstsq(A, in_pts[:, 2], rcond=None)
        return float(coef[0]), float(coef[1]), float(coef[2])


def livox_msg_to_points_with_meta(msg):
    if getattr(msg, "_type", "") != "livox_ros_driver2/CustomMsg":
        raise RuntimeError(f"Unsupported lidar msg type: {getattr(msg, '_type', '')}, expected livox_ros_driver2/CustomMsg")
    pts_list = msg.points
    n = len(pts_list)
    if n == 0:
        points = np.zeros((0, 4), dtype=np.float64)
        offsets_s = np.zeros((0,), dtype=np.float64)
    else:
        # A single list comprehension + np.fromiter-style array build is
        # ~2-3× faster than indexed assignment in a Python for-loop and
        # produces fewer temporary objects, which keeps the GC quiet.
        arr = np.array(
            [(p.x, p.y, p.z, p.reflectivity, p.offset_time) for p in pts_list],
            dtype=np.float64,
        )
        points = arr[:, :4]
        offsets_s = arr[:, 4] * 1e-9
    base_ts = float(getattr(msg, "timebase", 0)) * 1e-9
    if base_ts <= 0.0:
        base_ts = float(msg.header.stamp.to_sec())
    return {
        "base_timestamp": base_ts,
        "points_xyzi": points,
        "offsets_s": offsets_s,
    }


def reduce_lidar_packet(packet, stride=1, max_points=0):
    stride = max(int(stride), 1)
    max_points = max(int(max_points), 0)
    points = packet["points_xyzi"]
    offsets = packet["offsets_s"]
    if stride > 1:
        points = points[::stride]
        offsets = offsets[::stride]
    if max_points > 0 and len(points) > max_points:
        points = points[:max_points]
        offsets = offsets[:max_points]
    return {
        "base_timestamp": packet["base_timestamp"],
        "points_xyzi": points,
        "offsets_s": offsets,
    }


def decode_image_msg(msg):
    msg_type = getattr(msg, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode CompressedImage")
        return frame
    if msg_type == "sensor_msgs/Image":
        h = int(msg.height)
        w = int(msg.width)
        enc = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if enc == "mono8":
            gray = data.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if enc in {"bgr8", "rgb8"}:
            frame = data.reshape(h, w, 3)
            if enc == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame.copy()
    raise RuntimeError(f"Unsupported image msg type: {msg_type}")


def message_stamp_sec(msg, fallback_stamp):
    """Prefer sensor header time over rosbag record time.

    The bag record stamp includes transport/recording latency. In this bag,
    camera record stamps are about 24-29 ms after image header stamps, while
    Livox timebase matches its header stamp. Mixing camera bag time with
    lidar sensor time shifts the IE pose used for pkl generation.
    """
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        try:
            ts = float(stamp.to_sec())
            if ts > 0.0:
                return ts
        except Exception:
            pass
    return float(fallback_stamp.to_sec())


def project_lidar_to_class(
    points_xyz_lidar,
    class_image,
    camera_matrix,
    dist_coeffs,
    rotation_lidar_from_camera,
    translation_lidar_from_camera,
    camera_model="pinhole",
    min_depth=0.2,
    class_max_depth=None,
    class_plane_filter=None,
    class_plane_coefs=None,
    zbuffer_px=0,
):
    """Project LiDAR points (in lidar frame) onto a class-id image and return
    the per-point class assignment.

    Mirrors the convention used by /mnt/ning_602/work/tracking
    (`core/fusion_tracking.py:_project_lidar_points_to_image`):
    target_frame stores `lidar_from_camera` so camera-frame points are
    `R^T @ (p_lidar - t)`, then `cv2.fisheye.projectPoints` (equidistant) or
    `cv2.projectPoints` (pinhole) projects them onto the raw image.

    `class_max_depth` is an optional `{class_id: max_camera_z_meters}` map.
    Points labeled with a class in the map whose camera-frame Z exceeds the
    cap are relabeled to SEG_BACKGROUND_CLASS — far-field projection error
    grows fast (a 2 mrad calibration residual is ~5 cm at 25 m), so we kill
    the long-tail false positives at distance for the classes that suffer
    most from boundary jitter.

    `class_plane_filter` is an optional
    `{class_id: {"fit_max_depth": float, "tolerance": float}}` map. For each
    class, fit z = a*x + b*y + c (LiDAR frame) on points with
    `camera_Z <= fit_max_depth` and drop the remaining (far) points whose
    LiDAR z deviates from the fitted plane by more than `tolerance` metres.
    Runs after `class_max_depth` so the plane filter only cleans the
    surviving 3m < camera_Z <= class_max_depth window.

    `zbuffer_px` keeps only the nearest LiDAR point per projected pixel block
    before class lookup. Without this visibility check, a farther point can be
    labeled from a foreground semantic pixel and later reproject onto walls,
    people, or railings in neighboring frames.

    Returns (M, 4) float64: (x_lidar, y_lidar, z_lidar, class_id) for points
    that landed inside the image.
    """
    points = np.asarray(points_xyz_lidar, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] >= 3:
        points = points[:, :3]
    else:
        points = points.reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((0, 4), dtype=np.float64)

    R = np.asarray(rotation_lidar_from_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation_lidar_from_camera, dtype=np.float64).reshape(3)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    points_camera = (R.T @ (points - t).T).T
    front_mask = points_camera[:, 2] > float(min_depth)
    if not front_mask.any():
        return np.zeros((0, 4), dtype=np.float64)

    front_idx = np.where(front_mask)[0]
    front_pts = points_camera[front_mask].reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    if str(camera_model).lower() == "equidistant":
        img_pts, _ = cv2.fisheye.projectPoints(
            objectPoints=front_pts, rvec=rvec, tvec=tvec, K=K, D=D,
        )
    else:
        img_pts, _ = cv2.projectPoints(
            objectPoints=front_pts, rvec=rvec, tvec=tvec,
            cameraMatrix=K, distCoeffs=D,
        )
    pixels = img_pts.reshape(-1, 2)

    cols = np.round(pixels[:, 0]).astype(np.int64)
    rows = np.round(pixels[:, 1]).astype(np.int64)
    h, w = class_image.shape[:2]
    in_bounds = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not in_bounds.any():
        return np.zeros((0, 4), dtype=np.float64)

    keep_global = front_idx[in_bounds]
    keep_cols = cols[in_bounds]
    keep_rows = rows[in_bounds]
    if int(zbuffer_px) > 0 and len(keep_global) > 0:
        block = max(1, int(zbuffer_px))
        block_cols = keep_cols // block
        block_rows = keep_rows // block
        block_ids = block_rows * ((w + block - 1) // block) + block_cols
        order = np.lexsort((points_camera[keep_global, 2], block_ids))
        sorted_blocks = block_ids[order]
        first = np.r_[True, sorted_blocks[1:] != sorted_blocks[:-1]]
        keep_local = order[first]
        keep_global = keep_global[keep_local]
        keep_cols = keep_cols[keep_local]
        keep_rows = keep_rows[keep_local]
    classes = class_image[keep_rows, keep_cols].astype(np.float64)

    if class_max_depth:
        kept_camera_z = points_camera[keep_global, 2]
        for cls_id, max_z in class_max_depth.items():
            far_mask = (classes == float(cls_id)) & (kept_camera_z > float(max_z))
            if far_mask.any():
                classes[far_mask] = float(SEG_BACKGROUND_CLASS)

    if class_plane_filter:
        kept_lidar = points[keep_global]
        kept_camera_z = points_camera[keep_global, 2]
        for cls_id, params in class_plane_filter.items():
            cls_val = float(cls_id)
            fit_depth = float(params.get("fit_max_depth", 3.0))
            tol = float(params.get("tolerance", 0.15))
            min_pts = int(params.get("min_fit_points", 10))
            cls_mask = (classes == cls_val)
            far_mask = cls_mask & (kept_camera_z > fit_depth)
            if not far_mask.any():
                continue
            pre_coef = (class_plane_coefs or {}).get(cls_id)
            if pre_coef is None:
                near_mask = cls_mask & (kept_camera_z <= fit_depth)
                if int(near_mask.sum()) < min_pts:
                    classes[far_mask] = float(SEG_BACKGROUND_CLASS)
                    continue
                near_pts = kept_lidar[near_mask]
                A = np.column_stack([near_pts[:, 0], near_pts[:, 1],
                                     np.ones(len(near_pts), dtype=np.float64)])
                try:
                    coef, *_ = np.linalg.lstsq(A, near_pts[:, 2], rcond=None)
                except np.linalg.LinAlgError:
                    classes[far_mask] = float(SEG_BACKGROUND_CLASS)
                    continue
                a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
            else:
                a, b, c = float(pre_coef[0]), float(pre_coef[1]), float(pre_coef[2])
            far_pts = kept_lidar[far_mask]
            pred_z = a * far_pts[:, 0] + b * far_pts[:, 1] + c
            bad = np.abs(far_pts[:, 2] - pred_z) > tol
            if bad.any():
                far_idx = np.where(far_mask)[0]
                classes[far_idx[bad]] = float(SEG_BACKGROUND_CLASS)

    out = np.empty((len(keep_global), 4), dtype=np.float64)
    out[:, :3] = points[keep_global]
    out[:, 3] = classes
    return out


SEG_BACKGROUND_CLASS = 0  # sentinel for "filtered-out small region";
# class 0 ("bird" in mapillary) never appears in road scenes and is filtered
# out downstream by any class-id whitelist (sidewalk=15, road=13, etc.).


def filter_small_segments(class_image, min_pixels, classes_to_filter=None):
    """Drop per-class connected components smaller than `min_pixels`.

    Parameters
    ----------
    class_image : (H, W) uint8 array of class ids.
    min_pixels  : int. Components with fewer pixels are replaced with
                  `SEG_BACKGROUND_CLASS` (will not match any real class id
                  during projection lookup).
    classes_to_filter : iterable of int, or None. If None, all class ids
                  present in the image are filtered. If a list, only those
                  ids are filtered (others left untouched).
    """
    if min_pixels <= 0:
        return class_image
    out = class_image.copy()
    if classes_to_filter is None:
        classes_to_filter = [int(c) for c in np.unique(class_image)
                             if c != SEG_BACKGROUND_CLASS]
    for cid in classes_to_filter:
        mask = (out == cid).astype(np.uint8)
        if not mask.any():
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, n_labels):  # 0 is background
            if stats[lbl, cv2.CC_STAT_AREA] < min_pixels:
                out[labels == lbl] = SEG_BACKGROUND_CLASS
    return out


def get_semantic_pcd(
    img,
    pcd_xyz_lidar,
    camera_matrix,
    dist_coeffs,
    rotation_lidar_from_camera,
    translation_lidar_from_camera,
    camera_model,
    predictor,
    predict_image_scale=1.0,
    min_segment_pixels=0,
    seg_filter_classes=None,
    class_erode_px=None,
    class_max_depth=None,
    class_plane_filter=None,
    class_plane_coefs=None,
    zbuffer_px=0,
    suppress_near_person_px=None,
    person_class=19,
):
    """Run segmentation on the raw camera image and assign each in-FOV LiDAR
    point the class id at its projected pixel.

    The image is *not* undistorted before the segmenter runs; tracking's
    pipeline operates on the raw fisheye image and we follow the same
    convention so that `cv2.fisheye.projectPoints` lands at the same pixels
    the segmenter actually saw.

    `min_segment_pixels` (default 0 = off) discards per-class connected
    components smaller than this many pixels — kills tiny mis-segmented
    blobs before they get projected onto lidar points.
    `seg_filter_classes` restricts the filter to specific class ids
    (e.g. `[15]` to only clean sidewalk), or None to filter all classes.
    """
    predict_image_scale = float(predict_image_scale)
    if predict_image_scale > 0.0 and abs(predict_image_scale - 1.0) > 1e-6:
        small_w = max(1, int(round(img.shape[1] * predict_image_scale)))
        small_h = max(1, int(round(img.shape[0] * predict_image_scale)))
        pred_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        cimg_small = predictor(pred_img)
        cimg = cv2.resize(cimg_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        cimg = predictor(img)
    if min_segment_pixels > 0:
        cimg = filter_small_segments(cimg, int(min_segment_pixels), seg_filter_classes)
    if class_erode_px:
        # Inset selected class boundaries by N pixels before projection.
        # Mask2Former boundaries have a 1-3 px uncertainty band that
        # accumulates into a "false-curb strip" in the world cloud after
        # 700+ frames. Eroding that band before sampling kills the
        # boundary-jitter source. We overwrite to SEG_BACKGROUND_CLASS so
        # downstream class-id whitelists ignore the eroded pixels.
        for cls_id, erode_px in class_erode_px.items():
            erode_px = int(erode_px)
            if erode_px <= 0:
                continue
            mask = (cimg == int(cls_id)).astype(np.uint8)
            if not mask.any():
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1))
            eroded = cv2.erode(mask, kernel)
            stripped = mask.astype(bool) & ~eroded.astype(bool)
            if stripped.any():
                cimg[stripped] = SEG_BACKGROUND_CLASS
    if suppress_near_person_px:
        # Mask2Former on Mapillary has a strong "person → sidewalk-under-feet"
        # prior baked into the training distribution. Every visible pedestrian
        # gets a ~25 px halo of sidewalk pixels around them regardless of what
        # they're actually standing on. Accumulated across 700+ frames this
        # paints sidewalk-class points on roads (under standing pedestrians)
        # and across crosswalks (under walking pedestrians, forming the
        # spurious sidewalk "bridge" that connects the two real sidewalks).
        # Killing the halo at the source removes both artifacts cleanly.
        person_mask = (cimg == int(person_class)).astype(np.uint8)
        if person_mask.any():
            for cls_id, radius in suppress_near_person_px.items():
                radius = int(radius)
                if radius <= 0:
                    continue
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * radius + 1, 2 * radius + 1))
                person_dil = cv2.dilate(person_mask, kernel).astype(bool)
                suppress = (cimg == int(cls_id)) & person_dil
                if suppress.any():
                    cimg[suppress] = SEG_BACKGROUND_CLASS
    sem_pcdata = project_lidar_to_class(
        pcd_xyz_lidar,
        cimg,
        camera_matrix,
        dist_coeffs,
        rotation_lidar_from_camera,
        translation_lidar_from_camera,
        camera_model=camera_model,
        class_max_depth=class_max_depth,
        class_plane_filter=class_plane_filter,
        class_plane_coefs=class_plane_coefs,
        zbuffer_px=zbuffer_px,
    )
    if len(sem_pcdata) == 0:
        return np.zeros((0, 4), dtype=np.float64), cimg
    return sem_pcdata, cimg


def configure_cpu_runtime(cpu_threads):
    cpu_threads = max(int(cpu_threads), 1)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    try:
        cv2.setNumThreads(cpu_threads)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(cpu_threads, 2)))
    except Exception:
        pass


def _ie_pose_to_enu_transform(ie_pose, ecef_origin, enu_from_ecef):
    r_enu_from_body = rotation_matrix_from_ie(ie_pose.roll_deg, ie_pose.pitch_deg, ie_pose.heading_deg)
    t_enu = pose_enu_from_ie(ie_pose, ecef_origin, enu_from_ecef)
    return r_enu_from_body, t_enu


def undistort_livox_points_to_base(points_xyzi, offsets_s, base_pose, ie_provider, ecef_origin, enu_from_ecef):
    """Motion-compensate a livox packet so every point is expressed at the
    packet's base timestamp.

    This used to be a Python loop calling `ie_provider.get_interpolated`
    once per point and building a fresh 3×3 rotation matrix per point;
    at 12k pts × 10 Hz that's 120k pose lookups + matmuls per second in
    Python and was the main reason the desktop felt frozen during
    capture. Everything below is now numpy-vectorized.
    """
    if len(points_xyzi) == 0:
        return points_xyzi
    out = np.array(points_xyzi, dtype=np.float64, copy=True)

    t_pts = float(base_pose.timestamp) + np.asarray(offsets_s, dtype=np.float64)
    interp = ie_provider.get_interpolated_batch(t_pts)
    if interp is None:
        return out

    R_per = rotation_matrices_from_ie_batch(
        interp["roll_deg"], interp["pitch_deg"], interp["heading_deg"]
    )  # (n, 3, 3)
    t_per = pose_enu_from_geodetic_batch(
        interp["latitude_deg"], interp["longitude_deg"], interp["height_m"],
        ecef_origin, enu_from_ecef,
    )  # (n, 3)

    base_r_enu_from_body, base_t_enu = _ie_pose_to_enu_transform(base_pose, ecef_origin, enu_from_ecef)

    pts_body = (R_BASE_FROM_LIDAR @ out[:, :3].T).T + T_FLU_LIDAR_FROM_SPAN[None, :]
    pts_enu = np.einsum("nij,nj->ni", R_per, pts_body) + t_per
    # base_R^T @ (pts_enu - base_t)  ==  (pts_enu - base_t) @ base_R   (row-vector form)
    pts_base_body = (pts_enu - base_t_enu[None, :]) @ base_r_enu_from_body
    # R_LIDAR_FROM_BASE @ x  ==  x @ R_BASE_FROM_LIDAR                 (row-vector form)
    pts_base_lidar = (pts_base_body - T_FLU_LIDAR_FROM_SPAN[None, :]) @ R_BASE_FROM_LIDAR

    out[:, :3] = pts_base_lidar
    return out


def undistort_livox_points_imu(points_xyzi, offsets_s, base_ts, imu_buffer):
    """Motion-compensate a Livox packet using raw IMU gyro integration.

    For each point at absolute time `base_ts + offset_s`, the body has rotated
    by R(t_base → t_off) since the packet's base timestamp. The point is
    measured in the body frame *at* t_off; we want it in the body frame at
    t_base. So:
        p_at_base = R(t_off → t_base) @ p_at_off
                  = R(t_base → t_off)^{-1} @ p_at_off
    Translation is ignored in v1 (worst-case 25 cm per-point at 5 m/s × 50 ms,
    comparable to LiDAR range noise).

    Assumes IMU and LiDAR share the same rotation frame — true for the Livox
    internal `/imu` (extrinsic R = identity in `livox_ie.yaml`).
    """
    if imu_buffer is None or not hasattr(imu_buffer, "t") or len(imu_buffer.t) < 2:
        return points_xyzi
    if len(points_xyzi) == 0:
        return points_xyzi
    out = np.array(points_xyzi, dtype=np.float64, copy=True)
    t_targets = float(base_ts) + np.asarray(offsets_s, dtype=np.float64)
    R_off_from_base = imu_buffer.rotation_between(float(base_ts), t_targets)  # (N, 3, 3)
    # einsum with R[..., j, i] picks the j-th row of the transpose → applies R^T.
    out[:, :3] = np.einsum("nji,nj->ni", R_off_from_base, out[:, :3])
    return out


def find_bracketing_lidar(lidar_queue, t_img):
    if len(lidar_queue) < 2:
        return None
    l_next = None
    l_last = None
    for item in reversed(lidar_queue):
        if float(item["timestamp"]) < float(t_img):
            l_last = item
            break
        l_next = item
    if l_last is None or l_next is None:
        return None
    return l_last, l_next


def choose_nearest_lidar(lidar_queue, t_img, max_dt):
    pair = find_bracketing_lidar(lidar_queue, t_img)
    if pair is None:
        return None
    l_last, l_next = pair
    dt_last = abs(float(t_img) - float(l_last["timestamp"]))
    dt_next = abs(float(l_next["timestamp"]) - float(t_img))
    chosen = l_next if dt_next < dt_last else l_last
    chosen_dt = min(dt_last, dt_next)
    if max_dt is not None and chosen_dt > float(max_dt):
        return None
    return chosen


def transform_lidar_points_between_timestamps(points_src_lidar, src_pose, dst_pose, ecef_origin, enu_from_ecef):
    src_r_enu_from_body = rotation_matrix_from_ie(src_pose.roll_deg, src_pose.pitch_deg, src_pose.heading_deg)
    dst_r_enu_from_body = rotation_matrix_from_ie(dst_pose.roll_deg, dst_pose.pitch_deg, dst_pose.heading_deg)
    src_t_enu = pose_enu_from_ie(src_pose, ecef_origin, enu_from_ecef)
    dst_t_enu = pose_enu_from_ie(dst_pose, ecef_origin, enu_from_ecef)
    pts_body_src = lidar_to_ie_body(points_src_lidar[:, :3])
    pts_enu = (src_r_enu_from_body @ pts_body_src.T).T + src_t_enu[None, :]
    pts_body_dst = (dst_r_enu_from_body.T @ (pts_enu - dst_t_enu[None, :]).T).T
    out = np.array(points_src_lidar, dtype=np.float64, copy=True)
    out[:, :3] = ie_body_to_lidar(pts_body_dst)
    return out


def transform_semantic_lidar_to_world(sem_pcd_lidar_xyzc, dst_pose, ecef_origin, enu_from_ecef):
    if len(sem_pcd_lidar_xyzc) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    dst_r_enu_from_body = rotation_matrix_from_ie(dst_pose.roll_deg, dst_pose.pitch_deg, dst_pose.heading_deg)
    dst_t_enu = pose_enu_from_ie(dst_pose, ecef_origin, enu_from_ecef)
    pts_body = lidar_to_ie_body(sem_pcd_lidar_xyzc[:, :3])
    pts_enu = (dst_r_enu_from_body @ pts_body.T).T + dst_t_enu[None, :]
    out = np.zeros_like(sem_pcd_lidar_xyzc, dtype=np.float64)
    out[:, :3] = pts_enu
    out[:, 3] = sem_pcd_lidar_xyzc[:, 3]
    return out


def main():
    cmkdir("result/outdoor/originpics")
    cmkdir("result/outdoor/sempics")
    parser = argparse.ArgumentParser(description="Outdoor semantic point cloud builder for livox bag + IE txt")
    parser.add_argument("-c", "--config", default="config/outdoor_config_livox_ie.json")
    parser.add_argument("-b", "--bag", default=None)
    parser.add_argument("--ie-txt", default=None, help="Path to IE txt such as 0421-AM-2026.txt")
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument("--lidar-topic", default=None)
    parser.add_argument("-f", "--fastfoward", default=None, type=float)
    parser.add_argument("-d", "--duration", default=None, type=float)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--max-sync-dt", default=None, type=float, help="Max abs(camera_ts - lidar_ts) in seconds")
    parser.add_argument("--lidar-queue-size", default=None, type=int)
    parser.add_argument("--cpu-threads", default=None, type=int)
    parser.add_argument("--lidar-stride", default=None, type=int, help="Keep every Nth lidar point before processing.")
    parser.add_argument("--max-lidar-points", default=None, type=int, help="Optional hard cap per lidar packet after stride.")
    parser.add_argument("--predict-image-scale", default=None, type=float, help="Scale factor for segmentation inference image.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    bag_path = args.bag if args.bag is not None else config["bag_file"]
    ie_txt = args.ie_txt if args.ie_txt is not None else config.get("ie_txt", "data/outdoor/0421-AM-2026.txt")
    camera_topic = args.camera_topic if args.camera_topic is not None else config.get("camera_topic", "/front_camera/image/compressed")
    lidar_topic = args.lidar_topic if args.lidar_topic is not None else config.get("LiDAR_topic", "/lidar")
    start_offset = args.fastfoward if args.fastfoward is not None else config.get("start_time", 0.0)
    duration = args.duration if args.duration is not None else config.get("play_time", -1)
    max_sync_dt = args.max_sync_dt if args.max_sync_dt is not None else float(config.get("max_sync_dt", 0.12))
    # Default lowered from 300 (30 s of raw lidar @ 10 Hz, ~120 MB) to 50
    # (~5 s, ~20 MB). Image-vs-lidar sync windows are well under a second,
    # so a 5 s buffer is more than enough; the old default was a memory
    # liability on long bags and a contributor to the system stalling.
    lidar_queue_size = args.lidar_queue_size if args.lidar_queue_size is not None else int(config.get("lidar_queue_size", 50))
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else int(config.get("cpu_threads", 2))
    lidar_stride = args.lidar_stride if args.lidar_stride is not None else int(config.get("lidar_stride", 2))
    max_lidar_points = args.max_lidar_points if args.max_lidar_points is not None else int(config.get("max_lidar_points", 0))
    predict_image_scale = (
        args.predict_image_scale if args.predict_image_scale is not None else float(config.get("predict_image_scale", 0.75))
    )
    min_segment_pixels = int(config.get("min_segment_pixels", 0))
    seg_filter_classes = config.get("seg_filter_classes", None)
    if seg_filter_classes is not None:
        seg_filter_classes = [int(c) for c in seg_filter_classes]
    # Per-class boundary erosion + far-field cutoff. Both keyed by class id
    # so we can dial sidewalk/road/etc. independently. Defaults are tuned
    # for sidewalk(15) on the 0421-AM bag (see: 2D seg looks clean but the
    # 3D world cloud accumulates curb-edge jitter and far-field projection
    # error into a false sidewalk-on-road strip).
    sidewalk_class = int(config.get("sidewalk_class", 15))
    class_erode_px = dict(config.get("class_erode_px", {sidewalk_class: 2}))
    class_erode_px = {int(k): int(v) for k, v in class_erode_px.items()}
    class_max_depth = dict(config.get("class_max_depth", {sidewalk_class: 25.0}))
    class_max_depth = {int(k): float(v) for k, v in class_max_depth.items()}
    class_plane_filter = dict(config.get("class_plane_filter", {}))
    class_plane_filter = {int(k): dict(v) for k, v in class_plane_filter.items()}
    plane_window = int(config.get("ground_plane_window", 30))
    plane_inlier_thresh = float(config.get("ground_plane_inlier_thresh", 0.05))
    plane_ransac_iter = int(config.get("ground_plane_ransac_iter", 200))
    plane_min_fit_points = int(config.get("ground_plane_min_fit_points", 50))
    rolling_planes = {
        cls_id: RollingGroundPlane(
            window=plane_window,
            ransac_iter=plane_ransac_iter,
            inlier_thresh=plane_inlier_thresh,
            min_fit_points=plane_min_fit_points,
        )
        for cls_id in class_plane_filter
    }
    suppress_near_person_px = dict(config.get(
        "suppress_near_person_px", {sidewalk_class: 25}))
    suppress_near_person_px = {int(k): int(v) for k, v in suppress_near_person_px.items()}
    person_class = int(config.get("person_class", 19))
    projection_zbuffer_px = int(config.get("projection_zbuffer_px", 3))
    road_class = config.get("road_class", 13)

    color_classes = get_colors(config["cmap"])
    camera_matrix = np.asarray(config["intrinsic"], dtype=np.float64)
    dist_coeffs = np.asarray(config["distortion_matrix"], dtype=np.float64)
    camera_model = str(config.get("camera_model", "pinhole")).lower()
    if camera_model in {"equidistantcamera", "fisheye"}:
        camera_model = "equidistant"
    if camera_model not in {"pinhole", "equidistant"}:
        raise ValueError(f"Unsupported camera_model in config: {config.get('camera_model')!r}")

    # Tracking project's `target_frame` block stores `lidar_from_camera` as
    # the rotation/translation pair used to transform a camera-frame point
    # into the lidar (target) frame. We accept either that direct form via
    # `lidar_from_camera_*` keys, or fall back to deriving it from the
    # legacy 4x4 `extrinsic` (camera_from_lidar) when the new keys are absent.
    if "lidar_from_camera_rotation" in config and "lidar_from_camera_translation" in config:
        rotation_lidar_from_camera = np.asarray(config["lidar_from_camera_rotation"], dtype=np.float64).reshape(3, 3)
        translation_lidar_from_camera = np.asarray(config["lidar_from_camera_translation"], dtype=np.float64).reshape(3)
    else:
        legacy = np.asarray(config["extrinsic"], dtype=np.float64).reshape(4, 4)
        R_camera_from_lidar = legacy[:3, :3]
        t_camera_from_lidar = legacy[:3, 3]
        rotation_lidar_from_camera = R_camera_from_lidar.T
        translation_lidar_from_camera = -R_camera_from_lidar.T @ t_camera_from_lidar
    colors = color_classes.astype("uint8")

    configure_cpu_runtime(cpu_threads)
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)
    # Lower scheduling priority so the GUI/IDE stays responsive even when
    # the model + lidar pipeline pegs CPU. Best-effort; not all platforms
    # allow this without privileges.
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass

    poses = load_ie_poses(ie_txt)
    if not poses:
        raise RuntimeError(f"No IE pose rows parsed from {ie_txt}")
    ecef_origin = geodetic_to_ecef(poses[0].latitude_deg, poses[0].longitude_deg, poses[0].height_m)
    enu_from_ecef = ecef_to_enu_matrix(poses[0].latitude_deg, poses[0].longitude_deg)

    # Single pose source: IE.txt for global ENU anchor. Per-point sub-frame
    # motion compensation comes from raw IMU (see ImuBuffer below), not from
    # the pose provider — pose interpolation at frame rate inherits noise
    # that corrupts intra-scan distortion correction.
    ie_provider = IePoseProvider(poses)
    print(f"[pose] IE-only (n={len(poses)} samples)")

    # Build IMU buffer for sub-frame de-distortion via first-pass bag read.
    # 200 Hz Livox internal IMU, hardware-synced with the LiDAR scanner.
    dedistort_method = str(config.get("dedistort_method", "imu")).lower()
    imu_buffer = None
    if dedistort_method == "imu":
        imu_topic = config.get("imu_topic", "/imu")
        imu_buffer = ImuBuffer()
        n_imu = 0
        with Bag(bag_path) as bag_imu:
            bag_start = bag_imu.get_start_time() + float(start_offset)
            bag_start_t = genpy.Time(bag_start)
            bag_end_t = None
            if duration is not None and float(duration) > 0:
                bag_end_t = genpy.Time(bag_start + float(duration))
            for _topic, msg, stamp in bag_imu.read_messages(
                topics=[imu_topic], start_time=bag_start_t, end_time=bag_end_t):
                t = message_stamp_sec(msg, stamp)
                gv = msg.angular_velocity
                imu_buffer.add(t, [gv.x, gv.y, gv.z])
                n_imu += 1
        imu_buffer.finalize()
        if n_imu < 2:
            print(f"[imu] WARNING: only {n_imu} samples from {imu_topic}; "
                  f"de-distortion will be a no-op")
            imu_buffer = None
        else:
            print(f"[imu] loaded {n_imu} samples from {imu_topic}, "
                  f"t∈[{imu_buffer.t[0]:.3f}, {imu_buffer.t[-1]:.3f}]")

    rospy.init_node("fix_distortion", anonymous=False, log_level=rospy.DEBUG, disable_signals=True)
    fixCloudPubHandle = rospy.Publisher("dedistortion_cloud", PointCloud2, queue_size=5)
    semanticCloudPubHandle = rospy.Publisher("SemanticCloud", PointCloud2, queue_size=5)
    roadCloudPubHandle = rospy.Publisher("RoadCloud", PointCloud2, queue_size=5)
    imgPubHandle = rospy.Publisher("Img", Image, queue_size=5)
    semimgPubHandle = rospy.Publisher("SemanticImg", Image, queue_size=5)
    groundTruthPubHandle = rospy.Publisher("ground_truth", Path, queue_size=0)
    _ = groundTruthPubHandle
    print("ros ready")

    predictor_kwargs = {}
    class_margin_min = config.get("class_margin_min")
    if class_margin_min:
        predictor_kwargs["class_margin_min"] = {int(k): float(v) for k, v in class_margin_min.items()}
    predictor = getattr(predict, config["predict_func"])(
        config["model_config"], config["model_file"], **predictor_kwargs,
    )
    print("torch ready")

    bag = None
    store_file = None
    road_chunks_file = None
    road_chunks_path = None
    pose_save = []
    road_chunk_count = 0
    index = 0
    msg_total = 0
    msg_lidar = 0
    msg_camera = 0
    processed_with_lidar = 0

    lidar_queue = deque(maxlen=max(10, int(lidar_queue_size)))
    image_queue = deque()

    try:
        bag = Bag(bag_path)
        start = bag.get_start_time() + float(start_offset)
        start_t = genpy.Time.from_sec(start)
        end_t = None if float(duration) == -1 else genpy.Time.from_sec(start + float(duration))
        bagread = bag.read_messages(topics=[lidar_topic, camera_topic], start_time=start_t, end_time=end_t)
        print("bag ready")

        cmkdir(config["save_folder"] + "/originpics")
        cmkdir(config["save_folder"] + "/sempics")
        store_file = open(config["save_folder"] + "/outdoor.pkl", "wb")
        # Stream road-class points to a chunks file instead of holding the
        # whole-run list in memory (was ~1.6 GB resident on the 21-min bag).
        # Reassembled into road.pcd in `finally` only if RAM allows.
        road_chunks_path = config["save_folder"] + "/road_chunks.pkl"
        road_chunks_file = open(road_chunks_path, "wb")

        for topic, msg, stamp in bagread:
            if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                break
            msg_total += 1
            ts = float(stamp.to_sec())
            if topic == lidar_topic:
                msg_lidar += 1
                try:
                    packet = livox_msg_to_points_with_meta(msg)
                except Exception as e:
                    print(f"skip lidar frame at {ts:.3f}: {e}")
                    continue
                packet = reduce_lidar_packet(packet, stride=lidar_stride, max_points=max_lidar_points)
                base_ts = float(packet["base_timestamp"])
                undistorted_xyzi = undistort_livox_points_imu(
                    packet["points_xyzi"],
                    packet["offsets_s"],
                    base_ts,
                    imu_buffer,
                )
                lidar_queue.append({"timestamp": base_ts, "points_xyzi": undistorted_xyzi})
            elif topic == camera_topic:
                msg_camera += 1
                image_queue.append((message_stamp_sec(msg, stamp), msg))
            else:
                continue

            while image_queue:
                if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                    break
                img_ts, img_msg = image_queue[0]
                nearest = choose_nearest_lidar(lidar_queue, img_ts, max_sync_dt)
                if nearest is None:
                    break
                if args.max_frames is not None and index >= int(args.max_frames):
                    request_max_frames_stop(args.max_frames)
                    image_queue.clear()
                    break
                try:
                    img = decode_image_msg(img_msg)
                except Exception as e:
                    print(f"skip image frame at {img_ts:.3f}: {e}")
                    image_queue.popleft()
                    continue

                lidar_pose = ie_provider.get_interpolated(float(nearest["timestamp"]))
                image_pose = ie_provider.get_interpolated(float(img_ts))
                if lidar_pose is None or image_pose is None:
                    image_queue.popleft()
                    continue

                align_pcd = transform_lidar_points_between_timestamps(
                    nearest["points_xyzi"],
                    lidar_pose,
                    image_pose,
                    ecef_origin,
                    enu_from_ecef,
                )

                fixcloud = get_i_pcd_msg(align_pcd)
                fixcloud.header.frame_id = "lidar"
                fixCloudPubHandle.publish(fixcloud)

                index += 1
                processed_with_lidar += 1
                print(f"processing frame {index} img_ts={img_ts:.3f} lidar_ts={nearest['timestamp']:.3f}")

                q = ie_pose_to_quaternion_xyzw(image_pose)
                t_enu = pose_enu_from_ie(image_pose, ecef_origin, enu_from_ecef)
                pose_save.append(np.array([t_enu[0], t_enu[1], t_enu[2], q[0], q[1], q[2], q[3]], dtype=np.float64))

                R_enu_from_body_cur = rotation_matrix_from_ie(
                    image_pose.roll_deg, image_pose.pitch_deg, image_pose.heading_deg)
                R_enu_from_lidar_cur = R_enu_from_body_cur @ R_BASE_FROM_LIDAR
                t_enu_lidar_cur = R_enu_from_body_cur @ T_FLU_LIDAR_FROM_SPAN + t_enu

                class_plane_coefs_cur = {}
                for cls_id, est in rolling_planes.items():
                    coefs = est.fit_in_target_lidar(R_enu_from_lidar_cur, t_enu_lidar_cur)
                    if coefs is not None:
                        class_plane_coefs_cur[cls_id] = coefs

                imgPubHandle.publish(bri.cv2_to_imgmsg(img))
                cv2.imwrite(config["save_folder"] + "/originpics/%06d.png" % index, img)

                sem_pcd_lidar, semimg = get_semantic_pcd(
                    img,
                    align_pcd,
                    camera_matrix,
                    dist_coeffs,
                    rotation_lidar_from_camera,
                    translation_lidar_from_camera,
                    camera_model,
                    predictor,
                    predict_image_scale=predict_image_scale,
                    min_segment_pixels=min_segment_pixels,
                    seg_filter_classes=seg_filter_classes,
                    class_erode_px=class_erode_px,
                    class_max_depth=class_max_depth,
                    class_plane_filter=class_plane_filter,
                    class_plane_coefs=class_plane_coefs_cur,
                    zbuffer_px=projection_zbuffer_px,
                    suppress_near_person_px=suppress_near_person_px,
                    person_class=person_class,
                )
                cv2.imwrite(config["save_folder"] + "/sempics/%06d.png" % index, semimg)
                semimg_vis = colors[semimg.flatten()].reshape((*semimg.shape, 3))
                semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg_vis, "bgr8"))

                if rolling_planes and len(sem_pcd_lidar) > 0:
                    pts_lidar_xyz = sem_pcd_lidar[:, :3]
                    cam_z = (rotation_lidar_from_camera.T @ (pts_lidar_xyz - translation_lidar_from_camera).T)[2]
                    for cls_id, est in rolling_planes.items():
                        params = class_plane_filter.get(cls_id, {})
                        fit_depth = float(params.get("fit_max_depth", 3.0))
                        near_mask = (sem_pcd_lidar[:, 3] == float(cls_id)) & (cam_z <= fit_depth) & (cam_z > 0.0)
                        if near_mask.any():
                            est.add(R_enu_from_lidar_cur, t_enu_lidar_cur, pts_lidar_xyz[near_mask])

                sem_world_pcd = transform_semantic_lidar_to_world(sem_pcd_lidar, image_pose, ecef_origin, enu_from_ecef)
                pickle.dump(sem_world_pcd, store_file)
                store_file.flush()

                if len(sem_world_pcd) != 0:
                    sem_msg = get_rgba_pcd_msg(sem_world_pcd)
                    sem_msg.header.frame_id = "world"
                    semanticCloudPubHandle.publish(sem_msg)
                    road_world_pcd = sem_world_pcd[sem_world_pcd[:, 3] == road_class]
                    if len(road_world_pcd) != 0:
                        pickle.dump(road_world_pcd, road_chunks_file)
                        road_chunk_count += 1
                        road_msg = get_rgba_pcd_msg(road_world_pcd)
                        road_msg.header.frame_id = "world"
                        roadCloudPubHandle.publish(road_msg)
                else:
                    print("semantic point publish skipped: empty semantic frame saved for alignment")
                image_queue.popleft()
            if STOP_REQUESTED or STOP_AFTER_MAX_FRAMES or rospy.is_shutdown():
                break
    except rospy.ROSInterruptException:
        request_stop("rospy.ROSInterruptException")
    except KeyboardInterrupt:
        request_stop("KeyboardInterrupt")
    finally:
        if store_file is not None:
            store_file.close()
        if road_chunks_file is not None:
            road_chunks_file.close()
        if bag is not None:
            bag.close()
        pose_array = np.stack(pose_save) if len(pose_save) != 0 else np.empty((0, 7))
        np.savetxt(config["save_folder"] + "/pose.csv", pose_array, delimiter=",")
        if road_chunks_path is not None and road_chunk_count > 0:
            try:
                chunks = []
                with open(road_chunks_path, "rb") as rf:
                    while True:
                        try:
                            chunks.append(pickle.load(rf))
                        except EOFError:
                            break
                if chunks:
                    save_nppc(np.vstack(chunks), config["save_folder"] + "/road.pcd")
                # Reassembly succeeded; the chunks file is redundant.
                try:
                    os.remove(road_chunks_path)
                except OSError:
                    pass
            except (MemoryError, Exception) as e:
                # Keep the chunks file around so the user can reassemble
                # offline (e.g. via tools/save_semantic_pcd.py with
                # --keep-classes <road_class>).
                print(
                    f"warning: failed to write road.pcd ({e!r}); "
                    f"chunks preserved at {road_chunks_path}"
                )
        if STOP_REASON:
            print(f"exit: {STOP_REASON}")
        print(
            "summary: total_msgs=%d lidar=%d camera=%d processed_frames=%d aligned_frames=%d"
            % (msg_total, msg_lidar, msg_camera, index, processed_with_lidar)
        )
        if index == 0:
            print("warning: processed_frames is 0. Check topics/start_time/duration/max_sync_dt and IE txt range.")


if __name__ == "__main__":
    main()
