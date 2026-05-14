"""Hardcoded mechanical extrinsics for the survey cart.

base/body axes are FLU (forward, left, up).
- base <-  LiDAR: R_BASE_FROM_LIDAR + T_BASE_FROM_LIDAR_M
- base <-  SPAN:  R_BASE_FROM_SPAN  + T_BASE_FROM_SPAN_M
                  (SPAN is RFU per NovAtel IE convention)

Convenience derived constants (LiDAR <- base) and the SPAN-relative LiDAR
translation used when the IE pose is given for the SPAN body are exported
so callers don't recompute them.

Values mirror /mnt/ning_602/work/tracking/configs/lidar_config.yaml
(ie_lidar_extrinsics block).
"""
import numpy as np


R_BASE_FROM_LIDAR = np.array(
    [[0.9063, 0.0, 0.4226],
     [0.0,    1.0, 0.0],
     [-0.4226, 0.0, 0.9063]],
    dtype=np.float64,
)
T_BASE_FROM_LIDAR_M = np.array([0.0315, 0.0, 0.1314], dtype=np.float64)

R_BASE_FROM_SPAN = np.array(
    [[0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
T_BASE_FROM_SPAN_M = np.array([-0.1854, 0.0, -0.242], dtype=np.float64)

R_LIDAR_FROM_BASE = R_BASE_FROM_LIDAR.T
T_FLU_LIDAR_FROM_SPAN = T_BASE_FROM_LIDAR_M - T_BASE_FROM_SPAN_M
