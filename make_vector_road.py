#!/usr/bin/python3
import sys
import pickle
import rospy
from sklearn.cluster import DBSCAN
from util import get_rgba_pcd_msg
from sensor_msgs.msg import PointCloud2,Image
import json
import numpy as np
import pandas as pd
import argparse
from pclpy import pcl
import tf
from cv_bridge import CvBridge
import time
import glob
import cv2
import re
from tf import transformations
from predict import get_colors

import alphashape
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection

height = {'pole': 5, 'lane':-1.1}

global sempcd
global args
global index
global poses
global br
global savepcd
global odom_trans
global last_points
global vectors
global top_vectors
global bottom_vectors
global lanepcd


class myqueue(list):
    def __init__(self, cnt=-1):
        self.cnt = cnt
        self.index = 0

    def append(self, obj):
        self.index+=1
        if len(self) >= self.cnt and self.cnt != -1:
            self.remove(self[0])
        super().append(obj)

    def is_empty(self):
        if len(self) == 0:
            return True
        else:
            return False

def color2int32(tup):
    return np.array([*tup[1:], 255]).astype(np.uint8).view('uint32')[0]


def class2color(cls,alpha = False):
    c = color_classes[cls]
    if not alpha:
        return np.array(c).astype(np.uint8)
    else:
        return np.array([*c, 255]).astype(np.uint8)

def save_nppc(nparr,fname):
    s = nparr.shape
    if s[1] == 4:#rgb
        tmp = pcl.PointCloud.PointXYZRGBA(nparr[:,:3],np.array([color_classes[int(i)] for i in nparr[:,3]]))
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname,tmp)
    return tmp

def draw_line(p1,p2):
    assert isinstance(p1,np.ndarray) or isinstance(p1,set)
    assert isinstance(p2,np.ndarray) or isinstance(p2,set)
    assert p1.shape == p2.shape
    if len(p1.shape) == 2 or p1.shape[0]== 2:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        line = np.stack((x,y),axis=1)
    else:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2+(p1[2]-p2[2])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        z = np.linspace(p1[2],p2[2],n)
        line = np.stack((x,y,z),axis=1)
    return line


def pcd_trans(pcd,dt,dr,inverse = False):
    length = len(pcd)
    if not isinstance(pcd,np.ndarray):
        pcd = np.array(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    if inverse:
        mat44 = np.matrix(mat44).I
    pcd[:3] = np.dot(mat44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd


def pointcloud_to_bev(points, resolution=0.2, x_range=None, y_range=None, z_range=None, min_points_per_cell=1, padding=0.0):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points must be an Nx3 or wider array')
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    if min_points_per_cell < 1:
        raise ValueError('min_points_per_cell must be >= 1')
    if padding < 0:
        raise ValueError('padding must be >= 0')

    xyz = points[:, :3].astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=1)
    if points.shape[1] > 3:
        cls = points[:, 3]
        valid &= np.isfinite(cls)
        cls = cls[valid].astype(np.int32, copy=False)
    else:
        cls = None
    xyz = xyz[valid]
    point_indices = np.flatnonzero(valid).astype(np.int32, copy=False)
    if len(xyz) == 0:
        empty_shape = (0, 0)
        return {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }

    if x_range is not None:
        keep = (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] < x_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if y_range is not None:
        keep = (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] < y_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if z_range is not None:
        keep = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] < z_range[1])
        xyz = xyz[keep]
        point_indices = point_indices[keep]
        if cls is not None:
            cls = cls[keep]
    if len(xyz) == 0:
        empty_shape = (0, 0)
        return {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }

    origin_xy = np.array([
        x_range[0] if x_range is not None else xyz[:, 0].min() - padding,
        y_range[0] if y_range is not None else xyz[:, 1].min() - padding,
    ], dtype=np.float32)
    max_xy = np.array([
        x_range[1] if x_range is not None else xyz[:, 0].max() + padding,
        y_range[1] if y_range is not None else xyz[:, 1].max() + padding,
    ], dtype=np.float32)

    span = np.maximum(max_xy - origin_xy, resolution)
    width = int(np.ceil(span[0] / resolution))
    height = int(np.ceil(span[1] / resolution))
    ij = np.floor((xyz[:, :2] - origin_xy) / resolution).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)

    count_map = np.zeros((height, width), dtype=np.uint16)
    np.add.at(count_map, (ij[:, 1], ij[:, 0]), 1)

    x_sum_map = np.zeros((height, width), dtype=np.float64)
    y_sum_map = np.zeros((height, width), dtype=np.float64)
    z_sum_map = np.zeros((height, width), dtype=np.float64)
    np.add.at(x_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 0])
    np.add.at(y_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 1])
    np.add.at(z_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])

    x_mean_map = np.divide(x_sum_map, count_map, out=np.zeros_like(x_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)
    y_mean_map = np.divide(y_sum_map, count_map, out=np.zeros_like(y_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)
    z_mean_map = np.divide(z_sum_map, count_map, out=np.zeros_like(z_sum_map, dtype=np.float64), where=count_map > 0).astype(np.float32)

    z_min_map = np.full((height, width), np.inf, dtype=np.float32)
    z_max_map = np.full((height, width), -np.inf, dtype=np.float32)
    np.minimum.at(z_min_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])
    np.maximum.at(z_max_map, (ij[:, 1], ij[:, 0]), xyz[:, 2])
    z_min_map[~np.isfinite(z_min_map)] = 0.0
    z_max_map[~np.isfinite(z_max_map)] = 0.0

    z_sq_sum_map = np.zeros((height, width), dtype=np.float64)
    np.add.at(z_sq_sum_map, (ij[:, 1], ij[:, 0]), xyz[:, 2] * xyz[:, 2])
    z_var_map = np.divide(z_sq_sum_map, count_map, out=np.zeros_like(z_sq_sum_map, dtype=np.float64), where=count_map > 0) - z_mean_map.astype(np.float64) ** 2
    z_var_map = np.maximum(z_var_map, 0.0).astype(np.float32)
    z_std_map = np.sqrt(z_var_map, dtype=np.float32)

    xyz_mean_map = np.stack((x_mean_map, y_mean_map, z_mean_map), axis=-1)
    bev = count_map >= min_points_per_cell

    z_median_map = np.zeros((height, width), dtype=np.float32)
    point_indices_map = {}
    class_hist_map = {}
    major_class_map = np.full((height, width), -1, dtype=np.int32)

    flat_ids = ij[:, 1] * width + ij[:, 0]
    unique_ids = np.unique(flat_ids)
    for flat_id in unique_ids:
        mask = flat_ids == flat_id
        row = int(flat_id // width)
        col = int(flat_id % width)
        z_median_map[row, col] = np.median(xyz[mask, 2]).astype(np.float32)
        point_indices_map[(row, col)] = point_indices[mask].copy()
        if cls is not None:
            classes, hist = np.unique(cls[mask], return_counts=True)
            classes = classes.astype(np.int32, copy=False)
            hist = hist.astype(np.int32, copy=False)
            class_hist_map[(row, col)] = {
                int(c): int(h) for c, h in zip(classes, hist)
            }
            major_class_map[row, col] = int(classes[np.argmax(hist)])

    return {
        'bev': bev,
        'count_map': count_map,
        'x_mean_map': x_mean_map,
        'y_mean_map': y_mean_map,
        'z_mean_map': z_mean_map,
        'xyz_mean_map': xyz_mean_map,
        'z_min_map': z_min_map,
        'z_max_map': z_max_map,
        'z_median_map': z_median_map,
        'z_var_map': z_var_map,
        'z_std_map': z_std_map,
        'major_class_map': major_class_map,
        'class_hist_map': class_hist_map,
        'point_indices_map': point_indices_map,
        'origin_xy': origin_xy,
        'resolution': float(resolution),
    }


def morph_open(binary_img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(binary_img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def flood_fill_holes(binary_img):
    filled = binary_img.copy()
    height, width = filled.shape[:2]
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(binary_img, holes)


def keep_largest_connected_component(binary_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return binary_img.copy()
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output = np.zeros_like(binary_img)
    output[labels == largest_label] = 255
    return output


def rotate_mask_to_major_axis(binary_img):
    mask = np.asarray(binary_img)
    if mask.ndim != 2:
        raise ValueError('binary_img must be a 2D array')

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(mask_u8)
    if points is None or len(points) < 5:
        return {
            'mask': mask_u8,
            'angle_deg': 0.0,
            'center': (mask_u8.shape[1] * 0.5, mask_u8.shape[0] * 0.5),
            'matrix': np.eye(2, 3, dtype=np.float32),
        }

    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle += 90.0

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    height, width = mask_u8.shape[:2]
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_width = int(np.ceil(width * cos + height * sin))
    new_height = int(np.ceil(width * sin + height * cos))

    rot_mat[0, 2] += new_width * 0.5 - cx
    rot_mat[1, 2] += new_height * 0.5 - cy

    rotated = cv2.warpAffine(
        mask_u8,
        rot_mat,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return {
        'mask': rotated,
        'angle_deg': float(angle),
        'center': (float(cx), float(cy)),
        'matrix': rot_mat.astype(np.float32),
    }


# Morphological thinning without opencv-contrib; suitable for extracting a coarse centerline.
def skeletonize_mask(binary_img):
    mask = np.asarray(binary_img)
    if mask.ndim != 2:
        raise ValueError('binary_img must be a 2D array')

    work = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = cv2.erode(work, element)
        if cv2.countNonZero(work) == 0:
            break

    return skeleton


def find_longest_top_bottom_edges(binary_img):
    if isinstance(binary_img, dict):
        binary_img = binary_img.get('mask')

    mask = np.asarray(binary_img)
    if mask.ndim != 2:
        raise ValueError('binary_img must be a 2D array')

    mask_u8 = (mask > 0).astype(np.uint8)
    height, width = mask_u8.shape
    if height == 0 or width == 0:
        return {
            'top': None,
            'bottom': None,
            'line_mask': np.zeros_like(mask_u8, dtype=np.uint8),
        }

    cols = np.flatnonzero(mask_u8.any(axis=0))
    if len(cols) == 0:
        return {
            'top': None,
            'bottom': None,
            'line_mask': np.zeros_like(mask_u8, dtype=np.uint8),
        }

    mid_col = int(cols[len(cols) // 2])
    rows = np.flatnonzero(mask_u8[:, mid_col])
    if len(rows) == 0:
        return {
            'top': None,
            'bottom': None,
            'line_mask': np.zeros_like(mask_u8, dtype=np.uint8),
        }

    top_point = np.array([mid_col, int(rows[0])], dtype=np.int32)
    bottom_point = np.array([mid_col, int(rows[-1])], dtype=np.int32)

    line_mask = np.zeros_like(mask_u8, dtype=np.uint8)
    line_mask[top_point[1], top_point[0]] = 255
    line_mask[bottom_point[1], bottom_point[0]] = 255

    return {
        'top': {
            'point': top_point,
            'points': top_point.reshape(1, 2),
            'col': int(mid_col),
        },
        'bottom': {
            'point': bottom_point,
            'points': bottom_point.reshape(1, 2),
            'col': int(mid_col),
        },
        'line_mask': line_mask,
    }


def bev_pixels_to_points(bev_data, rows, cols, z_value=None):
    rows = np.asarray(rows, dtype=np.int32).reshape(-1)
    cols = np.asarray(cols, dtype=np.int32).reshape(-1)
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    height, width = bev_data['bev'].shape[:2]
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.any(inside):
        return np.zeros((0, 3), dtype=np.float32)

    rows = rows[inside]
    cols = cols[inside]
    resolution = float(bev_data['resolution'])
    origin_xy = np.asarray(bev_data['origin_xy'], dtype=np.float32)
    xy = np.stack((
        origin_xy[0] + (cols.astype(np.float32) + 0.5) * resolution,
        origin_xy[1] + (rows.astype(np.float32) + 0.5) * resolution,
    ), axis=1)
    if z_value is None and 'z_mean_map' in bev_data:
        z = bev_data['z_mean_map'][rows, cols].astype(np.float32, copy=False).reshape(-1, 1)
    else:
        z = np.full((len(xy), 1), 0.0 if z_value is None else z_value, dtype=np.float32)
    return np.hstack((xy, z))


def bev_mask_to_points(bev_data, mask, z_value=None):
    if mask is None or mask.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rows, cols = np.nonzero(mask > 0)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return bev_pixels_to_points(bev_data, rows, cols, z_value=z_value)


def rotated_points_to_original_pixels(points, rotation_info, output_shape):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    inv_mat = cv2.invertAffineTransform(rotation_info['matrix'])
    homogeneous = np.hstack((points, np.ones((len(points), 1), dtype=np.float32)))
    original = homogeneous @ inv_mat.T
    original = np.rint(original).astype(np.int32)

    height, width = output_shape[:2]
    original[:, 0] = np.clip(original[:, 0], 0, width - 1)
    original[:, 1] = np.clip(original[:, 1], 0, height - 1)
    return original


def merge_road_masks(mask_entries, resolution=0.2, min_votes=1):
    occupied_points = []
    for entry in mask_entries:
        occupied = bev_mask_to_points(entry['bev'], entry['mask'])
        if len(occupied) != 0:
            occupied_points.append(occupied)

    if not occupied_points:
        empty_shape = (0, 0)
        empty_bev = {
            'bev': np.zeros(empty_shape, dtype=bool),
            'count_map': np.zeros(empty_shape, dtype=np.uint16),
            'x_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'y_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'z_mean_map': np.zeros(empty_shape, dtype=np.float32),
            'xyz_mean_map': np.zeros((0, 0, 3), dtype=np.float32),
            'z_min_map': np.zeros(empty_shape, dtype=np.float32),
            'z_max_map': np.zeros(empty_shape, dtype=np.float32),
            'z_median_map': np.zeros(empty_shape, dtype=np.float32),
            'z_var_map': np.zeros(empty_shape, dtype=np.float32),
            'z_std_map': np.zeros(empty_shape, dtype=np.float32),
            'major_class_map': np.full(empty_shape, -1, dtype=np.int32),
            'class_hist_map': {},
            'point_indices_map': {},
            'origin_xy': np.zeros(2, dtype=np.float32),
            'resolution': float(resolution),
        }
        return empty_bev, np.zeros(empty_shape, dtype=np.uint8)

    occupied_points = np.vstack(occupied_points)
    merged_bev = pointcloud_to_bev(
        occupied_points,
        resolution=resolution,
        min_points_per_cell=max(1, int(min_votes)),
    )
    merged_mask = merged_bev['bev'].astype(np.uint8) * 255
    if merged_mask.size != 0:
        merged_mask = morph_close(merged_mask, kernel_size=3, iterations=1)
        merged_mask = keep_largest_connected_component(merged_mask)
    return merged_bev, merged_mask


def filter_points_by_bev_mask(points, bev_data, mask):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return points
    if mask is None or mask.size == 0:
        return points[:0]

    height, width = mask.shape[:2]
    resolution = float(bev_data['resolution'])
    origin_xy = np.asarray(bev_data['origin_xy'], dtype=np.float32)

    ij = np.floor((points[:, :2] - origin_xy) / resolution).astype(np.int32)
    inside = (
        (ij[:, 0] >= 0) & (ij[:, 0] < width) &
        (ij[:, 1] >= 0) & (ij[:, 1] < height)
    )
    keep = np.zeros(len(points), dtype=bool)
    if np.any(inside):
        mask_values = mask[ij[inside, 1], ij[inside, 0]] > 0
        keep[np.flatnonzero(inside)] = mask_values
    return points[keep]


def get_lane_centers(pcd):
    centers = []
    if len(pcd) == 0:
        return centers
    dbs.fit(pcd)
    labels = dbs.fit_predict(pcd)  # label
    cluster = list(set(labels))
    n = len(cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        if abs(c[:,0].max()-c[:,0].min()) > 0.3:
            if c[:,0].mean() < 0:
                center = np.array((c[:, 0].max()-0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))
            else:
                center = np.array((c[:, 0].min()+0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))
        else:
            center = np.array((c[:,0].mean(),c[:,1].mean(),-2.3))
        centers.append(center)
    return centers

def get_pole_centers(pcd):
    centers = []
    if len(pcd) == 0:
        return centers
    pole_dbs.fit(pcd)
    labels = pole_dbs.fit_predict(pcd[:,:2])  # label
    cluster = list(set(labels))
    n = len(cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        center = np.array((c[:,0].mean(),c[:,1].mean(),c[:,2].max()))
        centers.append(center)
    return centers

def bev2world(frame_data):
    mask, bev_data, pose = (frame_data["mask"], frame_data["bev"], frame_data.get("pose"))
    if mask is None or mask.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rows, cols = np.nonzero(mask > 0)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    local_points = bev_pixels_to_points(bev_data, rows, cols)
    if len(local_points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    coord_frame = frame_data.get("coord_frame", "local")
    if coord_frame == "world" or pose is None or len(pose) < 7:
        return local_points.astype(np.float32, copy=False)

    return pcd_trans(local_points, pose[:3], pose[3:7], False).astype(np.float32, copy=False)

#TODO 道路延伸方向, 如果之前有dirc，则滤波
def find_dirc(pcd, pcd_last=None, dirc=None):
    def _stack_points(data):
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            clouds = []
            for item in data:
                arr = np.asarray(item)
                if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) != 0:
                    if arr.shape[1] >= 3:
                        clouds.append(arr[:, :3])
                    else:
                        clouds.append(np.pad(arr[:, :2], ((0, 0), (0, 1))))
            if not clouds:
                return np.zeros((0, 3), dtype=np.float32)
            return np.vstack(clouds).astype(np.float32, copy=False)
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.shape[1] == 2:
            arr = np.pad(arr, ((0, 0), (0, 1)))
        return arr[:, :3].astype(np.float32, copy=False)

    def _normalize(vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if len(vec) < 2:
            return None
        vec2 = vec[:2]
        norm = np.linalg.norm(vec2)
        if norm < 1e-6:
            return None
        return vec2 / norm

    points = _stack_points(pcd)
    if len(points) < 4:
        base = _normalize(dirc)
        if base is not None:
            return base
        return np.array([0.0, 1.0], dtype=np.float32)

    xy = points[:, :2]
    radius = np.linalg.norm(xy, axis=1)
    if not np.isfinite(radius).all():
        valid = np.isfinite(radius)
        xy = xy[valid]
        radius = radius[valid]
    if len(xy) < 4:
        base = _normalize(dirc)
        if base is not None:
            return base
        return np.array([0.0, 1.0], dtype=np.float32)

    inner_thresh = np.percentile(radius, 35.0)
    outer_thresh = np.percentile(radius, 65.0)
    inner_xy = xy[radius <= inner_thresh]
    outer_xy = xy[radius >= outer_thresh]
    if len(inner_xy) == 0 or len(outer_xy) == 0:
        split = np.median(radius)
        inner_xy = xy[radius <= split]
        outer_xy = xy[radius > split]
    if len(inner_xy) == 0 or len(outer_xy) == 0:
        base = _normalize(dirc)
        if base is not None:
            return base
        return np.array([0.0, 1.0], dtype=np.float32)

    inner_center = inner_xy.mean(axis=0)
    outer_center = outer_xy.mean(axis=0)
    direction = _normalize(outer_center - inner_center)
    if direction is None:
        base = _normalize(dirc)
        if base is not None:
            return base
        return np.array([0.0, 1.0], dtype=np.float32)

    last_points = _stack_points(pcd_last)
    history = _normalize(dirc)
    if history is None and len(last_points) >= 4:
        last_xy = last_points[:, :2]
        last_radius = np.linalg.norm(last_xy, axis=1)
        inner_last = last_xy[last_radius <= np.percentile(last_radius, 35.0)]
        outer_last = last_xy[last_radius >= np.percentile(last_radius, 65.0)]
        if len(inner_last) != 0 and len(outer_last) != 0:
            history = _normalize(outer_last.mean(axis=0) - inner_last.mean(axis=0))

    if history is not None and np.dot(direction, history) < 0:
        direction = -direction

    return direction.astype(np.float32, copy=False)

def keep_largest_road_cluster(roads, eps=1, min_samples=20):
    roads = np.asarray(roads, dtype=np.float32)
    if roads.ndim != 2 or roads.shape[1] < 3 or len(roads) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(roads) < min_samples:
        return roads[:, :3].astype(np.float32, copy=False)

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=24).fit_predict(roads[:, :3])
    valid = labels >= 0
    if not np.any(valid):
        return roads[:, :3].astype(np.float32, copy=False)

    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    largest_label = cluster_ids[np.argmax(counts)]
    return roads[labels == largest_label, :3].astype(np.float32, copy=False)


def road_outline_by_alphashape(roads, alpha=0.8):
      roads = np.asarray(roads, dtype=np.float32)
      if len(roads) < 4:
          return np.zeros((0, 2), dtype=np.float32)

      points_xy = roads[:, :2]
      shape = alphashape.alphashape(points_xy, alpha)

      if shape is None or shape.is_empty:
          return np.zeros((0, 2), dtype=np.float32)

      if isinstance(shape, MultiPolygon):
          shape = max(shape.geoms, key=lambda g: g.area)

      if isinstance(shape, Polygon):
          contour = np.asarray(shape.exterior.coords, dtype=np.float32)
          return contour

      if isinstance(shape, LineString):
          return np.asarray(shape.coords, dtype=np.float32)

      return np.zeros((0, 2), dtype=np.float32)


def simplify_polyline_by_slope(polyline_xy, angle_thresh_deg=12.0, min_seg_length=0.15):
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    if len(polyline_xy) < 3:
        return polyline_xy

    is_closed = np.linalg.norm(polyline_xy[0] - polyline_xy[-1]) < 1e-6
    work = polyline_xy[:-1].copy() if is_closed else polyline_xy.copy()
    if len(work) < 3:
        return polyline_xy

    def _unit(vec):
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    cos_thresh = float(np.cos(np.deg2rad(angle_thresh_deg)))
    simplified = [work[0]]
    prev_dir = None

    for i in range(1, len(work)): 
        vec = work[i] - simplified[-1] 
        seg_len = float(np.linalg.norm(vec)) 
        if seg_len < min_seg_length: 
            continue
        curr_dir = _unit(vec) 
        if curr_dir is None: 
            continue
        if prev_dir is None: 
            simplified.append(work[i]) 
            prev_dir = curr_dir 
            continue
        if abs(float(np.dot(prev_dir, curr_dir))) >= cos_thresh: 
            simplified[-1] = work[i] 
            prev_dir = _unit(simplified[-1] - simplified[-2]) if len(simplified) >= 2 else curr_dir 
        else: 
            simplified.append(work[i]) 
            prev_dir = _unit(simplified[-1] - simplified[-2])
        
    if len(simplified) == 1 or np.linalg.norm(simplified[-1] - work[-1]) >= min_seg_length: 
        simplified.append(work[-1])
    simplified = np.asarray(simplified, dtype=np.float32)
    if is_closed: 
        if len(simplified) >= 3: 
            first_dir = _unit(simplified[1] - simplified[0]) 
            last_dir = _unit(simplified[-1] - simplified[-2]) 
            if first_dir is not None and last_dir is not None and abs(float(np.dot(first_dir, last_dir))) >= cos_thresh:
                simplified[0] = simplified[-1] 
                simplified = simplified[:-1]
        simplified = np.vstack((simplified, simplified[:1])) 
    
    return simplified.astype(np.float32, copy=False)


def _collect_lines_from_geometry(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, GeometryCollection):
        lines = []
        for g in geom.geoms:
            lines.extend(_collect_lines_from_geometry(g))
        return lines
    return []


def polygon_centerline_and_mid_tangent(polyline_xy, ref_dir=None, slice_step=0.2, min_width=0.3):
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    empty = {
        'polygon': None,
        'centerline': np.zeros((0, 2), dtype=np.float32),
        'midpoint': None,
        'dirc': None,
    }
    if polyline_xy.ndim != 2 or polyline_xy.shape[1] < 2 or len(polyline_xy) < 4:
        return empty

    ring = polyline_xy[:, :2]
    if np.linalg.norm(ring[0] - ring[-1]) > 1e-6:
        ring = np.vstack((ring, ring[:1]))

    polygon = Polygon(ring)
    if polygon.is_empty or not polygon.is_valid or polygon.area < 1e-4:
        return empty

    axis = find_max_extent_direction(ring, ref_dir=ref_dir)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return empty
    axis = axis / axis_norm
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)

    coords = np.asarray(polygon.exterior.coords, dtype=np.float32)[:, :2]
    axis_vals = coords @ axis
    lateral_vals = coords @ lateral
    s_min = float(axis_vals.min())
    s_max = float(axis_vals.max())
    t_span = float(lateral_vals.max() - lateral_vals.min())
    if s_max - s_min < slice_step or t_span < min_width:
        return empty

    half_extent = max(t_span, slice_step) + 2.0
    centers = []
    s_values = np.arange(s_min, s_max + slice_step * 0.5, slice_step, dtype=np.float32)
    for s in s_values:
        p0 = axis * s - lateral * half_extent
        p1 = axis * s + lateral * half_extent
        cross_line = LineString([tuple(p0), tuple(p1)])
        inter = polygon.intersection(cross_line)
        segments = _collect_lines_from_geometry(inter)
        if not segments:
            continue
        seg = max(segments, key=lambda g: g.length)
        if seg.length < min_width:
            continue
        seg_coords = np.asarray(seg.coords, dtype=np.float32)
        midpoint = (seg_coords[0, :2] + seg_coords[-1, :2]) * 0.5
        centers.append(midpoint)

    if len(centers) < 2:
        return empty

    centerline = np.asarray(centers, dtype=np.float32)
    axis_proj = centerline @ axis
    centerline = centerline[np.argsort(axis_proj)]

    keep = [centerline[0]]
    for pt in centerline[1:]:
        if float(np.linalg.norm(pt - keep[-1])) >= max(slice_step * 0.5, 1e-3):
            keep.append(pt)
    centerline = np.asarray(keep, dtype=np.float32)
    if len(centerline) < 2:
        return empty

    mid_idx = len(centerline) // 2
    if len(centerline) >= 3:
        if mid_idx == 0:
            tangent = centerline[1] - centerline[0]
        elif mid_idx == len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[mid_idx + 1] - centerline[mid_idx - 1]
    else:
        tangent = centerline[-1] - centerline[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-6:
        tangent = axis.copy()
    else:
        tangent = tangent / tangent_norm

    if ref_dir is not None:
        ref = np.asarray(ref_dir, dtype=np.float32).reshape(-1)
        if len(ref) >= 2:
            ref = ref[:2]
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm > 1e-6:
                ref = ref / ref_norm
                if float(np.dot(tangent, ref)) < 0:
                    tangent = -tangent

    return {
        'polygon': polygon,
        'centerline': centerline.astype(np.float32, copy=False),
        'midpoint': centerline[mid_idx].astype(np.float32, copy=False),
        'dirc': tangent.astype(np.float32, copy=False),
    }


def render_road_debug_image(polyline_xy, centerline_xy=None, midpoint=None, dirc=None, left_edge_xy=None, right_edge_xy=None, image_size=900, margin=40):
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    items = []
    for arr in (polyline_xy, centerline_xy, left_edge_xy, right_edge_xy):
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) != 0:
            items.append(arr[:, :2])
    if midpoint is not None:
        midpoint = np.asarray(midpoint, dtype=np.float32).reshape(-1)
        if len(midpoint) >= 2:
            items.append(midpoint[:2][None, :])
    if not items:
        return canvas

    pts = np.vstack(items)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = float((image_size - 2 * margin) / max(span[0], span[1]))

    def _to_px(arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        px = (arr - min_xy) * scale + margin
        px[:, 1] = image_size - px[:, 1]
        return np.rint(px).astype(np.int32)

    poly_px = _to_px(polyline_xy[:, :2])
    if len(poly_px) >= 2:
        cv2.polylines(canvas, [poly_px.reshape(-1, 1, 2)], True, (0, 160, 0), 2, cv2.LINE_AA)

    if centerline_xy is not None:
        center_px = _to_px(np.asarray(centerline_xy, dtype=np.float32)[:, :2])
        if len(center_px) >= 2:
            cv2.polylines(canvas, [center_px.reshape(-1, 1, 2)], False, (255, 0, 0), 2, cv2.LINE_AA)

    if left_edge_xy is not None:
        left_px = _to_px(np.asarray(left_edge_xy, dtype=np.float32)[:, :2])
        if len(left_px) >= 2:
            cv2.polylines(canvas, [left_px.reshape(-1, 1, 2)], False, (0, 200, 255), 2, cv2.LINE_AA)

    if right_edge_xy is not None:
        right_px = _to_px(np.asarray(right_edge_xy, dtype=np.float32)[:, :2])
        if len(right_px) >= 2:
            cv2.polylines(canvas, [right_px.reshape(-1, 1, 2)], False, (255, 200, 0), 2, cv2.LINE_AA)

    if midpoint is not None:
        mid_px = _to_px(np.asarray(midpoint, dtype=np.float32)[:2][None, :])[0]
        cv2.circle(canvas, tuple(mid_px), 5, (0, 0, 255), -1, cv2.LINE_AA)
        if dirc is not None:
            dir_vec = np.asarray(dirc, dtype=np.float32).reshape(-1)
            if len(dir_vec) >= 2:
                dir_vec = dir_vec[:2]
                dir_norm = float(np.linalg.norm(dir_vec))
                if dir_norm > 1e-6:
                    dir_vec = dir_vec / dir_norm
                    arrow_len = max(50, int(image_size * 0.12))
                    end_px = np.rint(mid_px + np.array([dir_vec[0], -dir_vec[1]]) * arrow_len).astype(np.int32)
                    cv2.arrowedLine(canvas, tuple(mid_px), tuple(end_px), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.2)

    return canvas


def extract_road_edges_by_slices(points_xy, dirc, slice_step=0.2, min_points_per_slice=10, side_percentile=20.0):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    dirc = np.asarray(dirc, dtype=np.float32).reshape(-1)
    empty = {
        "left": np.zeros((0, 2), dtype=np.float32),
        "right": np.zeros((0, 2), dtype=np.float32),
        "corners": np.zeros((0, 2), dtype=np.float32),
    }
    if points_xy.ndim != 2 or points_xy.shape[1] < 2 or len(points_xy) < 2 or len(dirc) < 2:
        return empty

    forward = dirc[:2].astype(np.float32, copy=False)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return empty
    forward = forward / norm
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

    s_vals = points_xy @ forward
    t_vals = points_xy @ lateral
    s_min = float(s_vals.min())
    s_max = float(s_vals.max())
    if s_max - s_min < slice_step:
        return empty

    left_pts = []
    right_pts = []
    bins = np.arange(s_min, s_max + slice_step, slice_step, dtype=np.float32)
    for start in bins[:-1]:
        end = start + slice_step
        mask = (s_vals >= start) & (s_vals < end)
        if np.count_nonzero(mask) < min_points_per_slice:
            continue
        s_mid = float((start + end) * 0.5)
        slice_t = t_vals[mask]
        left_t = float(np.percentile(slice_t, side_percentile))
        right_t = float(np.percentile(slice_t, 100.0 - side_percentile))
        left_pts.append(forward * s_mid + lateral * left_t)
        right_pts.append(forward * s_mid + lateral * right_t)

    if len(left_pts) < 2 or len(right_pts) < 2:
        return empty

    left = np.asarray(left_pts, dtype=np.float32)
    right = np.asarray(right_pts, dtype=np.float32)
    corners = np.asarray([left[0], left[-1], right[0], right[-1]], dtype=np.float32)
    return {"left": left, "right": right, "corners": corners}


def find_parallel_segments_near_axis(polyline_xy, axis_dir, min_seg_length=0.5, axis_angle_thresh_deg=20.0, pair_angle_thresh_deg=10.0, min_lateral_gap=0.5):
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    axis_dir = np.asarray(axis_dir, dtype=np.float32).reshape(-1)
    if len(polyline_xy) < 2 or len(axis_dir) < 2:
        return []

    axis = axis_dir[:2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return []
    axis = axis / axis_norm
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
    cos_axis = float(np.cos(np.deg2rad(axis_angle_thresh_deg)))
    cos_pair = float(np.cos(np.deg2rad(pair_angle_thresh_deg)))

    segments = []
    num_edges = len(polyline_xy) - 1
    for i in range(num_edges):
        p1 = polyline_xy[i]
        p2 = polyline_xy[i + 1]
        vec = p2 - p1
        seg_len = float(np.linalg.norm(vec))
        if seg_len < min_seg_length:
            continue
        seg_dir = vec / seg_len
        if abs(float(np.dot(seg_dir, axis))) < cos_axis:
            continue
        if float(np.dot(seg_dir, axis)) < 0:
            seg_dir = -seg_dir
            p1, p2 = p2, p1
        midpoint = (p1 + p2) * 0.5
        lat = float(np.dot(midpoint, lateral))
        axis_pos = float(np.dot(midpoint, axis))
        segments.append({
            'p1': p1.astype(np.float32, copy=False),
            'p2': p2.astype(np.float32, copy=False),
            'dir': seg_dir.astype(np.float32, copy=False),
            'length': seg_len,
            'lat': lat,
            'axis_pos': axis_pos,
        })

    if len(segments) < 2:
        return []

    best_pair = None
    best_score = -1e9
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]
            if abs(float(np.dot(s1['dir'], s2['dir']))) < cos_pair:
                continue
            lateral_gap = abs(s1['lat'] - s2['lat'])
            if lateral_gap < min_lateral_gap:
                continue
            axis_overlap = -abs(s1['axis_pos'] - s2['axis_pos'])
            score = s1['length'] + s2['length'] + 0.5 * lateral_gap + 0.2 * axis_overlap
            if score > best_score:
                best_score = score
                best_pair = [s1, s2]

    if best_pair is None:
        return []
    best_pair.sort(key=lambda s: s['lat'])
    return best_pair


def find_max_extent_direction(polyline_xy, ref_dir=None, angle_step_deg=1.0):
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    if polyline_xy.ndim != 2 or polyline_xy.shape[1] < 2 or len(polyline_xy) < 2:
        return np.array([0.0, 1.0], dtype=np.float32)

    pts = polyline_xy[:, :2]
    best_dir = None
    best_span = -1.0
    for theta_deg in np.arange(0.0, 180.0, angle_step_deg, dtype=np.float32):
        theta = np.deg2rad(theta_deg)
        cand = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        proj = pts @ cand
        span = float(proj.max() - proj.min())
        if span > best_span:
            best_span = span
            best_dir = cand

    if best_dir is None:
        return np.array([0.0, 1.0], dtype=np.float32)

    if ref_dir is not None:
        ref = np.asarray(ref_dir, dtype=np.float32).reshape(-1)
        if len(ref) >= 2:
            ref = ref[:2]
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm > 1e-6:
                ref = ref / ref_norm
                if float(np.dot(best_dir, ref)) < 0:
                    best_dir = -best_dir

    return best_dir.astype(np.float32, copy=False)


def process():
    global sempcd
    global args
    global index
    global poses
    global br
    global last_points
    global vectors
    global top_vectors
    global bottom_vectors
    global lanepcd
    global roadpcd
    global last_roads_cluster
    global last_road_dirc

    if args.trajectory:
        p = poses[index]
        rotation = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((p[0], p[1], p[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            roads = sempcd[sempcd[:, 3] == config['road_class']]
            if len(roadpcd) < window_road:
                roadpcd.append(roads)
            else:
                roadpcd.append(roads)
                if index % step == 0:
                    roads = np.vstack(roadpcd)
                    roads = pcd_trans(roads, p, rotation, True)
                    roads = roads[(roads[:, 1] > 3) & (roads[:, 1] < 10)]

                    roads_cluster = keep_largest_road_cluster(roads[:, :3], eps=1, min_samples=20)
                    print('road cluster:', len(roads_cluster), '/', len(roads))
                    contour_xy = road_outline_by_alphashape(roads_cluster, alpha=0.8)
                    contour_xy = simplify_polyline_by_slope(contour_xy, angle_thresh_deg=15.0, min_seg_length=0.15)
                    if len(contour_xy) >= 4:
                        center_info = polygon_centerline_and_mid_tangent(
                            contour_xy,
                            ref_dir=last_road_dirc,
                            slice_step=0.2,
                            min_width=0.3,
                        )
                        dirc = center_info['dirc']
                        edge_info = {
                            'left': np.zeros((0, 2), dtype=np.float32),
                            'right': np.zeros((0, 2), dtype=np.float32),
                        }
                        if dirc is not None:
                            last_road_dirc = dirc
                            print('dirc:', dirc)
                            seg_pair = find_parallel_segments_near_axis(
                                contour_xy,
                                dirc,
                                min_seg_length=0.5,
                                axis_angle_thresh_deg=20.0,
                                pair_angle_thresh_deg=10.0,
                                min_lateral_gap=0.5,
                            )
                            if len(seg_pair) == 2:
                                edge_info['left'] = np.vstack((seg_pair[0]['p1'], seg_pair[0]['p2'])).astype(np.float32, copy=False)
                                edge_info['right'] = np.vstack((seg_pair[1]['p1'], seg_pair[1]['p2'])).astype(np.float32, copy=False)
                            print('road edges:', len(edge_info['left']), len(edge_info['right']))
                        debug_img = render_road_debug_image(
                            contour_xy,
                            centerline_xy=center_info['centerline'],
                            midpoint=center_info['midpoint'],
                            dirc=dirc,
                            left_edge_xy=edge_info['left'],
                            right_edge_xy=edge_info['right'],
                        )
                        cv2.imwrite(config['save_folder'] + '/road_debug.png', debug_img)
                    print(contour_xy)



    if args.filters:
        sempcd = sempcd[np.in1d(sempcd[:, 3], args.filters)]
    sem_msg = get_rgba_pcd_msg(sempcd)
    sem_msg.header.frame_id = 'world'
    semanticCloudPubHandle.publish(sem_msg)

    if args.semantic and index < len(simgs):
        simg = cv2.imread(simgs[index],0)
        semimg = colors[simg.flatten()].reshape((*simg.shape,3))
        semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg, 'bgr8'))
    if args.origin and index < len(imgs):
        imgPubHandle.publish(bri.cv2_to_imgmsg(cv2.imread(imgs[index]), 'bgr8'))

    if args.trajectory:
        index += 1


# parse arguments
parser = argparse.ArgumentParser(description='Rebuild semantic point cloud')
parser.add_argument('-c','--config',help='The config file path, recommand use this method to start the tool')
parser.add_argument('-i','--input',type=argparse.FileType('rb'))
parser.add_argument('-m','--mode',choices=['outdoor','indoor'],help="Depend on the way to store the pickle file")
parser.add_argument('-f','--filters', default=None,nargs='+',type=int,help='Default to show all the classes, the meaning of each class refers to class.json')
parser.add_argument('-s','--save',default=None,help='Save to pcd file')
parser.add_argument('-t','--trajectory',default=None,help='Trajectory file, use to follow the camera')
parser.add_argument('--semantic',default=None,help='Semantic photos folder')
parser.add_argument('--origin',default=None,help='Origin photos folder')
parser.add_argument('--vector',default=None,help='Do the vectorization, only available when filters are accepted',action='store_true')
args = parser.parse_args()

if args.config:
    with open(args.config,'r') as f:
        config = json.load(f)

args.input = (args.input or open(config['save_folder']+(config['mode'] == 'indoor' and '/indoor.pkl' or '/outdoor.pkl'),'rb'))
args.mode = (args.mode or config['mode'])
args.trajectory = (args.trajectory or config['save_folder']+'/pose.csv')
args.save = (args.save or config['save_folder']+'/result.pcd')
args.semantic = (args.semantic or config['save_folder']+'/sempics')
args.origin = (args.origin or config['save_folder']+'/originpics')
args.vector = (args.vector) or config['vector']
# init variables

window = 10
window_road = 10
step = 1

# start ros
rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.DEBUG)
semanticCloudPubHandle = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
vecPubHandle = rospy.Publisher('VectorCloud', PointCloud2, queue_size=5)
testPubHandle = rospy.Publisher('TestCloud', PointCloud2, queue_size=5)
semimgPubHandle = rospy.Publisher('SemanticImg',Image,queue_size = 5)
imgPubHandle = rospy.Publisher('Img',Image,queue_size = 5)

color_classes = get_colors(config['cmap'])
savepcd = []
vectors = []
top_vectors = []
bottom_vectors = []
bri = CvBridge()
index = 0
br = tf.TransformBroadcaster()
dbs = DBSCAN(eps = 0.3,min_samples=10,n_jobs=24)
pole_dbs = DBSCAN(eps = 0.3,min_samples=50,n_jobs=24)
last_points = myqueue(1)
lanepcd = myqueue(window)
polepcd = myqueue(window)
roadpcd = myqueue(window_road)

last_roads_cluster = None
last_road_dirc = None

pairs_all = []
vec_world = []

road_masks = myqueue(window)


if args.semantic:
    simgs = glob.glob(args.semantic+'/*')
    simgs.sort()
    #simgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))
    colors = color_classes.astype('uint8')

if args.origin:
    imgs = glob.glob(args.origin+'/*')
    imgs.sort()
    #imgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))

if args.trajectory:
    poses = np.loadtxt(args.trajectory, delimiter=',')


if args.mode == 'indoor':
    sempcds = pickle.load(args.input)
    for sempcd in sempcds:
        process()
    savepcd = np.concatenate(sempcds)
elif args.mode == 'outdoor':
    try:
        while index < 50:
            sempcd = pickle.load(args.input)
            savepcd.append(sempcd)
            process()
            print(index)
    except EOFError:
        print('done')

if isinstance(savepcd, list):
    if len(savepcd) != 0:
        savepcd = np.concatenate(savepcd)
    else:
        savepcd = np.zeros((0, 4), dtype=np.float32)

if args.vector:
    poles = savepcd[np.in1d(savepcd[:, 3], [config['pole_class']])]
    poles = poles[poles[:,2]>-2]
    poles = poles[poles[:,2]<8]
    pole_centers = get_pole_centers(poles[:, :3])
    poles = []
    for pole in pole_centers:
        poles.append(draw_line(pole,np.array([pole[0],pole[1],-2.3])))
    if len(poles) != 0:
        polemsg = get_rgba_pcd_msg(np.vstack(poles),color2int32((255,0,255,0)))
        vecPubHandle.publish(polemsg)


if args.save is not None:
    save_nppc(savepcd,args.save)
    save_dir = '/'.join(args.save.split('/')[:-1])
    vector_parts = []
    if len(vectors) != 0:
        all_vectors = np.vstack(vectors)
        vector_parts.append(all_vectors)
        save_nppc(all_vectors, save_dir + '/vector_points.pcd')
    if len(top_vectors) != 0:
        save_nppc(np.vstack(top_vectors), save_dir + '/vector_top_points.pcd')
    if len(bottom_vectors) != 0:
        save_nppc(np.vstack(bottom_vectors), save_dir + '/vector_bottom_points.pcd')
    if args.vector and len(poles) != 0:
        vector_parts.append(np.vstack(poles))
    if len(vector_parts) != 0:
        v = np.vstack(vector_parts)
        save_nppc(v, save_dir + '/vector.pcd')



def filter():
    srcpcd = './worldSemanticCloud.pcd'
    wsc = pcl.PointCloud.PointXYZRGBA()
    fpc = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile(srcpcd,wsc)
    f = pcl.filters.RadiusOutlierRemoval.PointXYZRGBA()
    f.setInputCloud(wsc)
    f.setRadiusSearch(0.1)
    f.setMinNeighborsInRadius(10)
    f.filter(fpc)
