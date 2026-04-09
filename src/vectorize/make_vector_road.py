#!/usr/bin/python3
import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import alphashape
import cv2
import numpy as np
import pandas as pd
import rospy
import tf
from cv_bridge import CvBridge
from pclpy import pcl
from sensor_msgs.msg import Image, PointCloud2
from shapely.geometry import LineString, MultiPolygon, Polygon
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import get_colors
from util import get_rgba_pcd_msg


class MyQueue(list):
    # Fixed-size FIFO used to keep the rolling road window.
    def __init__(self, cnt=-1):
        super().__init__()
        self.cnt = cnt

    def append(self, obj):
        if self.cnt != -1 and len(self) >= self.cnt:
            self.pop(0)
        super().append(obj)

    def is_full(self):
        return self.cnt != -1 and len(self) >= self.cnt


def save_nppc(nparr, fname):
    if nparr.shape[1] == 4:
        tmp = pcl.PointCloud.PointXYZRGBA(
            nparr[:, :3],
            np.array([color_classes[int(i)] for i in nparr[:, 3]]),
        )
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname, tmp)
    return tmp


def road_outline_by_alphashape(roads, alpha=0.8):
    # Extract a coarse outer boundary from the current road points.
    roads = np.asarray(roads, dtype=np.float32)
    if len(roads) < 4:
        return np.zeros((0, 2), dtype=np.float32)

    shape = alphashape.alphashape(roads[:, :2], alpha)
    if shape is None or shape.is_empty:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda geom: geom.area)

    if isinstance(shape, Polygon):
        return np.asarray(shape.exterior.coords, dtype=np.float32)
    if isinstance(shape, LineString):
        return np.asarray(shape.coords, dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def simplify_polyline_by_slope(polyline_xy, angle_thresh_deg=12.0, min_seg_length=0.15):
    # Merge short nearly-collinear segments to stabilize later pairing.
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    if len(polyline_xy) < 3:
        return polyline_xy

    is_closed = np.linalg.norm(polyline_xy[0] - polyline_xy[-1]) < 1e-6
    work = polyline_xy[:-1].copy() if is_closed else polyline_xy.copy()
    if len(work) < 3:
        return polyline_xy

    def unit(vec):
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
        curr_dir = unit(vec)
        if curr_dir is None:
            continue
        if prev_dir is None:
            simplified.append(work[i])
            prev_dir = curr_dir
            continue
        if abs(float(np.dot(prev_dir, curr_dir))) >= cos_thresh:
            simplified[-1] = work[i]
            prev_dir = unit(simplified[-1] - simplified[-2]) if len(simplified) >= 2 else curr_dir
        else:
            simplified.append(work[i])
            prev_dir = unit(simplified[-1] - simplified[-2])

    if len(simplified) == 1 or np.linalg.norm(simplified[-1] - work[-1]) >= min_seg_length:
        simplified.append(work[-1])

    simplified = np.asarray(simplified, dtype=np.float32)
    if is_closed:
        if len(simplified) >= 3:
            first_dir = unit(simplified[1] - simplified[0])
            last_dir = unit(simplified[-1] - simplified[-2])
            if first_dir is not None and last_dir is not None and abs(float(np.dot(first_dir, last_dir))) >= cos_thresh:
                simplified[0] = simplified[-1]
                simplified = simplified[:-1]
        simplified = np.vstack((simplified, simplified[:1]))

    return simplified.astype(np.float32, copy=False)


def keep_largest_road_cluster(roads, eps=1, min_samples=20):
    # Remove small disconnected road blobs before contour extraction.
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


def cluster_points(points_xy, eps, min_samples):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) == 0:
        return []
    if len(points_xy) < min_samples:
        return [points_xy.astype(np.float32, copy=False)]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xy)
    clusters = []
    for label in np.unique(labels):
        if label < 0:
            continue
        cluster = points_xy[labels == label]
        if len(cluster) != 0:
            clusters.append(cluster.astype(np.float32, copy=False))
    return clusters


def cluster_labels(points_xy, eps, min_samples):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) == 0:
        return np.full((0,), -1, dtype=np.int32)
    if len(points_xy) < min_samples:
        return np.zeros((len(points_xy),), dtype=np.int32)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xy).astype(np.int32)


def fit_edge_segment_from_points(points_xy, axis):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    if len(points_xy) < 2 or len(axis) < 2:
        return None

    axis = axis[:2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return None
    axis = axis / axis_norm

    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    seg_dir = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    if float(np.dot(seg_dir, axis)) < 0.0:
        seg_dir = -seg_dir

    proj = centered @ seg_dir
    if len(proj) == 0:
        return None

    p1 = mean + seg_dir * float(np.min(proj))
    p2 = mean + seg_dir * float(np.max(proj))
    seg_len = float(np.linalg.norm(p2 - p1))
    if seg_len < 1e-6:
        return None

    midpoint = 0.5 * (p1 + p2)
    return {
        'p1': p1.astype(np.float32),
        'p2': p2.astype(np.float32),
        'dir': seg_dir.astype(np.float32),
        'length': seg_len,
        'midpoint': midpoint.astype(np.float32),
    }


def find_edge_segments_in_frenet(
    edge_points_xy,
    centerpoint,
    dirc,
    min_lateral_gap=0.5,
    max_axis_offset=8.0,
    cluster_eps_longitudinal=1.2,
    cluster_eps_lateral=0.6,
    min_cluster_samples=6,
):
    edge_points_xy = np.asarray(edge_points_xy, dtype=np.float32)
    centerpoint = np.asarray(centerpoint, dtype=np.float32).reshape(-1)
    dirc = np.asarray(dirc, dtype=np.float32).reshape(-1)
    if len(edge_points_xy) == 0 or len(centerpoint) < 2 or len(dirc) < 2:
        return []

    axis = dirc[:2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return []
    axis = axis / axis_norm
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = centerpoint[:2]

    rel = edge_points_xy[:, :2] - center[None, :]
    lon = rel @ axis
    lat = rel @ lateral
    valid = np.abs(lon) <= float(max_axis_offset)
    valid &= np.abs(lat) >= float(min_lateral_gap)
    if not np.any(valid):
        return []

    points_valid = edge_points_xy[valid, :2]
    lon_valid = lon[valid]
    lat_valid = lat[valid]

    def pick_side(side_sign):
        side_mask = lat_valid * side_sign > 0.0
        if not np.any(side_mask):
            return None

        side_points = points_valid[side_mask]
        side_lon = lon_valid[side_mask]
        side_lat = lat_valid[side_mask]
        frenet = np.column_stack((
            side_lon / max(float(cluster_eps_longitudinal), 1e-3),
            side_lat / max(float(cluster_eps_lateral), 1e-3),
        )).astype(np.float32)
        labels = cluster_labels(frenet, eps=1.0, min_samples=min_cluster_samples)
        if not np.any(labels >= 0):
            if len(side_points) < min_cluster_samples:
                return None
            point_clusters = [side_points]
            lon_clusters = [side_lon]
            lat_clusters = [side_lat]
        else:
            point_clusters = []
            lon_clusters = []
            lat_clusters = []
            for label in np.unique(labels):
                if label < 0:
                    continue
                mask = labels == label
                point_clusters.append(side_points[mask])
                lon_clusters.append(side_lon[mask])
                lat_clusters.append(side_lat[mask])

        best_seg = None
        best_score = -1e9
        for cluster_points_xy, cluster_lon, cluster_lat in zip(point_clusters, lon_clusters, lat_clusters):
            seg = fit_edge_segment_from_points(cluster_points_xy, axis)
            if seg is None:
                continue
            mean_abs_lon = float(np.mean(np.abs(cluster_lon)))
            mean_abs_lat = float(np.mean(np.abs(cluster_lat)))
            score = (
                1.5 * len(cluster_points_xy)
                + seg['length']
                - 0.2 * mean_abs_lon
                - 0.05 * mean_abs_lat
            )
            if score > best_score:
                best_score = score
                best_seg = seg
        return best_seg

    left_seg = pick_side(-1.0)
    right_seg = pick_side(1.0)
    if left_seg is None or right_seg is None:
        return []
    return [left_seg, right_seg]


def find_parallel_segments_around_center(
    # Search the contour for one left and one right segment aligned with travel direction.
    polyline_xy,
    centerpoint,
    dirc,
    min_seg_length=0.5,
    dir_angle_thresh_deg=15.0,
    pair_angle_thresh_deg=10.0,
    min_lateral_gap=0.5,
):
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    centerpoint = np.asarray(centerpoint, dtype=np.float32).reshape(-1)
    dirc = np.asarray(dirc, dtype=np.float32).reshape(-1)
    if len(polyline_xy) < 2 or len(centerpoint) < 2 or len(dirc) < 2:
        return []

    axis = dirc[:2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return []

    axis = axis / axis_norm
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = centerpoint[:2]
    cos_dir = float(np.cos(np.deg2rad(dir_angle_thresh_deg)))
    cos_pair = float(np.cos(np.deg2rad(pair_angle_thresh_deg)))

    left_segments = []
    right_segments = []
    for i in range(len(polyline_xy) - 1):
        p1 = polyline_xy[i, :2]
        p2 = polyline_xy[i + 1, :2]
        vec = p2 - p1
        seg_len = float(np.linalg.norm(vec))
        if seg_len < min_seg_length:
            continue

        seg_dir = vec / seg_len
        align = float(np.dot(seg_dir, axis))
        if abs(align) < cos_dir:
            continue
        if align < 0:
            seg_dir = -seg_dir
            p1, p2 = p2, p1

        midpoint = 0.5 * (p1 + p2)
        rel = midpoint - center
        lat = float(np.dot(rel, lateral))
        if abs(lat) < min_lateral_gap:
            continue

        seg = {
            'p1': p1.astype(np.float32, copy=False),
            'p2': p2.astype(np.float32, copy=False),
            'dir': seg_dir.astype(np.float32, copy=False),
            'length': seg_len,
            'lat': lat,
            'axis_offset': abs(float(np.dot(rel, axis))),
            'center_dist': float(np.linalg.norm(rel)),
        }
        if lat < 0:
            left_segments.append(seg)
        else:
            right_segments.append(seg)

    if not left_segments or not right_segments:
        return []

    best_pair = None
    best_score = -1e9
    for left in left_segments:
        for right in right_segments:
            if abs(float(np.dot(left['dir'], right['dir']))) < cos_pair:
                continue
            score = (
                left['length'] + right['length']
                - 0.3 * (left['axis_offset'] + right['axis_offset'])
                - 0.1 * abs(abs(left['lat']) - abs(right['lat']))
                - 0.05 * (left['center_dist'] + right['center_dist'])
            )
            if score > best_score:
                best_score = score
                best_pair = [left, right]

    return best_pair or []


def build_road_context(index):
    # Split the rolling buffer into front/current/last windows around the current index.
    front_start = index - dirc_window
    last_start = index + dirc_window

    current_points = np.vstack(road_savedpcd[dirc_window:(dirc_window + window_road)])
    front_points = np.vstack(road_savedpcd[:window_road])
    last_points = np.vstack(road_savedpcd[(2 * dirc_window):(2 * dirc_window + window_road)])

    return {
        'current_points': current_points,
        'current_center': current_points[:, :2].mean(axis=0).astype(np.float32),
        'front': {
            'pose': poses[front_start:(front_start + window_road)].copy(),
            'points': front_points,
            'centerpoint': front_points[:, :2].mean(axis=0).astype(np.float32),
        },
        'last': {
            'pose': poses[last_start:(last_start + window_road)].copy(),
            'points': last_points,
            'centerpoint': last_points[:, :2].mean(axis=0).astype(np.float32),
        },
    }


def build_canvas_transform(all_xy, canvas_size=1024, margin=60):
    # Fit world XY coordinates into a debug canvas with consistent padding.
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = float(min((canvas_size - 2 * margin) / span[0], (canvas_size - 2 * margin) / span[1]))

    def to_canvas(points_xy):
        pts = np.asarray(points_xy, dtype=np.float32)
        canvas_pts = np.empty((len(pts), 2), dtype=np.int32)
        canvas_pts[:, 0] = np.round((pts[:, 0] - min_xy[0]) * scale + margin).astype(np.int32)
        canvas_pts[:, 1] = np.round((max_xy[1] - pts[:, 1]) * scale + margin).astype(np.int32)
        return canvas_pts

    return to_canvas, np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)


def draw_road_debug_canvas(roads_cluster, contour_xy, road_ctx):
    # Draw the road cloud, contour and front/last centroids before segment pairing.
    all_xy_parts = [
        road_ctx['current_center'][None, :],
        road_ctx['front']['centerpoint'][None, :],
        road_ctx['last']['centerpoint'][None, :],
    ]
    if len(roads_cluster) != 0:
        all_xy_parts.append(roads_cluster[:, :2])
    if len(contour_xy) != 0:
        all_xy_parts.append(contour_xy[:, :2])

    to_canvas, canvas = build_canvas_transform(np.vstack(all_xy_parts))
    if len(roads_cluster) != 0:
        for px, py in to_canvas(roads_cluster[:, :2]):
            cv2.circle(canvas, (int(px), int(py)), 2, (60, 60, 60), -1)
    if len(contour_xy) >= 2:
        contour_canvas = to_canvas(contour_xy[:, :2]).reshape(-1, 1, 2)
        cv2.polylines(canvas, [contour_canvas], True, (0, 80, 255), 3)

    current_pt = tuple(to_canvas(road_ctx['current_center'][None, :])[0])
    front_pt = tuple(to_canvas(road_ctx['front']['centerpoint'][None, :])[0])
    last_pt = tuple(to_canvas(road_ctx['last']['centerpoint'][None, :])[0])
    cv2.circle(canvas, current_pt, 8, (255, 255, 255), -1)
    cv2.arrowedLine(canvas, current_pt, front_pt, (0, 170, 255), 4, tipLength=0.12)
    cv2.arrowedLine(canvas, current_pt, last_pt, (0, 200, 0), 4, tipLength=0.12)
    cv2.putText(canvas, 'current centroid', (current_pt[0] + 12, current_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(canvas, 'front', (front_pt[0] + 12, front_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 220), 2)
    cv2.putText(canvas, 'last', (last_pt[0] + 12, last_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
    return canvas, to_canvas


def save_debug_canvas(index, canvas):
    cv2.imwrite(f'demo_output/road_centroid_debug_{index:06d}.png', canvas)


def make_road_edge_record(index, road_ctx, left_seg, right_seg, dirc, road_z):
    # Store the minimal per-frame geometry needed for later edge merging.
    return {
        'index': int(index),
        'centroid': road_ctx['current_center'].astype(np.float32).tolist(),
        'dirc': np.asarray(dirc, dtype=np.float32).tolist(),
        'road_z': float(road_z),
        'left_edge': {
            'p1': left_seg['p1'].astype(np.float32).tolist(),
            'p2': left_seg['p2'].astype(np.float32).tolist(),
        },
        'right_edge': {
            'p1': right_seg['p1'].astype(np.float32).tolist(),
            'p2': right_seg['p2'].astype(np.float32).tolist(),
        },
    }


def process_road_vectorization(index):
    # Main road-only vectorization path for the current sampled frame.
    if args.mode != 'outdoor' or not road_savedpcd.is_full():
        return False

    road_ctx = build_road_context(index)
    roads_cluster = keep_largest_road_cluster(road_ctx['current_points'][:, :3], eps=1, min_samples=20)
    print('road cluster:', len(roads_cluster), '/', len(road_ctx['current_points']))

    contour_xy = road_outline_by_alphashape(roads_cluster, alpha=0.8)
    contour_xy = simplify_polyline_by_slope(contour_xy, angle_thresh_deg=7.0, min_seg_length=0.15)

    dirc = road_ctx['last']['centerpoint'] - road_ctx['front']['centerpoint']
    dirc_norm = float(np.linalg.norm(dirc))
    canvas, to_canvas = draw_road_debug_canvas(roads_cluster, contour_xy, road_ctx)

    # When the vehicle is nearly static, keep the debug image and skip direction-based pairing.
    if dirc_norm < static_dirc_thresh:
        save_debug_canvas(index, canvas)
        print(f'static skip @ {index}: dirc_norm={dirc_norm:.4f}, thresh={static_dirc_thresh:.4f}')
        print(
            f'front={road_ctx["front"]["centerpoint"]}, '
            f'last={road_ctx["last"]["centerpoint"]}, '
            f'current={road_ctx["current_center"]}'
        )
        return False

    seg_pair = find_edge_segments_in_frenet(
        contour_xy,
        road_ctx['current_center'],
        dirc / dirc_norm,
        min_lateral_gap=0.5,
        max_axis_offset=8.0,
        cluster_eps_longitudinal=1.5,
        cluster_eps_lateral=0.8,
        min_cluster_samples=4,
    )
    if len(seg_pair) != 2:
        print('skip: no left/right edge clusters around centerpoint')
        return False

    left_seg, right_seg = seg_pair
    left_canvas = to_canvas(np.vstack((left_seg['p1'], left_seg['p2']))).reshape(-1, 1, 2)
    right_canvas = to_canvas(np.vstack((right_seg['p1'], right_seg['p2']))).reshape(-1, 1, 2)
    cv2.polylines(canvas, [left_canvas], False, (255, 80, 80), 5)
    cv2.polylines(canvas, [right_canvas], False, (80, 220, 80), 5)
    road_z = float(np.median(road_ctx['current_points'][:, 2])) if len(road_ctx['current_points']) != 0 else 0.0
    road_edge_records.append(make_road_edge_record(index, road_ctx, left_seg, right_seg, dirc / dirc_norm, road_z))
    save_debug_canvas(index, canvas)
    return True


def publish_semantic_outputs(index, sempcd):
    # ROS publishing is kept separate from vectorization so debug exits stay simple.
    display_pcd = sempcd
    if args.filters:
        display_pcd = display_pcd[np.in1d(display_pcd[:, 3], args.filters)]

    sem_msg = get_rgba_pcd_msg(display_pcd)
    sem_msg.header.frame_id = 'world'
    semantic_cloud_pub.publish(sem_msg)

    if args.semantic and index < len(simgs):
        simg = cv2.imread(simgs[index], 0)
        semimg = colors[simg.flatten()].reshape((*simg.shape, 3))
        semimg_pub.publish(bri.cv2_to_imgmsg(semimg, 'bgr8'))
    if args.origin and index < len(imgs):
        img_pub.publish(bri.cv2_to_imgmsg(cv2.imread(imgs[index]), 'bgr8'))


def process():
    # Advance one frame: publish pose, optionally vectorize road, then publish images/clouds.
    global index

    if args.trajectory:
        pose = poses[index]
        rotation = pd.Series(pose[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((pose[0], pose[1], pose[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            process_road_vectorization(index)

    index += 1
    publish_semantic_outputs(index, sempcd)


parser = argparse.ArgumentParser(description='Rebuild semantic point cloud')
parser.add_argument('-c', '--config', help='The config file path, recommand use this method to start the tool')
parser.add_argument('-i', '--input', type=argparse.FileType('rb'))
parser.add_argument('-m', '--mode', choices=['outdoor', 'indoor'], help='Depend on the way to store the pickle file')
parser.add_argument('-f', '--filters', default=None, nargs='+', type=int, help='Default to show all the classes, the meaning of each class refers to class.json')
parser.add_argument('-s', '--save', default=None, help='Save to pcd file')
parser.add_argument('-t', '--trajectory', default=None, help='Trajectory file, use to follow the camera')
parser.add_argument('--semantic', default=None, help='Semantic photos folder')
parser.add_argument('--origin', default=None, help='Origin photos folder')
parser.add_argument('--vector', default="result/outdoor/road_edge_records.json", help='Do the vectorization, only available when filters are accepted', action='store_true')
parser.add_argument('--max_index', default=10000, type=int, help='Max index to process, default to process all data')
parser.add_argument('--start_index', default=None, type=int, help='Start processing from this frame index')
args = parser.parse_args()

if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)

args.input = args.input or open(config['save_folder'] + ('/indoor.pkl' if config['mode'] == 'indoor' else '/outdoor.pkl'), 'rb')
args.mode = args.mode or config['mode']
args.filters = args.filters or config.get('filters')
args.trajectory = args.trajectory or config['save_folder'] + '/pose.csv'
args.save = args.save or config['save_folder'] + '/result.pcd'
args.semantic = args.semantic or config['save_folder'] + '/sempics'
args.origin = args.origin or config['save_folder'] + '/originpics'
args.vector = args.vector or config['vector']
if args.max_index == parser.get_default('max_index'):
    args.max_index = config.get('max_index', args.max_index)
if args.start_index is None:
    args.start_index = config.get('start_index')

step = 2
window_road = 10
dirc_window = 20
static_dirc_thresh = 0.2
road_savedpcd = MyQueue(2 * dirc_window + window_road)
index = dirc_window
start_index = args.start_index if args.start_index is not None else dirc_window
start_index = max(start_index, dirc_window)
index = start_index
warmup_skip = max(0, start_index - dirc_window)
skipped = 0

rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.DEBUG)
semantic_cloud_pub = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
semimg_pub = rospy.Publisher('SemanticImg', Image, queue_size=5)
img_pub = rospy.Publisher('Img', Image, queue_size=5)

color_classes = get_colors(config['cmap'])
bri = CvBridge()
br = tf.TransformBroadcaster()

if args.semantic:
    simgs = sorted(glob.glob(args.semantic + '/*'))
    colors = color_classes.astype('uint8')
else:
    simgs = []
    colors = None

if args.origin:
    imgs = sorted(glob.glob(args.origin + '/*'))
else:
    imgs = []

if args.trajectory:
    poses = np.loadtxt(args.trajectory, delimiter=',')
else:
    poses = None

savepcd = []
road_edge_records = []
road_edge_record_path = 'demo_output/road/road_edge_records.json'
if args.mode == 'indoor':
    sempcds = pickle.load(args.input)
    for sempcd in sempcds:
        process()
    savepcd = np.concatenate(sempcds) if len(sempcds) != 0 else np.zeros((0, 4), dtype=np.float32)
elif args.mode == 'outdoor':
    try:
        while skipped < warmup_skip:
              pickle.load(args.input)
              skipped += 1
        while True:
            if index >= args.max_index:
                print('reach max index, stop processing')
                break
            sempcd = pickle.load(args.input)
            road_savedpcd.append(sempcd[sempcd[:, 3] == config['road_class']])
            if not road_savedpcd.is_full():
                continue
            if index % step != 0:
                index += 1
                continue
            process()
            print(index)
    except EOFError:
        print('done')
    savepcd = np.zeros((0, 4), dtype=np.float32)

if args.vector:
    os.makedirs(os.path.dirname(road_edge_record_path), exist_ok=True)
    with open(road_edge_record_path, 'w') as f:
        json.dump(road_edge_records, f, indent=2)
    print(f'saved road edge records to {road_edge_record_path} ({len(road_edge_records)} records)')

if args.save is not None and len(savepcd) != 0:
    save_nppc(savepcd, args.save)
