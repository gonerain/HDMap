import glob
import json
import os
import pickle
import time
from abc import ABC, abstractmethod

import alphashape
import cv2
import numpy as np
import pandas as pd
import rospy
import tf
from cv_bridge import CvBridge
from pclpy import pcl
from predict import get_colors
from sensor_msgs.msg import Image, PointCloud2
from shapely.geometry import LineString, MultiPolygon, Polygon
from sklearn.cluster import DBSCAN
from util import get_rgba_pcd_msg


class FixedQueue(list):
    def __init__(self, capacity=-1):
        super().__init__()
        self.capacity = capacity

    def append(self, obj):
        if self.capacity != -1 and len(self) >= self.capacity:
            self.pop(0)
        super().append(obj)

    def is_full(self):
        return self.capacity != -1 and len(self) >= self.capacity


def save_nppc(nparr, fname, color_classes):
    if nparr.shape[1] == 4:
        tmp = pcl.PointCloud.PointXYZRGBA(
            nparr[:, :3],
            np.array([color_classes[int(i)] for i in nparr[:, 3]]),
        )
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname, tmp)
    return tmp


def extract_outline_by_alphashape(points_xyz, alpha=0.8):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if len(points_xyz) < 4:
        return np.zeros((0, 2), dtype=np.float32)

    shape = alphashape.alphashape(points_xyz[:, :2], alpha)
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


def keep_largest_cluster(points_xyz, eps=1.0, min_samples=20):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3 or len(points_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points_xyz) < min_samples:
        return points_xyz[:, :3].astype(np.float32, copy=False)

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=24).fit_predict(points_xyz[:, :3])
    valid = labels >= 0
    if not np.any(valid):
        return points_xyz[:, :3].astype(np.float32, copy=False)

    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    largest_label = cluster_ids[np.argmax(counts)]
    return points_xyz[labels == largest_label, :3].astype(np.float32, copy=False)


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

    return {
        "p1": p1.astype(np.float32),
        "p2": p2.astype(np.float32),
        "dir": seg_dir.astype(np.float32),
        "length": seg_len,
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
            score = 1.5 * len(cluster_points_xy) + seg["length"] - 0.2 * mean_abs_lon - 0.05 * mean_abs_lat
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
            "p1": p1.astype(np.float32, copy=False),
            "p2": p2.astype(np.float32, copy=False),
            "dir": seg_dir.astype(np.float32, copy=False),
            "length": seg_len,
            "lat": lat,
            "axis_offset": abs(float(np.dot(rel, axis))),
            "center_dist": float(np.linalg.norm(rel)),
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
            if abs(float(np.dot(left["dir"], right["dir"]))) < cos_pair:
                continue
            score = (
                left["length"] + right["length"]
                - 0.3 * (left["axis_offset"] + right["axis_offset"])
                - 0.1 * abs(abs(left["lat"]) - abs(right["lat"]))
                - 0.05 * (left["center_dist"] + right["center_dist"])
            )
            if score > best_score:
                best_score = score
                best_pair = [left, right]

    return best_pair or []


def build_canvas_transform(all_xy, canvas_size=1024, margin=60):
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


class RuntimeContext:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.color_classes = get_colors(config["cmap"])
        self.bri = CvBridge()
        self.br = tf.TransformBroadcaster()

        rospy.init_node("make_vector", anonymous=False, log_level=rospy.DEBUG, disable_signals=True)
        self.semantic_cloud_pub = rospy.Publisher("SemanticCloud", PointCloud2, queue_size=5)
        self.semimg_pub = rospy.Publisher("SemanticImg", Image, queue_size=5)
        self.img_pub = rospy.Publisher("Img", Image, queue_size=5)

        if args.semantic:
            self.simgs = sorted(glob.glob(args.semantic + "/*"))
            self.colors = self.color_classes.astype("uint8")
        else:
            self.simgs = []
            self.colors = None

        if args.origin:
            self.imgs = sorted(glob.glob(args.origin + "/*"))
        else:
            self.imgs = []

        if args.trajectory:
            self.poses = np.loadtxt(args.trajectory, delimiter=",")
        else:
            self.poses = None

    def publish_frame(self, frame_index, sempcd):
        display_pcd = sempcd
        if self.args.filters:
            display_pcd = display_pcd[np.in1d(display_pcd[:, 3], self.args.filters)]

        sem_msg = get_rgba_pcd_msg(display_pcd)
        sem_msg.header.frame_id = "world"
        self.semantic_cloud_pub.publish(sem_msg)

        if self.args.semantic and frame_index < len(self.simgs):
            simg = cv2.imread(self.simgs[frame_index], 0)
            semimg = self.colors[simg.flatten()].reshape((*simg.shape, 3))
            self.semimg_pub.publish(self.bri.cv2_to_imgmsg(semimg, "bgr8"))

        if self.args.origin and frame_index < len(self.imgs):
            self.img_pub.publish(self.bri.cv2_to_imgmsg(cv2.imread(self.imgs[frame_index]), "bgr8"))

    def publish_pose(self, frame_index):
        if self.poses is None or frame_index >= len(self.poses):
            return
        pose = self.poses[frame_index]
        rotation = pd.Series(pose[3:7], index=["x", "y", "z", "w"])
        self.br.sendTransform(
            (pose[0], pose[1], pose[2]),
            rotation,
            rospy.Time(time.time()),
            "odom",
            "world",
        )


class VectorProcess(ABC):
    name = ""
    target_class_key = ""
    default_target_class = None
    requires_pose = True
    step = 50
    window_size = 10
    dirc_window = 20
    static_dirc_thresh = 0.2

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.target_class = args.target_class
        if self.target_class is None:
            self.target_class = config.get(self.target_class_key, self.default_target_class)
        if self.target_class is None:
            raise ValueError(f"target class is not configured for process '{self.name}'")
        self.history = FixedQueue(2 * self.dirc_window + self.window_size)
        self.records = []

    def ingest_frame(self, sempcd):
        self.history.append(sempcd[sempcd[:, 3] == self.target_class])

    def ready(self):
        return self.history.is_full()

    def should_process(self, logical_index):
        return logical_index % self.step == 0

    def logical_index(self, latest_frame_index):
        return latest_frame_index - self.dirc_window - self.window_size + 1

    def build_context(self, runtime, logical_index):
        front_start = logical_index - self.dirc_window
        last_start = logical_index + self.dirc_window
        current_points = np.vstack(self.history[self.dirc_window:(self.dirc_window + self.window_size)])
        front_points = np.vstack(self.history[:self.window_size])
        last_points = np.vstack(self.history[(2 * self.dirc_window):(2 * self.dirc_window + self.window_size)])

        if runtime.poses is None:
            front_pose = None
            last_pose = None
        else:
            front_pose = runtime.poses[front_start:(front_start + self.window_size)].copy()
            last_pose = runtime.poses[last_start:(last_start + self.window_size)].copy()

        return {
            "current_points": current_points,
            "current_center": current_points[:, :2].mean(axis=0).astype(np.float32),
            "front": {
                "pose": front_pose,
                "points": front_points,
                "centerpoint": front_points[:, :2].mean(axis=0).astype(np.float32),
            },
            "last": {
                "pose": last_pose,
                "points": last_points,
                "centerpoint": last_points[:, :2].mean(axis=0).astype(np.float32),
            },
        }

    def has_valid_context(self, process_ctx):
        return (
            len(process_ctx["current_points"]) != 0
            and len(process_ctx["front"]["points"]) != 0
            and len(process_ctx["last"]["points"]) != 0
        )

    def draw_debug_canvas(self, cluster_points, contour_xy, process_ctx):
        all_xy_parts = [
            process_ctx["current_center"][None, :],
            process_ctx["front"]["centerpoint"][None, :],
            process_ctx["last"]["centerpoint"][None, :],
        ]
        if len(cluster_points) != 0:
            all_xy_parts.append(cluster_points[:, :2])
        if len(contour_xy) != 0:
            all_xy_parts.append(contour_xy[:, :2])

        to_canvas, canvas = build_canvas_transform(np.vstack(all_xy_parts))
        if len(cluster_points) != 0:
            for px, py in to_canvas(cluster_points[:, :2]):
                cv2.circle(canvas, (int(px), int(py)), 2, (60, 60, 60), -1)
        if len(contour_xy) >= 2:
            contour_canvas = to_canvas(contour_xy[:, :2]).reshape(-1, 1, 2)
            cv2.polylines(canvas, [contour_canvas], True, (0, 80, 255), 3)

        current_pt = tuple(to_canvas(process_ctx["current_center"][None, :])[0])
        front_pt = tuple(to_canvas(process_ctx["front"]["centerpoint"][None, :])[0])
        last_pt = tuple(to_canvas(process_ctx["last"]["centerpoint"][None, :])[0])
        cv2.circle(canvas, current_pt, 8, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, current_pt, front_pt, (0, 170, 255), 4, tipLength=0.12)
        cv2.arrowedLine(canvas, current_pt, last_pt, (0, 200, 0), 4, tipLength=0.12)
        cv2.putText(canvas, "current centroid", (current_pt[0] + 12, current_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(canvas, "front", (front_pt[0] + 12, front_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 220), 2)
        cv2.putText(canvas, "last", (last_pt[0] + 12, last_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
        return canvas, to_canvas

    def save_debug_canvas(self, logical_index, canvas):
        debug_dir = self.config.get("debug_output_dir", "demo_output")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{self.name}_centroid_debug_{logical_index:06d}.png"), canvas)

    def save_origin_debug_image(self, runtime, frame_index, logical_index):
        if not runtime.imgs or frame_index < 0 or frame_index >= len(runtime.imgs):
            return
        image = cv2.imread(runtime.imgs[frame_index])
        if image is None:
            return
        debug_dir = self.config.get("debug_output_dir", "demo_output")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{self.name}_origin_{logical_index:06d}.png"), image)

    def output_path(self):
        if self.args.output:
            return self.args.output
        return self.config.get(f"{self.name}_vector_output", f"{self.name}_edge_records.json")

    def finalize(self):
        output_path = self.output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"saved {self.name} vector records to {output_path} ({len(self.records)} records)")

    @abstractmethod
    def process(self, runtime, logical_index):
        pass
