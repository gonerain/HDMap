import cv2
import numpy as np

from core.vector_common import default_demo_output_dir
from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import cluster_labels
from core.vector_common import simplify_polyline_by_slope
from core.geometry_utils import polygon_from_outline


class CrosswalkProcess(VectorProcess):
    name = "crosswalk"
    target_class_key = "crosswalk_class"
    default_target_class = 23
    cluster_eps = 0.8
    cluster_min_samples = 12
    outline_alpha = 1.0
    simplify_angle_thresh_deg = 10.0
    simplify_min_seg_length = 0.1
    min_crosswalk_length = 1.0
    max_crosswalk_length = 10.0
    min_crosswalk_width = 0.4
    max_crosswalk_width = 3.0
    min_aspect_ratio = 1.2
    max_aspect_ratio = 10.0
    min_rectangularity = 0.5

    def output_path(self):
        if self.args.output:
            return self.args.output
        return self.config.get(
            "crosswalk_vector_output",
            f"{default_demo_output_dir(self.name)}/crosswalk_records.json",
        )

    def make_record(self, logical_index, cluster_center, cluster_points, contour_xy, dirc, lateral, long_range, lat_range, crosswalk_z):
        center_xy = cluster_center.astype(np.float32)
        p1 = center_xy + dirc * long_range[0] + lateral * lat_range[0]
        p2 = center_xy + dirc * long_range[0] + lateral * lat_range[1]
        p3 = center_xy + dirc * long_range[1] + lateral * lat_range[1]
        p4 = center_xy + dirc * long_range[1] + lateral * lat_range[0]
        corners = np.vstack((p1, p2, p3, p4)).astype(np.float32)

        return {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "centroid": center_xy.tolist(),
            "dirc": np.asarray(dirc, dtype=np.float32).tolist(),
            "lateral": np.asarray(lateral, dtype=np.float32).tolist(),
            "crosswalk_z": float(crosswalk_z),
            "length": float(long_range[1] - long_range[0]),
            "width": float(lat_range[1] - lat_range[0]),
            "corners": corners.tolist(),
            "outline": np.asarray(contour_xy, dtype=np.float32).tolist(),
            "point_count": int(len(cluster_points)),
        }

    def estimate_crosswalk_axes(self, contour_xy, cluster_center, motion_dir):
        contour_xy = np.asarray(contour_xy, dtype=np.float32)
        cluster_center = np.asarray(cluster_center, dtype=np.float32).reshape(-1)
        motion_dir = np.asarray(motion_dir, dtype=np.float32).reshape(-1)
        if len(contour_xy) < 2 or len(cluster_center) < 2 or len(motion_dir) < 2:
            return None

        rel = contour_xy[:, :2] - cluster_center[:2][None, :]
        cov = rel.T @ rel
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis_major = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
        axis_major_norm = float(np.linalg.norm(axis_major))
        if axis_major_norm < 1e-6:
            return None
        axis_major /= axis_major_norm

        motion_lat = np.array([-motion_dir[1], motion_dir[0]], dtype=np.float32)
        if float(np.dot(axis_major, motion_lat)) < 0.0:
            axis_major = -axis_major
        axis_minor = np.array([-axis_major[1], axis_major[0]], dtype=np.float32)
        if float(np.dot(axis_minor, motion_dir[:2])) < 0.0:
            axis_minor = -axis_minor

        proj_major = rel @ axis_major
        proj_minor = rel @ axis_minor
        major_range = np.array([np.min(proj_major), np.max(proj_major)], dtype=np.float32)
        minor_range = np.array([np.min(proj_minor), np.max(proj_minor)], dtype=np.float32)

        return {
            "dirc": axis_major,
            "lateral": axis_minor,
            "long_range": major_range,
            "lat_range": minor_range,
            "length": float(major_range[1] - major_range[0]),
            "width": float(minor_range[1] - minor_range[0]),
        }

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        motion_dir = process_ctx["last"]["centerpoint"] - process_ctx["front"]["centerpoint"]
        motion_dir_norm = float(np.linalg.norm(motion_dir))
        if motion_dir_norm < self.static_dirc_thresh:
            print(f"static skip @ {logical_index}: dirc_norm={motion_dir_norm:.4f}, thresh={self.static_dirc_thresh:.4f}")
            canvas, _ = self.draw_debug_canvas(np.zeros((0, 3)), np.zeros((0, 2)), process_ctx)
            self.save_debug_canvas(logical_index, canvas)
            self.save_origin_debug_image(runtime, logical_index, logical_index)
            return False

        motion_dir = (motion_dir / motion_dir_norm).astype(np.float32)

        merged_points = process_ctx["current_points"]
        if len(merged_points) == 0:
            return False

        labels = cluster_labels(merged_points[:, :2], eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]

        print(f"{self.name} clusters: {len(valid_labels)} / {len(merged_points)} points")

        canvas, to_canvas = self.draw_debug_canvas(np.zeros((0, 3)), np.zeros((0, 2)), process_ctx)
        any_detected = False

        for label in valid_labels:
            mask = labels == label
            cluster_pts = merged_points[mask, :3]
            if len(cluster_pts) < self.cluster_min_samples:
                continue

            contour_xy = extract_outline_by_alphashape(cluster_pts, alpha=self.outline_alpha)
            if len(contour_xy) < 4:
                continue
            contour_xy = simplify_polyline_by_slope(
                contour_xy,
                angle_thresh_deg=self.simplify_angle_thresh_deg,
                min_seg_length=self.simplify_min_seg_length,
            )
            if len(contour_xy) < 3:
                continue

            cluster_center = cluster_pts[:, :2].mean(axis=0).astype(np.float32)
            axes = self.estimate_crosswalk_axes(contour_xy, cluster_center, motion_dir)
            if axes is None:
                continue

            record_dirc = axes["dirc"]
            record_lateral = axes["lateral"]
            long_range = axes["long_range"]
            lat_range = axes["lat_range"]
            crosswalk_length = axes["length"]
            crosswalk_width = axes["width"]
            aspect_ratio = crosswalk_length / max(crosswalk_width, 1e-6)

            if (
                crosswalk_length < self.min_crosswalk_length
                or crosswalk_length > self.max_crosswalk_length
                or crosswalk_width < self.min_crosswalk_width
                or crosswalk_width > self.max_crosswalk_width
            ):
                continue

            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            if self.min_rectangularity > 0:
                poly, _error_msg = polygon_from_outline(contour_xy, buffer_radius=0.1)
                if poly is None:
                    continue
                bbox_area = crosswalk_length * crosswalk_width
                rectangularity = poly.area / max(bbox_area, 1e-6)
                if rectangularity < self.min_rectangularity:
                    continue

            crosswalk_z = float(np.median(cluster_pts[:, 2])) if len(cluster_pts) != 0 else 0.0
            record = self.make_record(
                logical_index,
                cluster_center,
                cluster_pts,
                contour_xy,
                record_dirc,
                record_lateral,
                long_range,
                lat_range,
                crosswalk_z,
            )
            self.records.append(record)

            corners = np.array([
                cluster_center + record_dirc * long_range[0] + record_lateral * lat_range[0],
                cluster_center + record_dirc * long_range[0] + record_lateral * lat_range[1],
                cluster_center + record_dirc * long_range[1] + record_lateral * lat_range[1],
                cluster_center + record_dirc * long_range[1] + record_lateral * lat_range[0],
            ], dtype=np.float32)
            corner_canvas = to_canvas(corners).reshape(-1, 1, 2)
            cv2.polylines(canvas, [corner_canvas], True, (255, 80, 80), 4)

            if len(contour_xy) >= 2:
                contour_canvas = to_canvas(contour_xy[:, :2]).reshape(-1, 1, 2)
                cv2.polylines(canvas, [contour_canvas], True, (0, 180, 255), 2)

            any_detected = True

        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return any_detected
