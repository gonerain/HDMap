import numpy as np
import cv2

from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import keep_largest_cluster
from core.vector_common import simplify_polyline_by_slope


class SidewalkEdgeProcess(VectorProcess):
    name = "sidewalk"
    target_class_key = "sidewalk_class"
    default_target_class = 15
    requires_pose = False
    cluster_eps = 0.8
    cluster_min_samples = 12
    outline_alpha = 1.0
    simplify_angle_thresh_deg = 9.0
    simplify_min_seg_length = 0.12
    parallel_min_seg_length = 0.5
    parallel_dir_angle_thresh_deg = 20.0
    min_lateral_gap = 0.8

    def make_side_record(self, cluster_points, contour_xy, edge_seg, sidewalk_z):
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32) if len(cluster_points) != 0 else np.zeros((2,), dtype=np.float32)
        if edge_seg is None:
            edge = None
        else:
            edge = {
                "p1": edge_seg["p1"].astype(np.float32).tolist(),
                "p2": edge_seg["p2"].astype(np.float32).tolist(),
            }
        return {
            "centroid": centroid.tolist(),
            "sidewalk_z": float(sidewalk_z),
            "outline": np.asarray(contour_xy, dtype=np.float32).tolist(),
            "inner_edge": edge,
        }

    def make_record(self, logical_index, process_ctx, dirc, left_record, right_record):
        return {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "centroid": process_ctx["current_center"].astype(np.float32).tolist(),
            "dirc": np.asarray(dirc, dtype=np.float32).tolist(),
            "left_sidewalk": left_record,
            "right_sidewalk": right_record,
        }

    def extract_sidewalk_on_side(self, side_points, center_xy, axis, side_sign):
        if len(side_points) == 0:
            return None

        cluster_points = keep_largest_cluster(
            side_points[:, :3],
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        if len(cluster_points) == 0:
            return None

        contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
        contour_xy = simplify_polyline_by_slope(
            contour_xy,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )
        if len(contour_xy) < 3:
            return None

        edge_seg = self.pick_inner_edge_segment(contour_xy, center_xy, axis, side_sign)
        sidewalk_z = float(np.median(cluster_points[:, 2])) if len(cluster_points) != 0 else 0.0
        return {
            "cluster_points": cluster_points,
            "contour_xy": contour_xy,
            "edge_seg": edge_seg,
            "sidewalk_z": sidewalk_z,
        }

    def pick_inner_edge_segment(self, contour_xy, center_xy, axis, side_sign):
        axis = np.asarray(axis, dtype=np.float32).reshape(-1)[:2]
        center_xy = np.asarray(center_xy, dtype=np.float32).reshape(-1)[:2]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6 or len(contour_xy) < 2:
            return None

        axis = axis / axis_norm
        lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
        cos_dir = float(np.cos(np.deg2rad(self.parallel_dir_angle_thresh_deg)))

        best_seg = None
        best_score = None
        for i in range(len(contour_xy) - 1):
            p1 = contour_xy[i, :2]
            p2 = contour_xy[i + 1, :2]
            vec = p2 - p1
            seg_len = float(np.linalg.norm(vec))
            if seg_len < self.parallel_min_seg_length:
                continue

            seg_dir = vec / seg_len
            align = float(np.dot(seg_dir, axis))
            if abs(align) < cos_dir:
                continue
            if align < 0.0:
                seg_dir = -seg_dir
                p1, p2 = p2, p1

            midpoint = 0.5 * (p1 + p2)
            lat = float(np.dot(midpoint - center_xy, lateral))
            if side_sign * lat <= self.min_lateral_gap:
                continue

            score = (abs(lat), -seg_len)
            if best_score is None or score < best_score:
                best_score = score
                best_seg = {
                    "p1": p1.astype(np.float32, copy=False),
                    "p2": p2.astype(np.float32, copy=False),
                    "dir": seg_dir.astype(np.float32, copy=False),
                    "length": seg_len,
                    "lat": lat,
                }
        return best_seg

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        merged_points = process_ctx["current_points"]
        dirc = process_ctx["last"]["centerpoint"] - process_ctx["front"]["centerpoint"]
        dirc_norm = float(np.linalg.norm(dirc))
        if dirc_norm < self.static_dirc_thresh:
            print(f"static skip @ {logical_index}: dirc_norm={dirc_norm:.4f}, thresh={self.static_dirc_thresh:.4f}")
            return False
        dirc = (dirc / dirc_norm).astype(np.float32)
        lateral = np.array([-dirc[1], dirc[0]], dtype=np.float32)

        rel = merged_points[:, :2] - process_ctx["current_center"][None, :]
        lat = rel @ lateral
        left_points = merged_points[lat < -self.min_lateral_gap]
        right_points = merged_points[lat > self.min_lateral_gap]
        print(f"{self.name} split: left={len(left_points)} right={len(right_points)} total={len(merged_points)}")

        left_data = self.extract_sidewalk_on_side(left_points, process_ctx["current_center"], dirc, -1.0)
        right_data = self.extract_sidewalk_on_side(right_points, process_ctx["current_center"], dirc, 1.0)
        if left_data is None and right_data is None:
            print(f"skip: no valid sidewalk contour on either side @ {logical_index}")
            return False

        all_cluster_points = []
        all_contours = []
        for side_data in (left_data, right_data):
            if side_data is None:
                continue
            all_cluster_points.append(side_data["cluster_points"])
            all_contours.append(side_data["contour_xy"])

        canvas, to_canvas = self.draw_debug_canvas(
            np.vstack(all_cluster_points) if all_cluster_points else np.zeros((0, 3), dtype=np.float32),
            np.vstack(all_contours) if all_contours else np.zeros((0, 2), dtype=np.float32),
            process_ctx,
        )

        def draw_side(side_data, contour_color, edge_color):
            if side_data is None:
                return None
            contour_canvas = to_canvas(side_data["contour_xy"][:, :2]).reshape(-1, 1, 2)
            cv2.polylines(canvas, [contour_canvas], True, contour_color, 4)
            if side_data["edge_seg"] is not None:
                edge_canvas = to_canvas(
                    np.vstack((side_data["edge_seg"]["p1"], side_data["edge_seg"]["p2"]))
                ).reshape(-1, 1, 2)
                cv2.polylines(canvas, [edge_canvas], False, edge_color, 5)
            return self.make_side_record(
                side_data["cluster_points"],
                side_data["contour_xy"],
                side_data["edge_seg"],
                side_data["sidewalk_z"],
            )

        left_record = draw_side(left_data, (255, 160, 80), (255, 80, 80))
        right_record = draw_side(right_data, (80, 180, 255), (80, 220, 80))

        self.records.append(self.make_record(logical_index, process_ctx, dirc, left_record, right_record))
        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True
