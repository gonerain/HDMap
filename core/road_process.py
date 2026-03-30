import numpy as np
import cv2

from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import find_parallel_segments_around_center
from core.vector_common import keep_largest_cluster
from core.vector_common import simplify_polyline_by_slope


class RoadEdgeProcess(VectorProcess):
    name = "road"
    target_class_key = "road_class"
    default_target_class = 13
    cluster_eps = 1.0
    cluster_min_samples = 20
    outline_alpha = 0.8
    simplify_angle_thresh_deg = 7.0
    simplify_min_seg_length = 0.15
    parallel_min_seg_length = 0.5
    parallel_dir_angle_thresh_deg = 15.0
    parallel_pair_angle_thresh_deg = 10.0
    parallel_min_lateral_gap = 0.5

    def make_record(self, logical_index, process_ctx, left_seg, right_seg, dirc, road_z):
        return {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "centroid": process_ctx["current_center"].astype(np.float32).tolist(),
            "dirc": np.asarray(dirc, dtype=np.float32).tolist(),
            "road_z": float(road_z),
            "left_edge": {
                "p1": left_seg["p1"].astype(np.float32).tolist(),
                "p2": left_seg["p2"].astype(np.float32).tolist(),
            },
            "right_edge": {
                "p1": right_seg["p1"].astype(np.float32).tolist(),
                "p2": right_seg["p2"].astype(np.float32).tolist(),
            },
        }

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        cluster_points = keep_largest_cluster(
            process_ctx["current_points"][:, :3],
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        print(f"{self.name} cluster: {len(cluster_points)} / {len(process_ctx['current_points'])}")

        contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
        contour_xy = simplify_polyline_by_slope(
            contour_xy,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )

        dirc = process_ctx["last"]["centerpoint"] - process_ctx["front"]["centerpoint"]
        dirc_norm = float(np.linalg.norm(dirc))
        canvas, to_canvas = self.draw_debug_canvas(cluster_points, contour_xy, process_ctx)
        if dirc_norm < self.static_dirc_thresh:
            self.save_debug_canvas(logical_index, canvas)
            self.save_origin_debug_image(runtime, logical_index, logical_index)
            print(f"static skip @ {logical_index}: dirc_norm={dirc_norm:.4f}, thresh={self.static_dirc_thresh:.4f}")
            return False

        seg_pair = find_parallel_segments_around_center(
            contour_xy,
            process_ctx["current_center"],
            dirc / dirc_norm,
            min_seg_length=self.parallel_min_seg_length,
            dir_angle_thresh_deg=self.parallel_dir_angle_thresh_deg,
            pair_angle_thresh_deg=self.parallel_pair_angle_thresh_deg,
            min_lateral_gap=self.parallel_min_lateral_gap,
        )
        if len(seg_pair) != 2:
            print(f"skip: no parallel contour segments for {self.name} around centerpoint")
            return False

        left_seg, right_seg = seg_pair
        left_canvas = to_canvas(np.vstack((left_seg["p1"], left_seg["p2"]))).reshape(-1, 1, 2)
        right_canvas = to_canvas(np.vstack((right_seg["p1"], right_seg["p2"]))).reshape(-1, 1, 2)
        cv2.polylines(canvas, [left_canvas], False, (255, 80, 80), 5)
        cv2.polylines(canvas, [right_canvas], False, (80, 220, 80), 5)

        road_z = float(np.median(process_ctx["current_points"][:, 2])) if len(process_ctx["current_points"]) != 0 else 0.0
        self.records.append(self.make_record(logical_index, process_ctx, left_seg, right_seg, dirc / dirc_norm, road_z))
        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True
