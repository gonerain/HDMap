import cv2
import numpy as np

from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import keep_largest_cluster
from core.vector_common import simplify_polyline_by_slope


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
    min_crosswalk_width = 0.4

    def output_path(self):
        if self.args.output:
            return self.args.output
        return self.config.get("crosswalk_vector_output", "crosswalk_records.json")

    def make_record(self, logical_index, process_ctx, cluster_points, contour_xy, dirc, lateral, long_range, lat_range, crosswalk_z):
        center_xy = process_ctx["current_center"].astype(np.float32)
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

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        merged_points = process_ctx["current_points"]
        cluster_points = keep_largest_cluster(
            merged_points[:, :3],
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        print(f"{self.name} cluster: {len(cluster_points)} / {len(merged_points)}")
        if len(cluster_points) == 0:
            return False

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

        dirc = (dirc / dirc_norm).astype(np.float32)
        lateral = np.array([-dirc[1], dirc[0]], dtype=np.float32)
        rel = cluster_points[:, :2] - process_ctx["current_center"][None, :]
        longitudinal = rel @ dirc
        lateral_proj = rel @ lateral
        long_range = np.array([np.min(longitudinal), np.max(longitudinal)], dtype=np.float32)
        lat_range = np.array([np.min(lateral_proj), np.max(lateral_proj)], dtype=np.float32)

        crosswalk_length = float(long_range[1] - long_range[0])
        crosswalk_width = float(lat_range[1] - lat_range[0])
        if crosswalk_length < self.min_crosswalk_length or crosswalk_width < self.min_crosswalk_width:
            self.save_debug_canvas(logical_index, canvas)
            self.save_origin_debug_image(runtime, logical_index, logical_index)
            print(
                f"skip: crosswalk bbox too small @ {logical_index}, "
                f"length={crosswalk_length:.3f}, width={crosswalk_width:.3f}"
            )
            return False

        corners = np.array([
            process_ctx["current_center"] + dirc * long_range[0] + lateral * lat_range[0],
            process_ctx["current_center"] + dirc * long_range[0] + lateral * lat_range[1],
            process_ctx["current_center"] + dirc * long_range[1] + lateral * lat_range[1],
            process_ctx["current_center"] + dirc * long_range[1] + lateral * lat_range[0],
        ], dtype=np.float32)
        corner_canvas = to_canvas(corners).reshape(-1, 1, 2)
        cv2.polylines(canvas, [corner_canvas], True, (255, 80, 80), 4)

        crosswalk_z = float(np.median(cluster_points[:, 2])) if len(cluster_points) != 0 else 0.0
        self.records.append(
            self.make_record(
                logical_index,
                process_ctx,
                cluster_points,
                contour_xy,
                dirc,
                lateral,
                long_range,
                lat_range,
                crosswalk_z,
            )
        )
        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True
