import os
import numpy as np
import cv2

from core.vector_common import default_demo_output_dir
from core.vector_common import VectorProcess
from core.vector_common import build_canvas_transform
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

    def _debug_stage_dir(self, stage_name):
        debug_root = self.config.get("debug_output_dir", default_demo_output_dir(self.name))
        stage_dir = os.path.join(debug_root, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        return stage_dir

    def _draw_stage_canvas(self, process_ctx, points_xyz=None, polylines=None, segments=None):
        points_xyz = np.asarray(points_xyz if points_xyz is not None else np.zeros((0, 3), dtype=np.float32), dtype=np.float32)
        polylines = polylines or []
        segments = segments or []

        all_xy_parts = [
            process_ctx["current_center"][None, :],
            process_ctx["front"]["centerpoint"][None, :],
            process_ctx["last"]["centerpoint"][None, :],
        ]
        if len(points_xyz) != 0:
            all_xy_parts.append(points_xyz[:, :2])
        for polyline, _color, _closed, _thickness in polylines:
            polyline = np.asarray(polyline, dtype=np.float32)
            if len(polyline) != 0:
                all_xy_parts.append(polyline[:, :2])
        for p1, p2, _color, _thickness in segments:
            all_xy_parts.append(np.vstack((p1[:2], p2[:2])).astype(np.float32))

        to_canvas, canvas = build_canvas_transform(np.vstack(all_xy_parts))
        if len(points_xyz) != 0:
            for px, py in to_canvas(points_xyz[:, :2]):
                cv2.circle(canvas, (int(px), int(py)), 2, (60, 60, 60), -1)

        for polyline, color, is_closed, thickness in polylines:
            polyline = np.asarray(polyline, dtype=np.float32)
            if len(polyline) < 2:
                continue
            polyline_canvas = to_canvas(polyline[:, :2]).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polyline_canvas], is_closed, color, thickness)

        for p1, p2, color, thickness in segments:
            seg_canvas = to_canvas(np.vstack((p1[:2], p2[:2]))).reshape(-1, 1, 2)
            cv2.polylines(canvas, [seg_canvas], False, color, thickness)

        current_pt = tuple(to_canvas(process_ctx["current_center"][None, :])[0])
        front_pt = tuple(to_canvas(process_ctx["front"]["centerpoint"][None, :])[0])
        last_pt = tuple(to_canvas(process_ctx["last"]["centerpoint"][None, :])[0])
        cv2.circle(canvas, current_pt, 8, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, current_pt, front_pt, (0, 170, 255), 4, tipLength=0.12)
        cv2.arrowedLine(canvas, current_pt, last_pt, (0, 200, 0), 4, tipLength=0.12)
        cv2.putText(canvas, "current centroid", (current_pt[0] + 12, current_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(canvas, "front", (front_pt[0] + 12, front_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 220), 2)
        cv2.putText(canvas, "last", (last_pt[0] + 12, last_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
        return canvas

    def _save_stage_canvas(self, logical_index, stage_name, canvas):
        stage_dir = self._debug_stage_dir(stage_name)
        cv2.imwrite(os.path.join(stage_dir, f"{self.name}_{logical_index:06d}.png"), canvas)

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

        raw_points_canvas = self._draw_stage_canvas(
            process_ctx,
            points_xyz=process_ctx["current_points"][:, :3],
        )
        self._save_stage_canvas(logical_index, "road_raw_points", raw_points_canvas)

        alpha_contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
        cluster_alpha_canvas = self._draw_stage_canvas(
            process_ctx,
            points_xyz=cluster_points,
            polylines=[(alpha_contour_xy, (0, 80, 255), True, 3)],
        )
        self._save_stage_canvas(logical_index, "road_cluster_alpha", cluster_alpha_canvas)

        contour_xy = simplify_polyline_by_slope(
            alpha_contour_xy,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )

        dirc = process_ctx["last"]["centerpoint"] - process_ctx["front"]["centerpoint"]
        dirc_norm = float(np.linalg.norm(dirc))
        final_canvas = self._draw_stage_canvas(
            process_ctx,
            points_xyz=cluster_points,
            polylines=[(contour_xy, (255, 160, 80), True, 4)],
        )
        if dirc_norm < self.static_dirc_thresh:
            self._save_stage_canvas(logical_index, "road_simplified_edges", final_canvas)
            self.save_debug_canvas(logical_index, final_canvas)
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
            self._save_stage_canvas(logical_index, "road_simplified_edges", final_canvas)
            self.save_debug_canvas(logical_index, final_canvas)
            self.save_origin_debug_image(runtime, logical_index, logical_index)
            print(f"skip: no parallel contour segments for {self.name} around centerpoint")
            return False

        left_seg, right_seg = seg_pair
        final_canvas = self._draw_stage_canvas(
            process_ctx,
            points_xyz=cluster_points,
            polylines=[(contour_xy, (255, 160, 80), True, 4)],
            segments=[
                (left_seg["p1"], left_seg["p2"], (255, 80, 80), 5),
                (right_seg["p1"], right_seg["p2"], (80, 220, 80), 5),
            ],
        )

        road_z = float(np.median(process_ctx["current_points"][:, 2])) if len(process_ctx["current_points"]) != 0 else 0.0
        self.records.append(self.make_record(logical_index, process_ctx, left_seg, right_seg, dirc / dirc_norm, road_z))
        self._save_stage_canvas(logical_index, "road_simplified_edges", final_canvas)
        self.save_debug_canvas(logical_index, final_canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True
