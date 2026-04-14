import json
import os

import cv2
import numpy as np

from core.vector_common import VectorProcess
from core.vector_common import build_canvas_transform
from core.vector_common import cluster_labels
from core.vector_common import default_demo_output_dir
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import find_parallel_segments_around_center
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
    edge_z_max_line_dist = 0.6
    edge_z_endpoint_margin = 0.8
    edge_z_quantile = 0.3
    edge_z_min_samples = 12

    def __init__(self, args, config):
        super().__init__(args, config)
        self.step = max(int(config.get("road_step", config.get("vector_step", self.step))), 1)

        self.cluster_eps = float(config.get("road_cluster_eps", self.cluster_eps))
        self.cluster_min_samples = int(config.get("road_cluster_min_samples", self.cluster_min_samples))
        self.outline_alpha = float(config.get("road_outline_alpha", self.outline_alpha))

        self.max_clusters = max(int(config.get("road_max_clusters", 3)), 1)
        self.min_cluster_points = max(int(config.get("road_min_cluster_points", self.cluster_min_samples)), 3)
        self.max_cluster_center_dist = float(config.get("road_max_cluster_center_dist", 60.0))

        # Performance toggles: disable heavy stage image generation by default.
        self.stage_debug = bool(config.get("road_stage_debug", False))
        self.save_final_debug = bool(config.get("road_save_final_debug", True))
        self.save_origin_debug = bool(config.get("road_save_origin_debug", False))

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
        if not self.stage_debug:
            return
        stage_dir = self._debug_stage_dir(stage_name)
        cv2.imwrite(os.path.join(stage_dir, f"{self.name}_{logical_index:06d}.png"), canvas)

    def _estimate_segment_z(self, points_xyz, seg, fallback_z):
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        if len(points_xyz) == 0:
            return float(fallback_z)

        p1 = np.asarray(seg["p1"], dtype=np.float32)
        p2 = np.asarray(seg["p2"], dtype=np.float32)
        seg_vec = p2 - p1
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-6:
            return float(fallback_z)

        seg_dir = seg_vec / seg_len
        rel = points_xyz[:, :2] - p1[None, :]
        proj = rel @ seg_dir
        perp = rel - proj[:, None] * seg_dir[None, :]
        line_dist = np.linalg.norm(perp, axis=1)

        valid = proj >= -float(self.edge_z_endpoint_margin)
        valid &= proj <= seg_len + float(self.edge_z_endpoint_margin)
        valid &= line_dist <= float(self.edge_z_max_line_dist)

        z_values = points_xyz[valid, 2]
        if len(z_values) < self.edge_z_min_samples:
            midpoint = 0.5 * (p1 + p2)
            midpoint_dist = np.linalg.norm(points_xyz[:, :2] - midpoint[None, :], axis=1)
            nearest_count = min(max(self.edge_z_min_samples, 1), len(points_xyz))
            nearest_idx = np.argpartition(midpoint_dist, nearest_count - 1)[:nearest_count]
            z_values = points_xyz[nearest_idx, 2]

        if len(z_values) == 0:
            return float(fallback_z)
        return float(np.quantile(z_values, self.edge_z_quantile))

    def _select_road_points(self, current_points_xyz, center_xy):
        points = np.asarray(current_points_xyz, dtype=np.float32)
        if len(points) == 0:
            return points

        labels = cluster_labels(points[:, :2], eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        valid_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
        if not valid_labels:
            return points

        candidates = []
        for label in valid_labels:
            cluster = points[labels == label, :3]
            if len(cluster) < self.min_cluster_points:
                continue
            centroid = cluster[:, :2].mean(axis=0)
            center_dist = float(np.linalg.norm(centroid - center_xy))
            if center_dist > self.max_cluster_center_dist:
                continue
            score = float(len(cluster)) - 0.35 * center_dist
            candidates.append((score, cluster))

        if not candidates:
            return points

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [cluster for _, cluster in candidates[: self.max_clusters]]
        return np.vstack(selected) if selected else points

    def make_record(self, logical_index, process_ctx, left_seg, right_seg, dirc, road_z, left_z, right_z):
        center_z = 0.5 * (float(left_z) + float(right_z))
        return {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "centroid": process_ctx["current_center"].astype(np.float32).tolist(),
            "dirc": np.asarray(dirc, dtype=np.float32).tolist(),
            "road_z": float(road_z),
            "center_z": float(center_z),
            "left_edge": {
                "p1": left_seg["p1"].astype(np.float32).tolist(),
                "p2": left_seg["p2"].astype(np.float32).tolist(),
                "z": float(left_z),
            },
            "right_edge": {
                "p1": right_seg["p1"].astype(np.float32).tolist(),
                "p2": right_seg["p2"].astype(np.float32).tolist(),
                "z": float(right_z),
            },
        }

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        selected_points = self._select_road_points(process_ctx["current_points"][:, :3], process_ctx["current_center"])
        print(f"{self.name} points: {len(selected_points)} / {len(process_ctx['current_points'])}")
        if len(selected_points) < self.cluster_min_samples:
            return False

        if self.stage_debug:
            raw_points_canvas = self._draw_stage_canvas(process_ctx, points_xyz=process_ctx["current_points"][:, :3])
            self._save_stage_canvas(logical_index, "road_raw_points", raw_points_canvas)

        alpha_contour_xy = extract_outline_by_alphashape(selected_points, alpha=self.outline_alpha)
        if self.stage_debug:
            cluster_alpha_canvas = self._draw_stage_canvas(
                process_ctx,
                points_xyz=selected_points,
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
        if dirc_norm < self.static_dirc_thresh:
            if self.save_final_debug:
                final_canvas = self._draw_stage_canvas(
                    process_ctx,
                    points_xyz=selected_points,
                    polylines=[(contour_xy, (255, 160, 80), True, 4)],
                )
                self.save_debug_canvas(logical_index, final_canvas)
            if self.save_origin_debug:
                self.save_origin_debug_image(runtime, logical_index, logical_index)
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
            if self.save_final_debug:
                final_canvas = self._draw_stage_canvas(
                    process_ctx,
                    points_xyz=selected_points,
                    polylines=[(contour_xy, (255, 160, 80), True, 4)],
                )
                self.save_debug_canvas(logical_index, final_canvas)
            if self.save_origin_debug:
                self.save_origin_debug_image(runtime, logical_index, logical_index)
            print(f"skip: no parallel contour segments for {self.name} around centerpoint")
            return False

        left_seg, right_seg = seg_pair
        road_z = float(np.median(process_ctx["current_points"][:, 2])) if len(process_ctx["current_points"]) != 0 else 0.0
        left_z = self._estimate_segment_z(selected_points, left_seg, road_z)
        right_z = self._estimate_segment_z(selected_points, right_seg, road_z)
        self.records.append(self.make_record(logical_index, process_ctx, left_seg, right_seg, dirc / dirc_norm, road_z, left_z, right_z))

        if self.save_final_debug:
            final_canvas = self._draw_stage_canvas(
                process_ctx,
                points_xyz=selected_points,
                polylines=[(contour_xy, (255, 160, 80), True, 4)],
                segments=[
                    (left_seg["p1"], left_seg["p2"], (255, 80, 80), 5),
                    (right_seg["p1"], right_seg["p2"], (80, 220, 80), 5),
                ],
            )
            self._save_stage_canvas(logical_index, "road_simplified_edges", final_canvas)
            self.save_debug_canvas(logical_index, final_canvas)

        if self.save_origin_debug:
            self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True

    def _build_topology(self, fused):
        samples = fused.get("samples", [])
        if not samples:
            return {
                "nodes": [],
                "edges": [],
                "components": [],
                "boundary_pairs": [],
            }

        gap_limit = max(int(self.config.get("road_topology_max_index_gap", self.step * 3)), 1)
        geom_dist_limit = float(self.config.get("road_topology_geom_dist_max", 8.0))
        geom_heading_cos_min = float(self.config.get("road_topology_heading_cos_min", 0.55))
        nodes = []
        edges = []
        boundary_pairs = []

        for i, sample in enumerate(samples):
            center = np.asarray(sample.get("center", [0.0, 0.0]), dtype=np.float32)
            left = np.asarray(sample.get("left_edge", [0.0, 0.0, 0.0]), dtype=np.float32)
            right = np.asarray(sample.get("right_edge", [0.0, 0.0, 0.0]), dtype=np.float32)
            node_id = f"n{i:05d}"
            nodes.append(
                {
                    "id": node_id,
                    "sample_index": int(sample.get("index", -1)),
                    "center": [float(center[0]), float(center[1])],
                    "center_z": float(sample.get("center_z", 0.0)),
                    "road_z": float(sample.get("road_z", 0.0)),
                }
            )
            boundary_pairs.append(
                {
                    "node_id": node_id,
                    "left_edge": [float(left[0]), float(left[1]), float(left[2])],
                    "right_edge": [float(right[0]), float(right[1]), float(right[2])],
                }
            )

        comp_id = 0
        comp_ranges = []
        comp_start = 0
        for i in range(len(nodes) - 1):
            curr = nodes[i]
            nxt = nodes[i + 1]
            idx_gap = int(nxt["sample_index"] - curr["sample_index"])
            p1 = np.asarray(curr["center"], dtype=np.float32)
            p2 = np.asarray(nxt["center"], dtype=np.float32)
            seg = p2 - p1
            seg_len = float(np.linalg.norm(seg))
            heading = seg / seg_len if seg_len > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

            curr_lr = np.asarray(boundary_pairs[i]["left_edge"][:2], dtype=np.float32) - np.asarray(boundary_pairs[i]["right_edge"][:2], dtype=np.float32)
            nxt_lr = np.asarray(boundary_pairs[i + 1]["left_edge"][:2], dtype=np.float32) - np.asarray(boundary_pairs[i + 1]["right_edge"][:2], dtype=np.float32)
            curr_lr_n = curr_lr / max(float(np.linalg.norm(curr_lr)), 1e-6)
            nxt_lr_n = nxt_lr / max(float(np.linalg.norm(nxt_lr)), 1e-6)
            heading_cos = float(np.abs(np.dot(curr_lr_n, nxt_lr_n)))

            geom_connected = (seg_len <= geom_dist_limit) and (heading_cos >= geom_heading_cos_min)
            temporal_connected = idx_gap <= gap_limit
            connected = bool(geom_connected)

            edges.append(
                {
                    "from": curr["id"],
                    "to": nxt["id"],
                    "index_gap": idx_gap,
                    "length": seg_len,
                    "heading": [float(heading[0]), float(heading[1])],
                    "heading_cos": heading_cos,
                    "connected": connected,
                    "geom_connected": bool(geom_connected),
                    "temporal_connected": bool(temporal_connected),
                    "component_id": int(comp_id),
                }
            )
            if not connected:
                comp_ranges.append({"component_id": int(comp_id), "start_node": nodes[comp_start]["id"], "end_node": curr["id"]})
                comp_id += 1
                comp_start = i + 1

        comp_ranges.append({"component_id": int(comp_id), "start_node": nodes[comp_start]["id"], "end_node": nodes[-1]["id"]})

        return {
            "nodes": nodes,
            "edges": edges,
            "components": comp_ranges,
            "boundary_pairs": boundary_pairs,
            "params": {
                "max_index_gap": gap_limit,
                "geom_dist_max": geom_dist_limit,
                "heading_cos_min": geom_heading_cos_min,
            },
        }

    def finalize(self):
        super().finalize()

        try:
            import sys
            from pathlib import Path

            ROOT = Path(__file__).resolve().parents[1]
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.vectorize.fuse_road_edges import build_output
            from src.vectorize.fuse_road_edges import save_preview

            class Args:
                def __init__(self, params, input_path):
                    self.input = str(input_path)
                    self.method = params["method"]
                    self.width_min = params["width_min"]
                    self.width_max = params["width_max"]
                    self.width_dev = params["width_dev"]
                    self.max_backtrack = params["max_backtrack"]
                    self.default_width = params["default_width"]
                    self.centroid_thresh = params["centroid_thresh"]
                    self.dir_window = params["dir_window"]
                    self.center_window = params["center_window"]
                    self.edge_window = params["edge_window"]
                    self.width_window = params["width_window"]
                    self.ls_degree = params["ls_degree"]
                    self.preview = params["preview"]

            raw_path = Path(self.output_path())
            args = Args(self.fusion_params, raw_path)
            fused, debug = build_output(self.records, args)
            fused["topology"] = self._build_topology(fused)

            fused_path = os.path.splitext(str(raw_path))[0] + "_fused.json"
            output_dir = os.path.dirname(fused_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(fused_path, "w") as f:
                json.dump(fused, f, indent=2)
            print(f"saved fused road records to {fused_path}")

            if self.fusion_params["preview"]:
                save_preview(Path(fused_path), self.records, fused, debug)

        except Exception as e:
            import traceback

            print(f"Failed to fuse road records: {e}")
            traceback.print_exc()
            raise RuntimeError("road fusion failed during finalize()") from e
