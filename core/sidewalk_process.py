import json
import os
import warnings

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from core.vector_common import VectorProcess
from core.vector_common import cluster_labels
from core.vector_common import extract_outline_by_alphashape
from core.geometry_utils import (
    as_xy,
    close_outline,
    polygon_from_outline,
    outline_from_polygon,
    chaikin_smooth_closed,
    prune_short_edges_closed,
    validate_contour_parameters,
)


class SidewalkEdgeProcess(VectorProcess):
    name = "sidewalk"
    target_class_key = "sidewalk_class"
    default_target_class = 15
    requires_pose = False
    cluster_eps = 0.8
    cluster_min_samples = 12
    outline_alpha = 1.0
    merge_overlap_buffer = 0.2
    fused_outline_simplify_tol = 0.25
    fused_outline_smooth_iterations = 1
    fused_outline_min_seg_len = 0.35
    fused_outline_post_simplify_tol = 0.12
    fused_min_area = 2.0
    fused_min_support_frames = 3

    def __init__(self, args, config):
        super().__init__(args, config)
        self.step = max(int(config.get("sidewalk_step", config.get("vector_step", self.step))), 1)
        self.cluster_eps = float(config.get("sidewalk_cluster_eps", self.cluster_eps))
        self.cluster_min_samples = int(config.get("sidewalk_cluster_min_samples", self.cluster_min_samples))
        self.outline_alpha = float(config.get("sidewalk_outline_alpha", self.outline_alpha))
        self.max_candidates_per_frame = max(int(config.get("sidewalk_max_candidates_per_frame", 3)), 1)
        self.min_cluster_area = float(config.get("sidewalk_cluster_min_area", 1.2))
        self.min_cluster_density = float(config.get("sidewalk_cluster_min_density", 0.35))
        self.min_compactness = float(config.get("sidewalk_cluster_min_compactness", 0.004))
        self.max_center_distance = float(config.get("sidewalk_cluster_max_center_distance", 45.0))

    def make_record(self, logical_index, process_ctx, cluster_points, contour_xy, sidewalk_z):
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32) if len(cluster_points) != 0 else np.zeros((2,), dtype=np.float32)
        return {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "window_centroid": process_ctx["current_center"].astype(np.float32).tolist(),
            "centroid": centroid.tolist(),
            "sidewalk_z": float(sidewalk_z),
            "outline": as_xy(contour_xy).tolist(),
        }

    def _cluster_score(self, cluster_points, polygon, process_center_xy):
        area = float(polygon.area)
        perimeter = max(float(polygon.length), 1e-6)
        density = float(len(cluster_points)) / max(area, 1e-6)
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter))
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32)
        center_dist = float(np.linalg.norm(centroid - process_center_xy))

        if area < self.min_cluster_area:
            return None
        if density < self.min_cluster_density:
            return None
        if compactness < self.min_compactness:
            return None
        if center_dist > self.max_center_distance:
            return None

        score = (
            1.0 * float(len(cluster_points))
            + 6.0 * np.sqrt(max(area, 0.0))
            + 3.0 * density
            - 0.4 * center_dist
        )
        return {
            "score": float(score),
            "centroid": centroid,
        }

    def extract_candidate_outlines(self, merged_points, process_center_xy):
        if len(merged_points) == 0:
            return []

        labels = cluster_labels(merged_points[:, :2], eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        valid_labels = [label for label in np.unique(labels) if label >= 0]
        if not valid_labels and len(merged_points) >= self.cluster_min_samples:
            valid_labels = [0]
            labels = np.zeros((len(merged_points),), dtype=np.int32)

        candidates = []
        for label in valid_labels:
            cluster_points = merged_points[labels == label, :3]
            if len(cluster_points) < 3:
                continue

            contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
            polygon, error_msg = polygon_from_outline(contour_xy)
            if polygon is None:
                if error_msg:
                    print(f"sidewalk: polygon_from_outline failed: {error_msg}")
                continue

            scored = self._cluster_score(cluster_points, polygon, process_center_xy)
            if scored is None:
                continue

            sidewalk_z = float(np.median(cluster_points[:, 2])) if len(cluster_points) != 0 else 0.0
            candidates.append(
                {
                    "cluster_points": cluster_points,
                    "contour_xy": outline_from_polygon(polygon),
                    "polygon": polygon,
                    "sidewalk_z": sidewalk_z,
                    "score": scored["score"],
                }
            )

        if not candidates:
            return []
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: self.max_candidates_per_frame]

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        merged_points = process_ctx["current_points"]
        results = self.extract_candidate_outlines(merged_points, process_ctx["current_center"])
        if not results:
            print(f"skip: no valid sidewalk outline @ {logical_index}")
            return False

        if len(results) == 1:
            draw_points = results[0]["cluster_points"]
            draw_contour = results[0]["contour_xy"]
        else:
            draw_points = np.vstack([res["cluster_points"] for res in results])
            contour_parts = [res["contour_xy"] for res in results if len(res["contour_xy"]) != 0]
            draw_contour = np.vstack(contour_parts) if contour_parts else np.zeros((0, 2), dtype=np.float32)

        canvas, to_canvas = self.draw_debug_canvas(draw_points, draw_contour, process_ctx)
        for result in results:
            contour_canvas = to_canvas(result["contour_xy"][:, :2]).reshape(-1, 1, 2)
            cv2.polylines(canvas, [contour_canvas], True, (0, 180, 255), 5)
            self.records.append(
                self.make_record(
                    logical_index,
                    process_ctx,
                    result["cluster_points"],
                    result["contour_xy"],
                    result["sidewalk_z"],
                )
            )

        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True

    def init_track(self, record, track_id):
        outline_xy = as_xy(record["outline"])
        polygon, error_msg = polygon_from_outline(outline_xy)
        if polygon is None and error_msg:
            warnings.warn(f"init_track: polygon_from_outline failed: {error_msg}")
        centroid = as_xy(record["centroid"])
        return {
            "id": int(track_id),
            "centroid": centroid[0].tolist() if len(centroid) == 1 else [0.0, 0.0],
            "sidewalk_z": float(record["sidewalk_z"]),
            "area": float(polygon.area) if polygon is not None else 0.0,
            "outline": outline_xy.tolist(),
            "source_indices": [int(record["index"])],
            "outlines": [outline_xy.tolist()],
            "_polygon": polygon,
            "_outline_arrays": [outline_xy],
            "_z_values": [float(record["sidewalk_z"])],
        }

    def merge_track_outline(self, base_polygon, new_polygon, outline_arrays):
        if base_polygon is None or new_polygon is None:
            return base_polygon, False

        base_hit = base_polygon.buffer(self.merge_overlap_buffer)
        new_hit = new_polygon.buffer(self.merge_overlap_buffer)
        if base_hit.intersection(new_hit).is_empty:
            return base_polygon, False

        merged_points = []
        for arr in outline_arrays:
            arr = as_xy(arr)
            if len(arr) != 0:
                merged_points.append(arr)
        if not merged_points:
            merged = base_polygon.union(new_polygon)
            return merged, True

        merged_xy = np.vstack(merged_points)
        merged_outline = extract_outline_by_alphashape(
            np.column_stack((merged_xy, np.zeros((len(merged_xy),), dtype=np.float32))),
            alpha=self.outline_alpha,
        )
        merged_polygon, error_msg = polygon_from_outline(merged_outline)
        if merged_polygon is None:
            if error_msg:
                print(f"merge_track_outline: polygon_from_outline failed: {error_msg}")
            merged_polygon = base_polygon.union(new_polygon)
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = max(merged_polygon.geoms, key=lambda geom: geom.area)
        return merged_polygon, True

    def update_track_metadata(self, track):
        polygon = track["_polygon"]
        outline_xy = outline_from_polygon(polygon)
        if len(outline_xy) >= 4:
            simplify_tol = float(self.config.get("sidewalk_simplify_tol", self.fused_outline_simplify_tol))
            smooth_iterations = int(self.config.get("sidewalk_smooth_iterations", self.fused_outline_smooth_iterations))
            min_seg_len = float(self.config.get("sidewalk_min_seg_len", self.fused_outline_min_seg_len))
            post_simplify_tol = float(self.config.get("sidewalk_post_simplify_tol", self.fused_outline_post_simplify_tol))

            simplified = polygon.simplify(simplify_tol, preserve_topology=True)
            simplified_polygon, error_msg1 = polygon_from_outline(outline_from_polygon(simplified))
            if simplified_polygon is not None:
                polygon = simplified_polygon
                outline_xy = outline_from_polygon(polygon)
            elif error_msg1:
                print(f"update_track_metadata: simplified polygon_from_outline failed: {error_msg1}")

            smoothed_outline = chaikin_smooth_closed(
                outline_xy,
                iterations=smooth_iterations,
            )
            smoothed_outline = prune_short_edges_closed(smoothed_outline, min_seg_len=min_seg_len)
            smoothed_polygon, _ = polygon_from_outline(smoothed_outline)
            if smoothed_polygon is not None:
                polygon = smoothed_polygon
                outline_xy = outline_from_polygon(polygon)

            hardened = polygon.simplify(post_simplify_tol, preserve_topology=True)
            hardened_polygon, _ = polygon_from_outline(outline_from_polygon(hardened))
            if hardened_polygon is not None:
                polygon = hardened_polygon
                outline_xy = outline_from_polygon(polygon)

        track["_polygon"] = polygon
        track["outline"] = outline_xy.tolist()
        track["area"] = float(polygon.area) if polygon is not None else 0.0

        points = [as_xy(arr) for arr in track["_outline_arrays"] if len(as_xy(arr)) != 0]
        if points:
            merged_points = np.vstack(points)
            track["centroid"] = merged_points.mean(axis=0).astype(np.float32).tolist()
        else:
            track["centroid"] = [0.0, 0.0]

        if track["_z_values"]:
            track["sidewalk_z"] = float(np.mean(track["_z_values"]))

    def fuse_records(self):
        track_id = 0
        tracks = []
        for record in self.records:
            outline_xy = as_xy(record.get("outline", []))
            polygon, error_msg = polygon_from_outline(outline_xy)
            if polygon is None:
                continue

            matched_track = None
            merged_polygon = None
            for track in reversed(tracks):
                candidate_outline_arrays = track["_outline_arrays"] + [outline_xy]
                candidate_polygon, merged = self.merge_track_outline(track["_polygon"], polygon, candidate_outline_arrays)
                if merged:
                    matched_track = track
                    merged_polygon = candidate_polygon
                    break

            if matched_track is None:
                tracks.append(self.init_track(record, track_id))
                track_id += 1
                continue

            matched_track["_polygon"] = merged_polygon
            matched_track["_outline_arrays"].append(outline_xy)
            matched_track["_z_values"].append(float(record["sidewalk_z"]))
            matched_track["source_indices"].append(int(record["index"]))
            matched_track["outlines"].append(outline_xy.tolist())
            self.update_track_metadata(matched_track)

        for track in tracks:
            self.update_track_metadata(track)
            for key in ("_polygon", "_outline_arrays", "_z_values"):
                track.pop(key, None)

        min_area = float(self.config.get("sidewalk_min_area", self.fused_min_area))
        min_support_frames = int(self.config.get("sidewalk_min_support_frames", self.fused_min_support_frames))
        min_support_span = int(self.config.get("sidewalk_min_support_span", 0))
        filtered_tracks = []
        for track in tracks:
            source_indices = sorted(set(int(v) for v in track.get("source_indices", [])))
            support_frames = len(source_indices)
            support_span = (source_indices[-1] - source_indices[0]) if len(source_indices) >= 2 else 0
            area = float(track.get("area", 0.0))
            if support_frames < min_support_frames:
                continue
            if support_span < min_support_span:
                continue
            if area < min_area:
                continue
            track["source_indices"] = source_indices
            filtered_tracks.append(track)

        tracks = filtered_tracks
        tracks.sort(key=lambda item: len(item["source_indices"]), reverse=True)
        return tracks

    def finalize(self):
        super().finalize()

        fused = {
            "meta": {
                "input": self.output_path(),
                "total_records": len(self.records),
            },
            "sidewalks": self.fuse_records(),
        }

        raw_path = self.output_path()
        fused_path = os.path.splitext(raw_path)[0] + "_fused.json"
        output_dir = os.path.dirname(fused_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(fused_path, "w") as f:
            json.dump(fused, f, indent=2)
        print(f"saved fused sidewalk records to {fused_path} (sidewalks={len(fused['sidewalks'])})")
