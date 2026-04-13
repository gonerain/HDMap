import json
import os

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import keep_largest_cluster


def as_xy(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim == 1:
        return arr.reshape(1, -1)[:, :2]
    return arr[:, :2].astype(np.float32, copy=False)


def polygon_from_outline(outline_xy):
    outline_xy = as_xy(outline_xy)
    if len(outline_xy) < 3:
        return None
    try:
        polygon = Polygon(outline_xy)
    except Exception:
        return None
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(polygon, Polygon):
        return None
    if polygon.area <= 1e-6:
        return None
    return polygon


def outline_from_polygon(polygon):
    if polygon is None or polygon.is_empty:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(polygon, Polygon):
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(polygon.exterior.coords, dtype=np.float32)


def close_outline(points_xy):
    points_xy = as_xy(points_xy)
    if len(points_xy) == 0:
        return points_xy
    if np.linalg.norm(points_xy[0] - points_xy[-1]) < 1e-6:
        return points_xy
    return np.vstack((points_xy, points_xy[:1]))


def chaikin_smooth_closed(points_xy, iterations=2):
    points_xy = close_outline(points_xy)
    if len(points_xy) < 4:
        return points_xy

    work = points_xy[:-1].copy()
    for _ in range(max(int(iterations), 0)):
        if len(work) < 3:
            break
        new_points = []
        for idx in range(len(work)):
            p0 = work[idx]
            p1 = work[(idx + 1) % len(work)]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.extend((q.astype(np.float32), r.astype(np.float32)))
        work = np.asarray(new_points, dtype=np.float32)
    return close_outline(work)


class SidewalkEdgeProcess(VectorProcess):
    name = "sidewalk"
    target_class_key = "sidewalk_class"
    default_target_class = 15
    requires_pose = False
    cluster_eps = 0.8
    cluster_min_samples = 12
    outline_alpha = 1.0
    merge_overlap_buffer = 0.2
    fused_outline_simplify_tol = 0.15
    fused_outline_smooth_iterations = 2

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

    def extract_main_outline(self, merged_points):
        if len(merged_points) == 0:
            return None

        cluster_points = keep_largest_cluster(
            merged_points[:, :3],
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        if len(cluster_points) < 3:
            return None

        contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
        polygon = polygon_from_outline(contour_xy)
        if polygon is None:
            return None

        sidewalk_z = float(np.median(cluster_points[:, 2])) if len(cluster_points) != 0 else 0.0
        return {
            "cluster_points": cluster_points,
            "contour_xy": outline_from_polygon(polygon),
            "polygon": polygon,
            "sidewalk_z": sidewalk_z,
        }

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        merged_points = process_ctx["current_points"]
        result = self.extract_main_outline(merged_points)
        if result is None:
            print(f"skip: no valid sidewalk outline @ {logical_index}")
            return False

        canvas, to_canvas = self.draw_debug_canvas(
            result["cluster_points"],
            result["contour_xy"],
            process_ctx,
        )
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
        polygon = polygon_from_outline(outline_xy)
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
        merged_polygon = polygon_from_outline(merged_outline)
        if merged_polygon is None:
            merged_polygon = base_polygon.union(new_polygon)
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = max(merged_polygon.geoms, key=lambda geom: geom.area)
        return merged_polygon, True

    def update_track_metadata(self, track):
        polygon = track["_polygon"]
        outline_xy = outline_from_polygon(polygon)
        if len(outline_xy) >= 4:
            simplified = polygon.simplify(self.fused_outline_simplify_tol, preserve_topology=True)
            simplified_polygon = polygon_from_outline(outline_from_polygon(simplified))
            if simplified_polygon is not None:
                polygon = simplified_polygon
                outline_xy = outline_from_polygon(polygon)

            smoothed_outline = chaikin_smooth_closed(
                outline_xy,
                iterations=self.fused_outline_smooth_iterations,
            )
            smoothed_polygon = polygon_from_outline(smoothed_outline)
            if smoothed_polygon is not None:
                polygon = smoothed_polygon
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
            polygon = polygon_from_outline(outline_xy)
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
