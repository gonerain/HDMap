import json
import os

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from core.vector_common import default_demo_output_dir
from core.vector_common import VectorProcess
from core.vector_common import extract_outline_by_alphashape
from core.vector_common import keep_largest_cluster
from core.vector_common import simplify_polyline_by_slope


def as_xy(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim == 1:
        return arr.reshape(1, -1)[:, :2]
    return arr[:, :2].astype(np.float32, copy=False)


def line_length(polyline_xy):
    polyline_xy = as_xy(polyline_xy)
    if len(polyline_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1).sum())


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if len(vec) < 2:
        return None
    vec = vec[:2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return (vec / norm).astype(np.float32)


def polyline_to_linestring(polyline_xy):
    polyline_xy = as_xy(polyline_xy)
    if len(polyline_xy) < 2:
        return None
    return LineString(polyline_xy.tolist())


def polygon_area_from_outline(outline_xy):
    outline_xy = as_xy(outline_xy)
    if len(outline_xy) < 3:
        return 0.0
    try:
        poly = Polygon(outline_xy)
        return float(poly.area) if poly.is_valid else 0.0
    except Exception:
        return 0.0


def is_nearly_collinear(points_xy, ratio_thresh=1e-2, min_lateral_span=0.05):
    points_xy = as_xy(points_xy)
    if len(points_xy) < 3:
        return True
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    try:
        _u, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return True
    if len(singular_values) < 2 or singular_values[0] < 1e-6:
        return True
    axis = basis[0].astype(np.float32, copy=False)
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
    lateral_span = float((centered @ lateral).max() - (centered @ lateral).min())
    return float(singular_values[1] / singular_values[0]) < float(ratio_thresh) or lateral_span < float(min_lateral_span)


class SidewalkRoadFusedProcess(VectorProcess):
    name = "sidewalk_roadfused"
    target_class_key = "sidewalk_class"
    default_target_class = 15
    requires_pose = False
    step = 20

    road_fused_input_key = "road_fused_input"
    cluster_eps = 0.9
    cluster_min_samples = 12
    outline_alpha = 1.0
    simplify_angle_thresh_deg = 9.0
    simplify_min_seg_length = 0.15

    side_min_outside = 0.25
    side_max_outside = 8.0
    max_longitudinal_offset = 10.0
    contour_parallel_angle_thresh_deg = 20.0
    contour_min_seg_length = 0.4
    inner_max_edge_offset = 3.0
    fragment_join_gap = 1.6
    fragment_join_lateral_gap = 1.0
    fused_overlap_buffer = 0.8
    fused_point_merge_radius = 0.3
    fused_resample_step = 0.5

    def __init__(self, args, config):
        super().__init__(args, config)
        self.road_samples = self.load_road_samples()

    def load_road_samples(self):
        road_path = self.config.get(
            self.road_fused_input_key,
            os.path.join(default_demo_output_dir("road"), "road_records_fused.json"),
        )
        with open(road_path, "r") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        parsed = []
        for sample in samples:
            center = np.asarray(sample.get("center", []), dtype=np.float32)
            left_edge = np.asarray(sample.get("left_edge", []), dtype=np.float32)
            right_edge = np.asarray(sample.get("right_edge", []), dtype=np.float32)
            tangent = normalize(sample.get("t", []))
            if len(center) < 2 or len(left_edge) < 2 or len(right_edge) < 2 or tangent is None:
                continue
            parsed.append({
                "index": int(sample["index"]),
                "s": float(sample.get("s", 0.0)),
                "center": center[:3].astype(np.float32, copy=False),
                "left_edge": left_edge[:3].astype(np.float32, copy=False),
                "right_edge": right_edge[:3].astype(np.float32, copy=False),
                "t": tangent,
                "w": float(sample.get("w", np.linalg.norm(left_edge[:2] - right_edge[:2]))),
            })
        if not parsed:
            raise ValueError(f"no valid road samples in {road_path}")
        return parsed

    def output_path(self):
        if self.args.output:
            return self.args.output
        return self.config.get(
            f"{self.name}_vector_output",
            os.path.join(default_demo_output_dir(self.name), "sidewalk_roadfused_records.json"),
        )

    def find_road_sample(self, logical_index, current_center):
        current_center = np.asarray(current_center, dtype=np.float32).reshape(-1)
        best = None
        best_score = None
        for sample in self.road_samples:
            idx_gap = abs(int(sample["index"]) - int(logical_index))
            center_gap = float(np.linalg.norm(sample["center"][:2] - current_center[:2]))
            score = idx_gap + 0.15 * center_gap
            if best_score is None or score < best_score:
                best_score = score
                best = sample
        return best

    def split_points_by_road(self, merged_points, road_sample):
        center_xy = road_sample["center"][:2]
        axis = road_sample["t"]
        lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
        rel = merged_points[:, :2] - center_xy[None, :]
        lon = rel @ axis
        lat = rel @ lateral

        left_edge_lat = float(np.dot(road_sample["left_edge"][:2] - center_xy, lateral))
        right_edge_lat = float(np.dot(road_sample["right_edge"][:2] - center_xy, lateral))

        masks = {}
        for name, side_sign, edge_lat in (
            ("left", -1.0, left_edge_lat),
            ("right", 1.0, right_edge_lat),
        ):
            outside = side_sign * (lat - edge_lat)
            mask = np.abs(lon) <= self.max_longitudinal_offset
            mask &= outside >= self.side_min_outside
            mask &= outside <= self.side_max_outside
            masks[name] = {
                "points": merged_points[mask],
                "lon": lon[mask],
                "lat": lat[mask],
                "outside": outside[mask],
                "edge_lat": edge_lat,
                "side_sign": side_sign,
                "axis": axis,
                "lateral": lateral,
            }
        return masks

    def build_outline_fragments(self, contour_xy, center_xy, axis, lateral, edge_lat, side_sign):
        contour_xy = as_xy(contour_xy)
        if len(contour_xy) < 3:
            return []

        closed = contour_xy.copy()
        if np.linalg.norm(closed[0] - closed[-1]) > 1e-6:
            closed = np.vstack((closed, closed[:1]))

        cos_thresh = float(np.cos(np.deg2rad(self.contour_parallel_angle_thresh_deg)))
        fragments = []
        current = []

        for i in range(len(closed) - 1):
            p1 = closed[i]
            p2 = closed[i + 1]
            vec = p2 - p1
            seg_len = float(np.linalg.norm(vec))
            if seg_len < self.contour_min_seg_length:
                keep = False
            else:
                seg_dir = vec / seg_len
                if abs(float(np.dot(seg_dir, axis))) < cos_thresh:
                    keep = False
                else:
                    mid = 0.5 * (p1 + p2)
                    lat_mid = float(np.dot(mid - center_xy, lateral))
                    outside = side_sign * (lat_mid - edge_lat)
                    keep = self.side_min_outside <= outside <= self.inner_max_edge_offset

            if keep:
                if not current:
                    current = [p1.astype(np.float32), p2.astype(np.float32)]
                else:
                    if np.linalg.norm(current[-1] - p1) > 1e-5:
                        current.append(p1.astype(np.float32))
                    current.append(p2.astype(np.float32))
            elif current:
                fragments.append(np.asarray(current, dtype=np.float32))
                current = []

        if current:
            fragments.append(np.asarray(current, dtype=np.float32))

        if len(fragments) >= 2 and np.linalg.norm(fragments[0][0] - fragments[-1][-1]) < self.fragment_join_gap:
            merged = np.vstack((fragments[-1], fragments[0]))
            fragments = [merged] + fragments[1:-1]

        return [self.clean_polyline(fragment, axis) for fragment in fragments if len(fragment) >= 2]

    def clean_polyline(self, polyline_xy, axis):
        polyline_xy = as_xy(polyline_xy)
        if len(polyline_xy) < 2:
            return polyline_xy

        dedup = [polyline_xy[0]]
        for point in polyline_xy[1:]:
            if np.linalg.norm(point - dedup[-1]) >= 1e-3:
                dedup.append(point)
        polyline_xy = np.asarray(dedup, dtype=np.float32)

        proj = polyline_xy @ axis
        if len(polyline_xy) >= 2 and proj[-1] < proj[0]:
            polyline_xy = polyline_xy[::-1].copy()

        return simplify_polyline_by_slope(
            polyline_xy,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )

    def merge_fragments(self, fragments, axis, lateral):
        if not fragments:
            return np.zeros((0, 2), dtype=np.float32)

        prepared = []
        for fragment in fragments:
            fragment = self.clean_polyline(fragment, axis)
            if len(fragment) < 2:
                continue
            proj = fragment @ axis
            lat_mean = float(np.mean(fragment @ lateral))
            prepared.append({
                "polyline": fragment,
                "start_proj": float(np.min(proj)),
                "end_proj": float(np.max(proj)),
                "lat_mean": lat_mean,
            })

        if not prepared:
            return np.zeros((0, 2), dtype=np.float32)

        prepared.sort(key=lambda item: item["start_proj"])
        merged = prepared[0]["polyline"]

        for item in prepared[1:]:
            candidate = item["polyline"]
            gap = float(np.linalg.norm(candidate[0] - merged[-1]))
            lat_gap = abs(item["lat_mean"] - float(np.mean(merged @ lateral)))
            if gap <= self.fragment_join_gap and lat_gap <= self.fragment_join_lateral_gap:
                merged = self.concat_polylines(merged, candidate)
            else:
                if line_length(candidate) > line_length(merged):
                    merged = candidate

        return self.clean_polyline(merged, axis)

    def concat_polylines(self, poly_a, poly_b):
        poly_a = as_xy(poly_a)
        poly_b = as_xy(poly_b)
        if len(poly_a) == 0:
            return poly_b
        if len(poly_b) == 0:
            return poly_a
        if np.linalg.norm(poly_a[-1] - poly_b[0]) < 1e-3:
            return np.vstack((poly_a, poly_b[1:]))
        return np.vstack((poly_a, poly_b))

    def extract_sidewalk_on_side(self, side_input, road_sample, anchor_xyz):
        side_points = side_input["points"]
        if len(side_points) == 0:
            return None

        cluster_points = keep_largest_cluster(
            side_points[:, :3],
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        if len(cluster_points) < 3:
            return None

        contour_xy = extract_outline_by_alphashape(cluster_points, alpha=self.outline_alpha)
        contour_xy = simplify_polyline_by_slope(
            contour_xy,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )
        if len(contour_xy) < 3:
            return None

        center_xy = road_sample["center"][:2]
        axis = side_input["axis"]
        lateral = side_input["lateral"]
        fragments = self.build_outline_fragments(
            contour_xy,
            center_xy,
            axis,
            lateral,
            side_input["edge_lat"],
            side_input["side_sign"],
        )
        inner_polyline = self.merge_fragments(fragments, axis, lateral)
        if len(inner_polyline) < 2:
            return None

        sidewalk_z = float(np.median(cluster_points[:, 2])) if len(cluster_points) != 0 else 0.0
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32)
        return {
            "anchor": np.asarray(anchor_xyz[:2], dtype=np.float32).tolist(),
            "centroid": centroid.tolist(),
            "sidewalk_z": sidewalk_z,
            "outline": contour_xy.astype(np.float32).tolist(),
            "inner_polyline": inner_polyline.astype(np.float32).tolist(),
            "cluster_points": cluster_points,
        }

    def make_record(self, logical_index, road_sample, left_record, right_record):
        return {
            "index": int(logical_index),
            "road_center": road_sample["center"].astype(np.float32).tolist(),
            "road_tangent": road_sample["t"].astype(np.float32).tolist(),
            "road_width": float(road_sample["w"]),
            "left_road_edge": road_sample["left_edge"].astype(np.float32).tolist(),
            "right_road_edge": road_sample["right_edge"].astype(np.float32).tolist(),
            "left_sidewalk": left_record,
            "right_sidewalk": right_record,
        }

    def process(self, runtime, logical_index):
        if self.config["mode"] != "outdoor" or not self.ready():
            return False

        process_ctx = self.build_context(runtime, logical_index)
        if not self.has_valid_context(process_ctx):
            print(f"skip: empty history window for {self.name} @ {logical_index}")
            return False

        merged_points = process_ctx["current_points"]
        road_sample = self.find_road_sample(logical_index, process_ctx["current_center"])
        if road_sample is None:
            print(f"skip: no road sample for {self.name} @ {logical_index}")
            return False

        split = self.split_points_by_road(merged_points, road_sample)
        print(
            f"{self.name} split @ {logical_index}: "
            f"left={len(split['left']['points'])} right={len(split['right']['points'])} total={len(merged_points)}"
        )

        left_record = self.extract_sidewalk_on_side(split["left"], road_sample, road_sample["left_edge"])
        right_record = self.extract_sidewalk_on_side(split["right"], road_sample, road_sample["right_edge"])
        if left_record is None and right_record is None:
            print(f"skip: no valid sidewalk contour on either side @ {logical_index}")
            return False

        all_cluster_points = []
        all_contours = []
        for side_record in (left_record, right_record):
            if side_record is None:
                continue
            all_cluster_points.append(side_record.pop("cluster_points"))
            all_contours.append(as_xy(side_record["outline"]))

        canvas, to_canvas = self.draw_debug_canvas(
            np.vstack(all_cluster_points) if all_cluster_points else np.zeros((0, 3), dtype=np.float32),
            np.vstack(all_contours) if all_contours else np.zeros((0, 2), dtype=np.float32),
            process_ctx,
        )

        for road_key, color in (("left_edge", (255, 80, 80)), ("right_edge", (80, 220, 80))):
            edge_xy = as_xy(road_sample[road_key])
            edge_canvas = to_canvas(edge_xy).reshape(-1, 1, 2)
            cv2.circle(canvas, tuple(edge_canvas[0, 0]), 6, color, -1)

        def draw_side(record, contour_color, inner_color):
            if record is None:
                return
            contour_canvas = to_canvas(as_xy(record["outline"])).reshape(-1, 1, 2)
            cv2.polylines(canvas, [contour_canvas], True, contour_color, 4)
            inner_canvas = to_canvas(as_xy(record["inner_polyline"])).reshape(-1, 1, 2)
            cv2.polylines(canvas, [inner_canvas], False, inner_color, 5)

        draw_side(left_record, (255, 160, 80), (255, 80, 80))
        draw_side(right_record, (80, 180, 255), (80, 220, 80))

        self.records.append(self.make_record(logical_index, road_sample, left_record, right_record))
        self.save_debug_canvas(logical_index, canvas)
        self.save_origin_debug_image(runtime, logical_index, logical_index)
        return True

    def merged_outline(self, outline_arrays, polyline_arrays):
        points = []
        for arr in outline_arrays + polyline_arrays:
            arr = as_xy(arr)
            if len(arr) != 0:
                points.append(arr)
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        merged_points = np.vstack(points)
        if is_nearly_collinear(merged_points):
            return self.buffered_outline_from_polyline(polyline_arrays)
        try:
            outline = extract_outline_by_alphashape(
                np.column_stack((merged_points, np.zeros((len(merged_points),), dtype=np.float32))),
                alpha=self.outline_alpha,
            )
        except Exception:
            return self.buffered_outline_from_polyline(polyline_arrays)
        outline = simplify_polyline_by_slope(
            outline,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )
        if len(outline) < 3 or polygon_area_from_outline(outline) <= 1e-6:
            return self.buffered_outline_from_polyline(polyline_arrays)
        return outline

    def buffered_outline_from_polyline(self, polyline_arrays):
        points = []
        for arr in polyline_arrays:
            arr = as_xy(arr)
            if len(arr) != 0:
                points.append(arr)
        if not points:
            return np.zeros((0, 2), dtype=np.float32)

        merged_polyline = np.vstack(points).astype(np.float32, copy=False)
        if len(merged_polyline) < 2:
            return np.zeros((0, 2), dtype=np.float32)

        line = polyline_to_linestring(merged_polyline)
        if line is None:
            return np.zeros((0, 2), dtype=np.float32)

        # Degenerate sidewalk outlines appear in sparse tracks as nearly straight
        # line segments. Fall back to a thin buffered ribbon so fusion can finish
        # and downstream overlay still has an area-like contour to draw.
        buffer_radius = max(float(self.fused_point_merge_radius), 0.25)
        polygon = line.buffer(buffer_radius, cap_style=2, join_style=2)
        if polygon.is_empty:
            return np.zeros((0, 2), dtype=np.float32)
        outline = np.asarray(polygon.exterior.coords, dtype=np.float32)
        return simplify_polyline_by_slope(
            outline,
            angle_thresh_deg=self.simplify_angle_thresh_deg,
            min_seg_length=self.simplify_min_seg_length,
        )

    def prune_polyline_points(self, polyline_xy, other_buffer):
        polyline_xy = as_xy(polyline_xy)
        if len(polyline_xy) == 0:
            return polyline_xy
        kept = [point for point in polyline_xy if not other_buffer.contains(Point(float(point[0]), float(point[1])))]
        if len(kept) >= 2:
            return np.asarray(kept, dtype=np.float32)
        return polyline_xy

    def dedupe_sorted_points(self, points_xy):
        points_xy = as_xy(points_xy)
        if len(points_xy) == 0:
            return points_xy
        dedup = [points_xy[0]]
        for point in points_xy[1:]:
            if np.linalg.norm(point - dedup[-1]) >= self.fused_point_merge_radius:
                dedup.append(point)
        return np.asarray(dedup, dtype=np.float32)

    def rebuild_track_polyline(self, polyline_arrays, axis):
        axis = normalize(axis)
        if axis is None:
            return np.zeros((0, 2), dtype=np.float32)

        lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
        all_points = []
        for arr in polyline_arrays:
            arr = as_xy(arr)
            if len(arr) != 0:
                all_points.append(arr)
        if not all_points:
            return np.zeros((0, 2), dtype=np.float32)

        points = np.vstack(all_points)
        proj = points @ axis
        lat = points @ lateral
        order = np.argsort(proj)
        proj = proj[order]
        lat = lat[order]

        bins = []
        start = 0
        step = max(float(self.fused_resample_step), 1e-3)
        while start < len(proj):
            end = start + 1
            while end < len(proj) and proj[end] - proj[start] <= step:
                end += 1
            bins.append((start, end))
            start = end

        fused_points = []
        for start, end in bins:
            proj_med = float(np.median(proj[start:end]))
            lat_med = float(np.median(lat[start:end]))
            point = axis * proj_med + lateral * lat_med
            fused_points.append(point.astype(np.float32))

        if len(fused_points) < 2:
            idx_min = int(np.argmin(proj))
            idx_max = int(np.argmax(proj))
            fused = np.vstack((points[order[idx_min]], points[order[idx_max]])).astype(np.float32)
            return self.clean_polyline(fused, axis)

        fused = self.dedupe_sorted_points(np.asarray(fused_points, dtype=np.float32))
        return self.clean_polyline(fused, axis)

    def merge_track_polyline(self, base_polyline, new_polyline, axis):
        base_line = polyline_to_linestring(base_polyline)
        new_line = polyline_to_linestring(new_polyline)
        if base_line is None:
            return as_xy(new_polyline), False
        if new_line is None:
            return as_xy(base_polyline), False

        base_buffer = base_line.buffer(self.fused_overlap_buffer, cap_style=2, join_style=2)
        new_buffer = new_line.buffer(self.fused_overlap_buffer, cap_style=2, join_style=2)
        if base_buffer.intersection(new_buffer).is_empty:
            return as_xy(base_polyline), False

        remain_base = self.prune_polyline_points(base_polyline, new_buffer)
        remain_new = self.prune_polyline_points(new_polyline, base_buffer)
        merged_points = np.vstack((remain_base, remain_new))
        proj = merged_points @ axis
        merged_points = merged_points[np.argsort(proj)]
        merged_points = self.dedupe_sorted_points(merged_points)
        if len(merged_points) < 2:
            return as_xy(base_polyline if line_length(base_polyline) >= line_length(new_polyline) else new_polyline), True
        return self.clean_polyline(merged_points, axis), True

    def init_track(self, side_key, record, track_id):
        polyline_xy = as_xy(record["inner_polyline"])
        outline_xy = as_xy(record["outline"])
        centroid = as_xy(record["centroid"])
        return {
            "id": int(track_id),
            "side": side_key,
            "centroid": centroid[0].tolist() if len(centroid) == 1 else [0.0, 0.0],
            "sidewalk_z": float(record["sidewalk_z"]),
            "area": float(polygon_area_from_outline(outline_xy)),
            "outline": outline_xy.tolist(),
            "polyline": polyline_xy.tolist(),
            "source_indices": [],
            "anchors": [],
            "inner_polylines": [polyline_xy.tolist()],
            "_merged_polyline": polyline_xy,
            "_outline_arrays": [outline_xy],
            "_polyline_arrays": [polyline_xy],
            "_axis_sum": np.zeros((2,), dtype=np.float32),
        }

    def update_track_metadata(self, track):
        axis_ref = normalize(track["_axis_sum"])
        if axis_ref is None:
            axis_ref = np.array([1.0, 0.0], dtype=np.float32)
        track["_merged_polyline"] = self.rebuild_track_polyline(track["_polyline_arrays"], axis_ref)
        outline_xy = self.merged_outline(track["_outline_arrays"], track["_polyline_arrays"])
        track["outline"] = outline_xy.tolist()
        track["area"] = float(polygon_area_from_outline(outline_xy))
        track["polyline"] = as_xy(track["_merged_polyline"]).tolist()

        points = [arr for arr in track["_polyline_arrays"] if len(arr) != 0]
        if points:
            merged_points = np.vstack(points)
            track["centroid"] = merged_points.mean(axis=0).astype(np.float32).tolist()
        else:
            track["centroid"] = [0.0, 0.0]
        track["sidewalk_z"] = float(track["sidewalk_z"])

    def fuse_side_records(self, side_key):
        track_id = 0
        tracks = []
        for frame_record in self.records:
            side_record = frame_record.get(side_key)
            if not isinstance(side_record, dict):
                continue

            polyline_xy = as_xy(side_record.get("inner_polyline", []))
            if len(polyline_xy) < 2:
                continue

            axis = normalize(frame_record.get("road_tangent", []))
            if axis is None:
                continue

            matched_track = None
            merged_polyline = None
            for track in tracks:
                axis_sum = track["_axis_sum"] + axis
                axis_ref = normalize(axis_sum)
                if axis_ref is None:
                    axis_ref = axis
                merged_candidate, merged = self.merge_track_polyline(track["_merged_polyline"], polyline_xy, axis_ref)
                if merged:
                    matched_track = track
                    merged_polyline = merged_candidate
                    track["_axis_sum"] = axis_sum
                    break

            if matched_track is None:
                matched_track = self.init_track(side_key, side_record, track_id)
                matched_track["_axis_sum"] = axis.copy()
                matched_track["source_indices"].append(int(frame_record["index"]))
                anchor_xy = as_xy(side_record.get("anchor", []))
                if len(anchor_xy) == 1:
                    matched_track["anchors"].append(anchor_xy[0].tolist())
                self.update_track_metadata(matched_track)
                tracks.append(matched_track)
                track_id += 1
                continue

            matched_track["_merged_polyline"] = merged_polyline
            matched_track["_polyline_arrays"].append(polyline_xy)
            matched_track["_outline_arrays"].append(as_xy(side_record.get("outline", [])))
            matched_track["inner_polylines"].append(polyline_xy.tolist())
            matched_track["source_indices"].append(int(frame_record["index"]))
            anchor_xy = as_xy(side_record.get("anchor", []))
            if len(anchor_xy) == 1:
                matched_track["anchors"].append(anchor_xy[0].tolist())
            matched_track["sidewalk_z"] = float(
                0.5 * matched_track["sidewalk_z"] + 0.5 * float(side_record.get("sidewalk_z", matched_track["sidewalk_z"]))
            )
            self.update_track_metadata(matched_track)

        for track in tracks:
            if not track["source_indices"]:
                continue
            if not track["anchors"]:
                track["anchors"] = []
            self.update_track_metadata(track)
            for key in ("_merged_polyline", "_outline_arrays", "_polyline_arrays", "_axis_sum"):
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
            "left_sidewalks": self.fuse_side_records("left_sidewalk"),
            "right_sidewalks": self.fuse_side_records("right_sidewalk"),
        }

        raw_path = self.output_path()
        fused_path = os.path.splitext(raw_path)[0] + "_fused.json"
        with open(fused_path, "w") as f:
            json.dump(fused, f, indent=2)
        print(
            f"saved fused sidewalk roadfused records to {fused_path} "
            f"(left={len(fused['left_sidewalks'])}, right={len(fused['right_sidewalks'])})"
        )
