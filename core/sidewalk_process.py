import json
import os
import warnings

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from shapely.geometry import MultiPolygon, Polygon

from core.vector_common import FixedQueue
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


def _linefit_subset_simplify_closed(outline_xy, perp_tol=0.20):
    """Anchor-constrained piecewise-linear simplification of a closed polygon.

    Greedy Reumann-Witkam variant: walks the closed polygon starting at the
    sharpest turn (most-likely-corner) vertex, and at each step extends the
    chord forward as long as every interior vertex stays within `perp_tol`
    of chord(V_i, V_j). Only chord endpoints are kept. Output is a strict
    subset of the input (no synthetic vertices).

    Use case: alpha-shape outlines have small LiDAR-noise jaggies along
    actually-straight curb segments. This collapses those jaggies into a
    single straight edge while preserving genuine corners and concavities
    (those force a chord break naturally).

    Args:
        outline_xy: (N, 2) numpy array of CCW closed polygon vertices
                    (last vertex may or may not duplicate first).
        perp_tol:   maximum allowed perpendicular distance from any interior
                    vertex to its enclosing chord (metres).

    Returns:
        (M, 2) float32 array of kept vertices, M <= N. The polygon is
        returned open (no duplicated closing vertex).
    """
    pts = np.asarray(outline_xy, dtype=np.float64)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    if n < 4:
        return np.asarray(outline_xy, dtype=np.float32)

    # Start at the vertex with the sharpest turn — guarantees that real
    # corners are preserved (a sharp corner cannot be absorbed into a chord).
    e_prev = pts - np.roll(pts, 1, axis=0)
    e_next = np.roll(pts, -1, axis=0) - pts
    norm_prev = np.linalg.norm(e_prev, axis=1)
    norm_next = np.linalg.norm(e_next, axis=1)
    denom = np.maximum(norm_prev * norm_next, 1e-12)
    cos_turn = np.clip(np.einsum('ij,ij->i', e_prev, e_next) / denom, -1.0, 1.0)
    turn = np.arccos(cos_turn)
    start = int(np.argmax(turn))

    keep = [start]
    i = start
    for _ in range(n + 1):
        max_k = 1
        for k in range(2, n + 1):
            j_idx = (i + k) % n
            chord_start = pts[i]
            chord_end = pts[j_idx]
            v = chord_end - chord_start
            v_norm = float(np.hypot(v[0], v[1]))
            if v_norm < 1e-9:
                break
            mid_offsets = (i + np.arange(1, k)) % n
            mids = pts[mid_offsets]
            cross = v[0] * (mids[:, 1] - chord_start[1]) - v[1] * (mids[:, 0] - chord_start[0])
            d_max = float(np.abs(cross).max()) / v_norm
            if d_max > perp_tol:
                break
            max_k = k
            if j_idx == start:
                break
        next_i = (i + max_k) % n
        if next_i == start:
            break
        keep.append(next_i)
        i = next_i

    if len(keep) < 3:
        return np.asarray(outline_xy, dtype=np.float32)
    return pts[keep].astype(np.float32, copy=False)


def _interp_z_for_outline(xy_source, z_source, xy_query):
    """Interpolate z for 2-D query points from a scattered (x,y)→z cloud.

    Uses LinearNDInterpolator (Delaunay-based) so interior vertices get a
    smooth planar estimate; vertices that fall outside the convex hull of the
    source cloud (extrapolation) fall back to NearestNDInterpolator so no
    NaN values escape.
    """
    pts = np.asarray(xy_source, dtype=np.float64)
    zs  = np.asarray(z_source,  dtype=np.float64).ravel()
    qry = np.asarray(xy_query,  dtype=np.float64)
    if len(pts) < 3:
        return NearestNDInterpolator(pts, zs)(qry)
    z_vals = LinearNDInterpolator(pts, zs)(qry)
    nan_mask = np.isnan(z_vals)
    if nan_mask.any():
        z_vals[nan_mask] = NearestNDInterpolator(pts, zs)(qry[nan_mask])
    return z_vals.astype(np.float32)


class SidewalkEdgeProcess(VectorProcess):
    name = "sidewalk"
    target_class_key = "sidewalk_class"
    default_target_class = 15
    requires_pose = False
    # The sidewalk process never reads front/last windows (only current_points),
    # so dirc_window's "look ahead N frames" requirement is wasted overhead and
    # forces the bag's last 20 frames to be silently skipped. Override to 0 so
    # the only warm-up is `window_size=10` and the tail tracks all the way out.
    dirc_window = 0
    cluster_eps = 0.8
    cluster_min_samples = 12
    outline_alpha = 1.0
    merge_overlap_buffer = 0.2
    fused_outline_simplify_tol = 0.3
    fused_outline_smooth_iterations = 2
    fused_outline_min_seg_len = 0.35
    fused_outline_post_simplify_tol = 0.12
    # Vector-space morphological regularization is OFF by default. close
    # bridges gaps between disjoint sidewalk segments along the trajectory,
    # so the polygon balloons over the actual point footprint; open eats
    # small polygons. Both run only when the user opts in via config.
    fused_outline_close_buffer = 0.0
    fused_outline_open_buffer = 0.0
    fused_min_area = 2.0
    fused_min_support_frames = 3
    bev_res = 0.15
    bev_min_frames = 3
    bev_min_sidewalk_fraction = 0.5
    bev_grid_cap = 4_000_000
    # Default back to the 803a0e1 alpha-shape pipeline. The BEV temporal-vote
    # path was a regression on the 0421-AM bag (polygons shrink below the
    # actual sidewalk extent) — restore the older, simpler chain as the
    # default and keep the BEV code reachable behind explicit config flags.
    outline_method = "alpha"
    outline_bev_res = 0.15
    outline_bev_dilate_px = 1
    use_bev_vote = False
    # Voxel-downsample the per-cluster point set before alpha-shape. With the
    # body-origin fix the per-frame sidewalk class is dense (~1.5k pts/frame
    # × 10-frame window → 5k-15k pts per cluster), and alphashape on raw
    # input is the dominant per-frame cost. 0.15m preserves the polygon
    # envelope (cluster_eps=0.8 is much larger) while cutting input ~10x.
    alpha_input_voxel = 0.15
    alpha_input_max_points = 4000

    def __init__(self, args, config):
        super().__init__(args, config)
        self.step = max(int(config.get("sidewalk_step", config.get("vector_step", self.step))), 1)
        self.cluster_eps = float(config.get("sidewalk_cluster_eps", self.cluster_eps))
        self.merge_overlap_buffer = float(config.get("sidewalk_merge_overlap_buffer", self.merge_overlap_buffer))
        self.cluster_min_samples = int(config.get("sidewalk_cluster_min_samples", self.cluster_min_samples))
        self.outline_alpha = float(config.get("sidewalk_outline_alpha", self.outline_alpha))
        self.max_candidates_per_frame = max(int(config.get("sidewalk_max_candidates_per_frame", 3)), 1)
        self.min_cluster_area = float(config.get("sidewalk_cluster_min_area", 1.2))
        self.min_cluster_density = float(config.get("sidewalk_cluster_min_density", 0.35))
        self.min_compactness = float(config.get("sidewalk_cluster_min_compactness", 0.004))
        self.max_center_distance = float(config.get("sidewalk_cluster_max_center_distance", 45.0))

        # Phase A.2: BEV temporal voting on the sliding window of full-class
        # point clouds. A cell becomes a "stable sidewalk cell" only when at
        # least `bev_min_frames` distinct frames in the window saw sidewalk
        # there AND the per-cell sidewalk fraction (sidewalk-frames /
        # any-class-frames) is at least `bev_min_sidewalk_fraction`. The
        # representative point per cell is its center with the mean Z of the
        # contributing sidewalk hits. See doc/sidewalk_vectorization_roadmap.md.
        self.bev_res = float(config.get("sidewalk_bev_res", self.bev_res))
        self.bev_min_frames = max(int(config.get("sidewalk_bev_min_frames", self.bev_min_frames)), 1)
        self.bev_min_sidewalk_fraction = float(
            config.get("sidewalk_bev_min_sidewalk_fraction", self.bev_min_sidewalk_fraction)
        )
        self.bev_grid_cap = max(int(config.get("sidewalk_bev_grid_cap", self.bev_grid_cap)), 10_000)
        self.full_history = FixedQueue(2 * self.dirc_window + self.window_size)

        # Phase 1.1: BEV-contour outline replaces alpha-shape. Both the
        # per-frame outline and the per-track final outline use this. Set
        # `sidewalk_outline_method = "alpha"` to fall back to alphashape
        # for A/B comparison.
        self.outline_method = str(config.get("sidewalk_outline_method", self.outline_method)).lower()
        if self.outline_method not in {"bev", "alpha"}:
            raise ValueError(f"sidewalk_outline_method must be 'bev' or 'alpha', got {self.outline_method!r}")
        self.outline_bev_res = float(config.get("sidewalk_outline_bev_res", self.outline_bev_res))
        self.outline_bev_dilate_px = max(int(config.get("sidewalk_outline_bev_dilate_px", self.outline_bev_dilate_px)), 0)
        self.use_bev_vote = bool(config.get("sidewalk_use_bev_vote", self.use_bev_vote))
        self.alpha_input_voxel = float(config.get("sidewalk_alpha_input_voxel", self.alpha_input_voxel))
        self.alpha_input_max_points = max(int(config.get("sidewalk_alpha_input_max_points", self.alpha_input_max_points)), 0)

    def ingest_frame(self, sempcd):
        # Base class drops everything but the target class; Phase A.2 needs
        # the *full* per-frame cloud so the BEV vote can compute the per-cell
        # sidewalk fraction. Keep both queues in lockstep so window slicing
        # in process() lines up with the base class context — but only when
        # BEV voting is on. Otherwise full_history just burns RAM and
        # per-frame copies (~50 frames × full cloud each).
        if sempcd is None or len(sempcd) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            self.history.append(empty)
            if self.use_bev_vote:
                self.full_history.append(empty)
            return
        sempcd = np.asarray(sempcd, dtype=np.float32)
        self.history.append(sempcd[sempcd[:, 3] == self.target_class])
        if self.use_bev_vote:
            self.full_history.append(sempcd)

    def _bev_temporal_vote(self, full_frames):
        """Reduce a temporal window of full-class point clouds to one
        representative point per stable sidewalk cell.

        A cell is kept iff:
          * sidewalk_frame_count[cell] >= self.bev_min_frames
          * sidewalk_frame_count[cell] / total_frame_count[cell]
            >= self.bev_min_sidewalk_fraction

        Returns an (M, 3) float32 array of (x, y, mean_sidewalk_z).
        """
        sidewalk_class = int(self.target_class)
        non_empty = [np.asarray(f, dtype=np.float32) for f in full_frames if len(f) > 0]
        if not non_empty:
            return np.zeros((0, 3), dtype=np.float32)

        all_xy = np.concatenate([f[:, :2] for f in non_empty], axis=0)
        if len(all_xy) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        res = float(self.bev_res)
        if res <= 0.0:
            return np.zeros((0, 3), dtype=np.float32)

        xy_min = all_xy.min(axis=0).astype(np.float64)
        xy_max = all_xy.max(axis=0).astype(np.float64)
        grid_w = int(np.ceil((xy_max[0] - xy_min[0]) / res)) + 1
        grid_h = int(np.ceil((xy_max[1] - xy_min[1]) / res)) + 1
        if grid_w <= 0 or grid_h <= 0 or grid_w * grid_h > self.bev_grid_cap:
            # Degenerate or pathological window — fall back to raw sidewalk
            # points so we never silently drop a frame.
            sidewalk_pts = [f[f[:, 3] == sidewalk_class, :3] for f in non_empty]
            sidewalk_pts = [p for p in sidewalk_pts if len(p) > 0]
            if not sidewalk_pts:
                return np.zeros((0, 3), dtype=np.float32)
            return np.vstack(sidewalk_pts).astype(np.float32, copy=False)

        sidewalk_frame_count = np.zeros((grid_h, grid_w), dtype=np.int32)
        total_frame_count = np.zeros((grid_h, grid_w), dtype=np.int32)
        z_sum = np.zeros((grid_h, grid_w), dtype=np.float64)
        z_count = np.zeros((grid_h, grid_w), dtype=np.int32)

        for frame in non_empty:
            col_all = ((frame[:, 0].astype(np.float64) - xy_min[0]) / res).astype(np.int64)
            row_all = ((frame[:, 1].astype(np.float64) - xy_min[1]) / res).astype(np.int64)
            np.clip(col_all, 0, grid_w - 1, out=col_all)
            np.clip(row_all, 0, grid_h - 1, out=row_all)

            cell_all = row_all * grid_w + col_all
            unique_all = np.unique(cell_all)
            ur = (unique_all // grid_w).astype(np.int32)
            uc = (unique_all % grid_w).astype(np.int32)
            total_frame_count[ur, uc] += 1

            mask_sw = frame[:, 3] == sidewalk_class
            if not mask_sw.any():
                continue
            cell_sw = cell_all[mask_sw]
            unique_sw = np.unique(cell_sw)
            sr = (unique_sw // grid_w).astype(np.int32)
            sc = (unique_sw % grid_w).astype(np.int32)
            sidewalk_frame_count[sr, sc] += 1

            sw_pts = frame[mask_sw]
            sw_row = row_all[mask_sw].astype(np.int32)
            sw_col = col_all[mask_sw].astype(np.int32)
            np.add.at(z_sum, (sw_row, sw_col), sw_pts[:, 2].astype(np.float64))
            np.add.at(z_count, (sw_row, sw_col), 1)

        if not (sidewalk_frame_count > 0).any():
            return np.zeros((0, 3), dtype=np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(
                total_frame_count > 0,
                sidewalk_frame_count / np.maximum(total_frame_count, 1),
                0.0,
            )

        keep = (sidewalk_frame_count >= int(self.bev_min_frames)) & (
            fraction >= float(self.bev_min_sidewalk_fraction)
        )
        if not keep.any():
            return np.zeros((0, 3), dtype=np.float32)

        rows, cols = np.where(keep)
        z_safe = z_count[rows, cols]
        cz = np.where(z_safe > 0, z_sum[rows, cols] / np.maximum(z_safe, 1), 0.0)
        cx = (cols.astype(np.float64) + 0.5) * res + xy_min[0]
        cy = (rows.astype(np.float64) + 0.5) * res + xy_min[1]
        return np.column_stack([cx, cy, cz]).astype(np.float32)

    def _extract_outline_bev(self, points_xyz):
        """Outline extraction for the voted-cell point sets produced by A.2.

        Rasterize cells into a BEV occupancy grid, morphologically close
        gaps up to ~cluster_eps (DBSCAN already grouped within that
        radius), then take the largest external contour. Hits the right
        area envelope without the per-call O(N^2) work of alpha-shape.
        """
        pts = np.asarray(points_xyz, dtype=np.float32)
        if len(pts) < 3:
            return np.zeros((0, 2), dtype=np.float32)

        res = float(self.outline_bev_res)
        if res <= 0.0:
            return np.zeros((0, 2), dtype=np.float32)

        # Bridge gaps up to `cluster_eps`: DBSCAN already grouped cells
        # within that radius into one cluster, so the morph close should
        # be at least that wide to keep them connected in BEV.
        dilate = max(int(self.outline_bev_dilate_px), int(np.ceil(self.cluster_eps / max(res, 1e-3) / 2.0)))

        xy = pts[:, :2]
        margin = res * (1 + dilate + 1)
        xy_min = xy.min(axis=0) - margin
        xy_max = xy.max(axis=0) + margin
        grid_w = int(np.ceil((xy_max[0] - xy_min[0]) / res)) + 1
        grid_h = int(np.ceil((xy_max[1] - xy_min[1]) / res)) + 1
        if grid_w <= 0 or grid_h <= 0 or grid_w * grid_h > self.bev_grid_cap:
            return np.zeros((0, 2), dtype=np.float32)

        cols = ((xy[:, 0] - xy_min[0]) / res).astype(np.int32)
        rows = ((xy[:, 1] - xy_min[1]) / res).astype(np.int32)
        np.clip(cols, 0, grid_w - 1, out=cols)
        np.clip(rows, 0, grid_h - 1, out=rows)

        mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        mask[rows, cols] = 255
        if dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 2), dtype=np.float32)
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return np.zeros((0, 2), dtype=np.float32)

        cnt = contour.reshape(-1, 2).astype(np.float32)
        outline = np.empty_like(cnt)
        outline[:, 0] = (cnt[:, 0] + 0.5) * res + xy_min[0]
        outline[:, 1] = (cnt[:, 1] + 0.5) * res + xy_min[1]
        if not np.allclose(outline[0], outline[-1]):
            outline = np.vstack((outline, outline[:1])).astype(np.float32, copy=False)
        return outline

    def _voxel_downsample_xyz(self, points_xyz):
        """2D voxel downsample to cap alphashape input size.

        alphashape cost grows superlinearly in N. cluster_eps is 0.8m, so
        snapping to a 0.15m grid can't change which DBSCAN cluster a point
        belongs to and leaves the polygon envelope visually identical.
        """
        pts = np.asarray(points_xyz, dtype=np.float32)
        if len(pts) == 0:
            return pts
        voxel = float(self.alpha_input_voxel)
        if voxel > 0.0 and len(pts) > 32:
            keys = np.floor(pts[:, :2] / voxel).astype(np.int64)
            # Encode (kx, ky) into a single key for np.unique.
            key_min = keys.min(axis=0)
            keys -= key_min
            span_y = int(keys[:, 1].max()) + 1
            encoded = keys[:, 0] * span_y + keys[:, 1]
            _, first_idx = np.unique(encoded, return_index=True)
            pts = pts[first_idx]
        cap = int(self.alpha_input_max_points)
        if cap > 0 and len(pts) > cap:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(pts), cap, replace=False)
            pts = pts[sel]
        return pts

    def _extract_outline(self, points_xyz):
        if self.outline_method == "bev":
            return self._extract_outline_bev(points_xyz)
        reduced = self._voxel_downsample_xyz(points_xyz)
        return extract_outline_by_alphashape(reduced, alpha=self.outline_alpha)

    def make_record(self, logical_index, process_ctx, cluster_points, contour_xy, sidewalk_z):
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32) if len(cluster_points) != 0 else np.zeros((2,), dtype=np.float32)
        outline_xy = as_xy(contour_xy)
        if len(cluster_points) >= 3 and len(outline_xy) > 0:
            outline_z = _interp_z_for_outline(
                cluster_points[:, :2], cluster_points[:, 2], outline_xy
            ).tolist()
        else:
            outline_z = [float(sidewalk_z)] * len(outline_xy)
        rec = {
            "index": int(logical_index),
            "target_class": int(self.target_class),
            "window_centroid": process_ctx["current_center"].astype(np.float32).tolist(),
            "centroid": centroid.tolist(),
            "sidewalk_z": float(sidewalk_z),
            "outline": outline_xy.tolist(),
            "outline_z": outline_z,
        }
        # Underlying LiDAR points used for alpha-shape — kept as a numpy array
        # for track-level accumulation. Stripped before JSON serialization.
        rec["_cluster_points"] = np.asarray(cluster_points, dtype=np.float32)
        return rec

    def _cluster_score(self, cluster_points, polygon, process_center_xy):
        area = float(polygon.area)
        centroid = cluster_points[:, :2].mean(axis=0).astype(np.float32)

        # Only reject tiny fragmented point noise; candidate priority is area.
        if area < self.min_cluster_area:
            return None

        return {
            "score": area,
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

            contour_xy = self._extract_outline(cluster_points)
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

        # 803a0e1 default: cluster on the raw class-15 window points; let
        # DBSCAN + score-based filtering reject the noise. The BEV temporal-
        # voting path is reachable by setting `sidewalk_use_bev_vote: true`
        # for A/B comparison — by default it shrinks polygons below the real
        # sidewalk extent, so we don't run it here.
        if self.use_bev_vote:
            full_window = list(self.full_history[self.dirc_window:(self.dirc_window + self.window_size)])
            merged_points = self._bev_temporal_vote(full_window)
            if len(merged_points) == 0:
                print(f"skip: BEV vote produced no stable sidewalk cells @ {logical_index}")
                return False
        else:
            merged_points = process_ctx["current_points"]
            if len(merged_points) == 0:
                print(f"skip: empty class-{self.target_class} window @ {logical_index}")
                return False

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
        outline_z = np.asarray(record.get("outline_z", [float(record["sidewalk_z"])] * len(outline_xy)), dtype=np.float32)
        # Track-level LiDAR point accumulation: starts with this record's
        # cluster_points; merged records concatenate their points onto this.
        # update_track_metadata then re-runs alpha-shape on the union.
        cp = record.get("_cluster_points")
        input_points = np.asarray(cp, dtype=np.float32) if cp is not None else np.zeros((0, 3), dtype=np.float32)
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
            "_outline_z_arrays": [outline_z],
            "_z_values": [float(record["sidewalk_z"])],
            "_input_points": input_points,
        }

    def merge_track_outline(self, base_polygon, new_polygon, outline_arrays):
        # Phase 1.4: previously this re-ran alpha-shape on every accumulated
        # outline-point set after each successful merge — O(N^2) work over
        # the lifetime of a track, and the top contributor to fuse_records
        # cost on the 200-frame pkl. The final polished outline is now
        # computed once per track in update_track_metadata; here we only
        # need a cheap union for the running polygon (used for future
        # overlap tests).
        if base_polygon is None or new_polygon is None:
            return base_polygon, False

        base_hit = base_polygon.buffer(self.merge_overlap_buffer)
        if not base_hit.intersects(new_polygon):
            return base_polygon, False

        merged_polygon = base_polygon.union(new_polygon)
        if isinstance(merged_polygon, MultiPolygon):
            merged_polygon = max(merged_polygon.geoms, key=lambda geom: geom.area)
        return merged_polygon, True

    def update_track_metadata(self, track):
        # Per-track alpha-shape on accumulated LiDAR points: this is the whole
        # point — every output vertex is now a real LiDAR measurement, gaps
        # within a track that arose from per-window sparsity are bridged
        # naturally because the points across all merged records are visible
        # to a single alpha-shape pass.
        input_pts = track.get("_input_points")
        if input_pts is not None and len(input_pts) >= 4:
            try:
                contour_xy = self._extract_outline(np.asarray(input_pts, dtype=np.float32))
                rebuilt, _ = polygon_from_outline(as_xy(contour_xy))
                if rebuilt is not None and not rebuilt.is_empty and rebuilt.area > 0:
                    track["_polygon"] = rebuilt
            except Exception as e:
                print(f"update_track_metadata: per-track alpha-shape failed: {e}")

        polygon = track["_polygon"]
        outline_xy = outline_from_polygon(polygon)

        # Anchor-constrained line-fit simplification (V8): collapse straight
        # runs of LiDAR-jaggy alpha-shape vertices into single edges while
        # preserving real corners. All output vertices are a strict subset
        # of the input (no synthetic points, unlike Chaikin smoothing).
        linefit_enable = bool(self.config.get("sidewalk_linefit_enable", True))
        if linefit_enable and len(outline_xy) >= 4:
            perp_tol = float(self.config.get("sidewalk_linefit_perp_tol", 0.20))
            simplified_xy = _linefit_subset_simplify_closed(
                np.asarray(outline_xy, dtype=np.float64), perp_tol=perp_tol
            )
            if len(simplified_xy) >= 3:
                new_poly, _ = polygon_from_outline(simplified_xy)
                if new_poly is not None and not new_poly.is_empty and new_poly.area > 0:
                    polygon = new_poly
                    outline_xy = outline_from_polygon(polygon)

        if len(outline_xy) >= 4:
            simplify_tol = float(self.config.get("sidewalk_simplify_tol", self.fused_outline_simplify_tol))
            smooth_iterations = int(self.config.get("sidewalk_smooth_iterations", self.fused_outline_smooth_iterations))
            min_seg_len = float(self.config.get("sidewalk_min_seg_len", self.fused_outline_min_seg_len))
            post_simplify_tol = float(self.config.get("sidewalk_post_simplify_tol", self.fused_outline_post_simplify_tol))
            close_buf = float(self.config.get("sidewalk_close_buffer", self.fused_outline_close_buffer))
            open_buf = float(self.config.get("sidewalk_open_buffer", self.fused_outline_open_buffer))

            # Vector morphology: close fills concavities (alpha-shape teeth),
            # open shaves single-point spikes. Both before any line-simplify
            # so the simplifier sees a clean boundary.
            try:
                if close_buf > 0:
                    closed = polygon.buffer(close_buf).buffer(-close_buf)
                    if not closed.is_empty:
                        if isinstance(closed, MultiPolygon):
                            closed = max(closed.geoms, key=lambda g: g.area)
                        polygon = closed
                if open_buf > 0:
                    opened = polygon.buffer(-open_buf).buffer(open_buf)
                    if not opened.is_empty:
                        if isinstance(opened, MultiPolygon):
                            opened = max(opened.geoms, key=lambda g: g.area)
                        polygon = opened
                outline_xy = outline_from_polygon(polygon)
            except Exception as e:
                print(f"update_track_metadata: buffer regularize failed: {e}")

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

        # Snap each outline vertex to its nearest accumulated LiDAR sidewalk
        # point so every reported vertex is a real measured point (and z is
        # the LiDAR-measured z, not an interpolation). Vertices that survive
        # alpha-shape are usually already LiDAR points; bridge/union may
        # introduce a few synthetic ones — those get snapped here.
        snap_to_lidar = bool(self.config.get("sidewalk_snap_to_lidar", True))

        xy_parts, z_parts = [], []
        for arr, zarr in zip(track["_outline_arrays"], track.get("_outline_z_arrays", [])):
            xy = as_xy(arr)
            zv = np.asarray(zarr, dtype=np.float32).ravel()
            if len(xy) > 0 and len(zv) == len(xy):
                xy_parts.append(xy)
                z_parts.append(zv)

        if snap_to_lidar and xy_parts and len(outline_xy) > 0:
            from scipy.spatial import cKDTree
            src_xy = np.vstack(xy_parts)
            src_z  = np.concatenate(z_parts)
            tree   = cKDTree(src_xy)
            dist, idx = tree.query(outline_xy, k=1)
            # Only snap vertices that are within `snap_radius` of a LiDAR point.
            # Vertices farther than that (e.g. bridge interior points filling a
            # gap with no LiDAR support) keep their original geometry-derived
            # position to preserve polygon topology.
            snap_radius = float(self.config.get("sidewalk_snap_radius", 0.4))
            within = dist < snap_radius
            snapped_xy = outline_xy.astype(np.float64).copy()
            snapped_xy[within] = src_xy[idx[within]]
            # z: snapped vertices use the LiDAR z, others interpolate
            snapped_z = np.full(len(outline_xy), np.nan, dtype=np.float64)
            snapped_z[within] = src_z[idx[within]]
            if (~within).any():
                snapped_z[~within] = _interp_z_for_outline(
                    src_xy, src_z, outline_xy[~within]
                )
            # Drop consecutive duplicates after snap.
            keep = [0]
            for k in range(1, len(snapped_xy)):
                if not np.allclose(snapped_xy[k], snapped_xy[keep[-1]], atol=1e-6):
                    keep.append(k)
            snapped_xy = snapped_xy[keep]
            snapped_z  = snapped_z[keep]
            new_poly, _ = polygon_from_outline(snapped_xy)
            if new_poly is not None and not new_poly.is_empty and new_poly.area > 0:
                polygon = new_poly
                outline_xy = snapped_xy.astype(np.float32)
                track["outline_z"] = snapped_z.astype(np.float32).tolist()
            else:
                track["outline_z"] = _interp_z_for_outline(src_xy, src_z, outline_xy).tolist()
        elif xy_parts and len(outline_xy) > 0:
            src_xy = np.vstack(xy_parts)
            src_z  = np.concatenate(z_parts)
            track["outline_z"] = _interp_z_for_outline(src_xy, src_z, outline_xy).tolist()
        else:
            fallback_z = float(np.mean(track["_z_values"])) if track["_z_values"] else 0.0
            track["outline_z"] = [fallback_z] * len(outline_xy)

        track["_polygon"] = polygon
        track["outline"] = outline_xy.tolist() if hasattr(outline_xy, 'tolist') else list(outline_xy)
        track["area"] = float(polygon.area) if polygon is not None else 0.0

        if xy_parts:
            merged_points = np.vstack(xy_parts)
            track["centroid"] = merged_points.mean(axis=0).astype(np.float32).tolist()
        else:
            track["centroid"] = [0.0, 0.0]

        if track["_z_values"]:
            track["sidewalk_z"] = float(np.mean(track["_z_values"]))

    def _merge_contained_or_overlapping_tracks(self, tracks):
        """Pairwise post-fusion sweep.

        Merges track pairs that:
        - one contains the other,
        - intersect by ≥ threshold of the smaller area, OR
        - sit within `bridge_gap` metres of each other (so adjacent sidewalk
          segments separated by a small unmapped gap — e.g. a corner the
          vehicle didn't cover — get unified instead of being treated as two
          tracks).

        Tunables:
          sidewalk_pairwise_merge_min_overlap_ratio  (default 0.10)
          sidewalk_track_bridge_gap                  (default 0.0; 0 = off)
        """
        if len(tracks) < 2:
            return tracks

        threshold  = float(self.config.get("sidewalk_pairwise_merge_min_overlap_ratio", 0.10))
        bridge_gap = float(self.config.get("sidewalk_track_bridge_gap", 0.0))
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tracks):
                p_i = tracks[i].get("_polygon")
                if p_i is None or p_i.is_empty:
                    i += 1
                    continue
                bx_i = p_i.bounds  # (minx, miny, maxx, maxy)
                j = i + 1
                while j < len(tracks):
                    p_j = tracks[j].get("_polygon")
                    if p_j is None or p_j.is_empty:
                        j += 1
                        continue
                    # Cheap bbox reject before the expensive shapely calls.
                    # Inflate bbox by bridge_gap so almost-touching pairs still
                    # reach the gap-based merge path below.
                    bx_j = p_j.bounds
                    if (bx_i[2] + bridge_gap < bx_j[0] or bx_j[2] + bridge_gap < bx_i[0]
                            or bx_i[3] + bridge_gap < bx_j[1] or bx_j[3] + bridge_gap < bx_i[1]):
                        j += 1
                        continue
                    try:
                        contains = p_i.contains(p_j) or p_j.contains(p_i)
                        inter_area = float(p_i.intersection(p_j).area)
                    except Exception:
                        j += 1
                        continue
                    min_area = float(min(p_i.area, p_j.area))
                    overlap_ratio = inter_area / min_area if min_area > 1e-6 else 0.0
                    near = (bridge_gap > 0.0 and p_i.distance(p_j) <= bridge_gap)
                    if not (contains or overlap_ratio >= threshold or near):
                        j += 1
                        continue

                    if near and not (contains or overlap_ratio >= threshold):
                        from shapely.ops import nearest_points
                        from shapely.geometry import LineString, Point
                        bw = float(self.config.get("sidewalk_track_bridge_width", 1.0))
                        a, b = nearest_points(p_i, p_j)
                        ax, ay = a.x, a.y; bx, by = b.x, b.y
                        dx, dy = bx - ax, by - ay
                        L = (dx*dx + dy*dy) ** 0.5
                        if L > 1e-6:
                            ex, ey = dx / L, dy / L
                            ext = bw * 0.5 + 0.2
                            a2 = Point(ax - ex * ext, ay - ey * ext)
                            b2 = Point(bx + ex * ext, by + ey * ext)
                            bridge = LineString([a2, b2]).buffer(bw * 0.5, cap_style=2)
                            merged = p_i.union(p_j).union(bridge)
                        else:
                            merged = p_i.union(p_j)
                    else:
                        merged = p_i.union(p_j)
                    if isinstance(merged, MultiPolygon):
                        merged = max(merged.geoms, key=lambda geom: geom.area)
                    tracks[i]["_polygon"] = merged
                    tracks[i]["_outline_arrays"].extend(tracks[j].get("_outline_arrays", []))
                    if tracks[j].get("_outline_z_arrays"):
                        tracks[i].setdefault("_outline_z_arrays", []).extend(tracks[j]["_outline_z_arrays"])
                    tracks[i]["_z_values"].extend(tracks[j].get("_z_values", []))
                    tracks[i]["source_indices"].extend(tracks[j].get("source_indices", []))
                    tracks[i]["outlines"].extend(tracks[j].get("outlines", []))
                    # Concatenate accumulated LiDAR points (the whole point of
                    # this refactor — alpha-shape on combined points naturally
                    # bridges the gap without synthetic vertices).
                    pi_pts = tracks[i].get("_input_points", np.zeros((0, 3), dtype=np.float32))
                    pj_pts = tracks[j].get("_input_points", np.zeros((0, 3), dtype=np.float32))
                    if len(pi_pts) or len(pj_pts):
                        tracks[i]["_input_points"] = np.vstack([
                            np.asarray(pi_pts, dtype=np.float32),
                            np.asarray(pj_pts, dtype=np.float32),
                        ])
                    tracks.pop(j)
                    p_i = merged
                    changed = True
                i += 1
        return tracks

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
            oz = np.asarray(record.get("outline_z", [float(record["sidewalk_z"])] * len(outline_xy)), dtype=np.float32)
            matched_track["_outline_z_arrays"].append(oz)
            matched_track["_z_values"].append(float(record["sidewalk_z"]))
            matched_track["source_indices"].append(int(record["index"]))
            matched_track["outlines"].append(outline_xy.tolist())
            cp = record.get("_cluster_points")
            if cp is not None and len(cp) > 0:
                prev = matched_track.get("_input_points", np.zeros((0, 3), dtype=np.float32))
                matched_track["_input_points"] = np.vstack([
                    np.asarray(prev, dtype=np.float32),
                    np.asarray(cp, dtype=np.float32),
                ])
            # Phase 1.3: smoothing/simplify chain runs once per track at the
            # end, not after every merge.

        # Pairwise contains/overlap consolidation runs before per-track
        # metadata so the final outline reflects the merged extent.
        tracks = self._merge_contained_or_overlapping_tracks(tracks)

        for track in tracks:
            self.update_track_metadata(track)
            for key in ("_polygon", "_outline_arrays", "_outline_z_arrays", "_z_values", "_input_points"):
                track.pop(key, None)

        for track in tracks:
            track["source_indices"] = sorted(set(int(v) for v in track.get("source_indices", [])))
        tracks.sort(key=lambda item: len(item["source_indices"]), reverse=True)
        return tracks

    def finalize(self):
        # Strip non-serializable underscore-prefixed fields (notably
        # `_cluster_points` numpy arrays) from records before the parent
        # class writes them to JSON.
        stripped = []
        for r in self.records:
            stripped.append({k: v for k, v in r.items() if not k.startswith("_")})
        backup = self.records
        self.records = stripped
        try:
            super().finalize()
        finally:
            self.records = backup

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
