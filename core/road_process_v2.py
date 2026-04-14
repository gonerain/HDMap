import numpy as np

from core.road_process import RoadEdgeProcess
from core.vector_common import cluster_labels


class RoadEdgeProcessV2(RoadEdgeProcess):
    name = "road_v2"

    def __init__(self, args, config):
        super().__init__(args, config)
        # v2 defaults: larger step + adaptive clustering + geometry-first topology with gap bridge
        self.step = max(int(config.get("road_v2_step", config.get("road_step", 20))), 1)
        self.adaptive_density_ref = float(config.get("road_v2_density_ref", 220.0))
        self.adaptive_eps_scale = float(config.get("road_v2_eps_scale", 1.15))
        self.adaptive_min_samples_floor = int(config.get("road_v2_min_samples_floor", 10))
        self.adaptive_max_clusters = int(config.get("road_v2_max_clusters", max(self.max_clusters, 4)))
        self.adaptive_far_dist = float(config.get("road_v2_far_dist", 45.0))
        self.adaptive_far_max_clusters = int(config.get("road_v2_far_max_clusters", 6))

        self.topo_geom_dist_max = float(config.get("road_v2_topology_geom_dist_max", 9.0))
        self.topo_geom_dist_gap_scale = float(config.get("road_v2_topology_geom_dist_gap_scale", 0.12))
        self.topo_heading_cos_min = float(config.get("road_v2_topology_heading_cos_min", 0.45))
        self.topo_bridge_max_index_gap = int(config.get("road_v2_topology_bridge_max_index_gap", self.step * 8))

    def _dynamic_cluster_params(self, points_xyz):
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        if len(points_xyz) == 0:
            return self.cluster_eps, self.cluster_min_samples, self.max_clusters

        center = points_xyz[:, :2].mean(axis=0)
        radius = np.linalg.norm(points_xyz[:, :2] - center[None, :], axis=1)
        area = float(np.pi * max(np.percentile(radius, 75), 1.0) ** 2)
        density = float(len(points_xyz)) / max(area, 1e-3)

        density_ratio = np.sqrt(max(self.adaptive_density_ref / max(density, 1e-3), 0.2))
        dyn_eps = float(np.clip(self.cluster_eps * density_ratio * self.adaptive_eps_scale, 0.6, 2.2))
        dyn_min_samples = int(np.clip(round(self.cluster_min_samples / max(density_ratio, 1e-3)), self.adaptive_min_samples_floor, 64))

        far_ratio = float(np.mean(radius > self.adaptive_far_dist))
        if far_ratio > 0.25:
            dyn_max_clusters = max(self.adaptive_max_clusters, self.adaptive_far_max_clusters)
        else:
            dyn_max_clusters = self.adaptive_max_clusters

        return dyn_eps, dyn_min_samples, dyn_max_clusters

    def _select_road_points(self, current_points_xyz, center_xy):
        points = np.asarray(current_points_xyz, dtype=np.float32)
        if len(points) == 0:
            return points

        dyn_eps, dyn_min_samples, dyn_max_clusters = self._dynamic_cluster_params(points)
        labels = cluster_labels(points[:, :2], eps=dyn_eps, min_samples=dyn_min_samples)
        valid_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
        if not valid_labels:
            return points

        candidates = []
        for label in valid_labels:
            cluster = points[labels == label, :3]
            if len(cluster) < max(self.min_cluster_points, dyn_min_samples):
                continue
            centroid = cluster[:, :2].mean(axis=0)
            center_dist = float(np.linalg.norm(centroid - center_xy))
            if center_dist > self.max_cluster_center_dist:
                continue

            # Favor larger and spatially-distributed clusters to reduce truncation risk.
            spread = float(np.linalg.norm(np.std(cluster[:, :2], axis=0)))
            score = float(len(cluster)) + 8.0 * spread - 0.25 * center_dist
            candidates.append((score, cluster))

        if not candidates:
            return points

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [cluster for _, cluster in candidates[: max(dyn_max_clusters, 1)]]
        return np.vstack(selected) if selected else points

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

        # Build sequential edges; connectivity is geometry-first with gap-aware distance allowance.
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

            extra_gap = max(idx_gap - self.step, 0)
            geom_dist_limit = self.topo_geom_dist_max + self.topo_geom_dist_gap_scale * extra_gap
            geom_connected = (seg_len <= geom_dist_limit) and (heading_cos >= self.topo_heading_cos_min)
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
                    "geom_dist_limit": float(geom_dist_limit),
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
                "geom_dist_max": self.topo_geom_dist_max,
                "geom_dist_gap_scale": self.topo_geom_dist_gap_scale,
                "heading_cos_min": self.topo_heading_cos_min,
                "bridge_max_index_gap": self.topo_bridge_max_index_gap,
            },
        }
