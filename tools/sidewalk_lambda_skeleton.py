"""Minimal lambda-medial-axis skeleton for a sidewalk polygon.

Pipeline:
  1. Topology cleanup: shapely buffer(+eps/2).buffer(-eps/2) to weld small
     boundary glitches.
  2. Dense boundary samples -> scipy.spatial.Voronoi; keep Voronoi edges
     whose BOTH endpoints lie strictly inside the cleaned polygon.
  3. For each kept edge, lambda = polygon.exterior.distance(midpoint)
     (local half-width). Prune edges with lambda < lambda_min.

Outputs a BEV PNG showing:
  - cleaned polygon outline
  - raw MA edges (faint grey)
  - lambda-MA edges colored by lambda (cool = thin, warm = wide)
  - LiDAR sidewalk points (optional underlay)
plus a JSON sidecar with the polyline edges.
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, MultiPolygon


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True,
                   help="Path to sidewalk_records_*.json (uses sidewalks[i].outline).")
    p.add_argument("--sidewalk-index", type=int, default=0,
                   help="Which fused sidewalk to skeletonize (0-based).")
    p.add_argument("--output", required=True,
                   help="Output PNG; JSON sidecar will be written alongside.")
    p.add_argument("--pkl", default=None,
                   help="Optional outdoor.pkl for LiDAR underlay.")
    p.add_argument("--sidewalk-class", type=int, default=15)
    p.add_argument("--cleanup-eps-m", type=float, default=0.30,
                   help="Topology-cleanup buffer radius. ~0.15-0.3 of typical width.")
    p.add_argument("--boundary-sample-step-m", type=float, default=0.15,
                   help="Boundary samples spacing for Voronoi.")
    p.add_argument("--lambda-min-m", type=float, default=0.60,
                   help="Prune MA edges whose midpoint half-width < this.")
    p.add_argument("--bev-res-m", type=float, default=0.05)
    p.add_argument("--margin-m", type=float, default=2.0)
    p.add_argument("--pose", default=None,
                   help="pose.csv. If set, enable trajectory-direction pruning "
                        "of MA edges (思路 4): edges whose direction makes an "
                        "angle > --traj-angle-max-deg with the nearest pose's "
                        "forward axis are dropped.")
    p.add_argument("--traj-angle-max-deg", type=float, default=30.0,
                   help="Max angle (deg) between MA edge and nearest pose's "
                        "forward direction.")
    p.add_argument("--extract-trunk", action="store_true",
                   help="After lambda+traj pruning, extract longest weighted "
                        "path in the kept-edge graph as the centerline trunk.")
    p.add_argument("--resample-spacing-m", type=float, default=0.0,
                   help="If > 0, resample trunk by arclength at this spacing (m). "
                        "0 disables (raw trunk vertices kept).")
    return p.parse_args()


def quat_to_rotmat(qx, qy, qz, qw):
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_forward_xy(pose_row):
    """Vehicle +X (forward) axis projected into ENU XY, normalized."""
    R = quat_to_rotmat(*pose_row[3:7])
    fx = R[:2, 0]
    n = np.linalg.norm(fx)
    return fx / n if n > 1e-9 else np.array([1.0, 0.0])


def traj_prune_edges(edges, pose_xy, pose_fwd, angle_max_deg):
    """Keep edges whose direction is within angle_max_deg of nearest pose's
    forward axis. Returns (kept, dropped)."""
    from scipy.spatial import cKDTree
    if len(pose_xy) == 0:
        return list(edges), []
    tree = cKDTree(pose_xy)
    cos_thr = float(np.cos(np.radians(angle_max_deg)))
    kept, dropped = [], []
    for a, b, lam in edges:
        d = np.asarray(b) - np.asarray(a)
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        d_hat = d / norm
        mid = (np.asarray(a) + np.asarray(b)) * 0.5
        _, idx = tree.query(mid, k=1)
        cos_a = abs(float(np.dot(d_hat, pose_fwd[idx])))
        cos_a = min(1.0, cos_a)
        if cos_a >= cos_thr:
            kept.append((a, b, lam))
        else:
            dropped.append((a, b, lam))
    return kept, dropped


def build_graph(edges, tol=1e-3):
    """Map (a,b) edges to integer-indexed nodes + adjacency list."""
    coord_to_idx = {}
    nodes = []
    adj = []   # adj[i] = list of (j, length)

    def key(p):
        return (round(float(p[0]) / tol), round(float(p[1]) / tol))

    def get_node(p):
        k = key(p)
        idx = coord_to_idx.get(k)
        if idx is None:
            idx = len(nodes)
            coord_to_idx[k] = idx
            nodes.append((float(p[0]), float(p[1])))
            adj.append([])
        return idx

    for a, b, _lam in edges:
        ia = get_node(a)
        ib = get_node(b)
        if ia == ib:
            continue
        L = float(np.hypot(nodes[ib][0] - nodes[ia][0], nodes[ib][1] - nodes[ia][1]))
        adj[ia].append((ib, L))
        adj[ib].append((ia, L))
    return nodes, adj


def farthest_node(adj, src):
    """Dijkstra from src; return (farthest_idx, dist_array, prev_array)."""
    import heapq
    n = len(adj)
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    far = max(range(n), key=lambda i: -1.0 if dist[i] == INF else dist[i])
    return far, dist, prev


def resample_polyline_by_arclength(poly_xy, spacing):
    """Resample polyline at uniform arclength `spacing` (meters). Returns list
    of (x, y) with endpoints preserved."""
    pts = np.asarray(poly_xy, dtype=np.float64)
    if len(pts) < 2 or spacing <= 0:
        return [tuple(p) for p in pts]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= spacing:
        return [tuple(pts[0]), tuple(pts[-1])]
    n_seg = int(np.ceil(total / spacing))
    targets = np.linspace(0.0, total, n_seg + 1)
    out = []
    for t in targets:
        idx = np.searchsorted(s, t, side="right") - 1
        idx = max(0, min(len(seg) - 1, idx))
        if seg[idx] < 1e-9:
            out.append(tuple(pts[idx]))
            continue
        alpha = (t - s[idx]) / seg[idx]
        alpha = max(0.0, min(1.0, alpha))
        p = pts[idx] + alpha * (pts[idx + 1] - pts[idx])
        out.append((float(p[0]), float(p[1])))
    return out


def connected_components(adj):
    n = len(adj)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s] or not adj[s]:
            seen[s] = True
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def longest_path_dijkstra(adj):
    """Across all connected components: double-Dijkstra to find longest weighted
    path. Returns (best_path, best_length).
    On a tree this gives the diameter; on near-tree-with-small-cycles it's a
    good approximation."""
    if not adj:
        return [], 0.0
    comps = connected_components(adj)
    best_path, best_len = [], 0.0
    for comp in comps:
        if len(comp) < 2:
            continue
        a, _, _ = farthest_node(adj, comp[0])
        b, dist, prev = farthest_node(adj, a)
        # reconstruct
        path = []
        cur = b
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        L = float(dist[b])
        if L > best_len:
            best_len = L
            best_path = path
    return best_path, best_len


def topo_cleanup(poly_xy, eps):
    poly = Polygon(poly_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    cleaned = poly.buffer(eps / 2.0).buffer(-eps / 2.0)
    if cleaned.is_empty:
        return poly
    if isinstance(cleaned, MultiPolygon):
        cleaned = max(cleaned.geoms, key=lambda g: g.area)
    # Drop small holes.
    holes_kept = [list(h.coords) for h in cleaned.interiors
                  if Polygon(h).area > eps * eps]
    return Polygon(cleaned.exterior, holes_kept) if holes_kept else Polygon(cleaned.exterior)


def sample_boundary(poly, step):
    """Densely resample polygon boundary (exterior + holes) at given spacing."""
    out = []
    rings = [poly.exterior] + list(poly.interiors)
    for ring in rings:
        L = ring.length
        n = max(8, int(np.ceil(L / step)))
        for k in range(n):
            p = ring.interpolate(k * L / n)
            out.append([p.x, p.y])
    return np.asarray(out, dtype=np.float64)


def medial_axis(poly, boundary_samples, lambda_min):
    """Compute lambda-MA edges. Returns:
       raw_edges: list of (p1, p2, lambda)  (all edges with endpoints inside poly)
       kept_edges: list of (p1, p2, lambda) with lambda >= lambda_min
    """
    vor = Voronoi(boundary_samples)
    boundary = poly.exterior
    raw, kept = [], []
    # Pre-prep a slightly-shrunk polygon for "strictly inside" test
    poly_strict = poly.buffer(-1e-3)
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        a = vor.vertices[ridge[0]]
        b = vor.vertices[ridge[1]]
        pa, pb = Point(a), Point(b)
        if not (poly_strict.contains(pa) and poly_strict.contains(pb)):
            continue
        mid = (a + b) * 0.5
        lam = boundary.distance(Point(mid))
        for h in poly.interiors:
            lam = min(lam, h.distance(Point(mid)))
        raw.append((a, b, lam))
        if lam >= lambda_min:
            kept.append((a, b, lam))
    return raw, kept


def world_to_canvas(pts_xy, x_min, y_min, res, h_px):
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    col = ((pts[:, 0] - x_min) / res).astype(np.int32)
    row = (h_px - 1 - (pts[:, 1] - y_min) / res).astype(np.int32)
    return np.column_stack([col, row])


def lam_color(lam, lam_max):
    """Cool-warm: thin = blue, wide = red. Returns BGR."""
    t = float(np.clip(lam / max(lam_max, 1e-6), 0, 1))
    if t < 0.5:
        a = t * 2
        return (255, int(255 * a), int(255 * (1 - a)))
    a = (t - 0.5) * 2
    return (int(255 * (1 - a)), int(255 * (1 - a)), int(255 * a + 0))


def main():
    args = parse_args()

    records = json.load(open(args.records))
    sidewalks = records.get("sidewalks", [])
    if not sidewalks:
        raise SystemExit("no sidewalks in records")
    sw = sidewalks[int(args.sidewalk_index)]
    outline = np.asarray(sw["outline"], dtype=np.float64)
    print(f"loaded sidewalk[{args.sidewalk_index}]: {len(outline)} vertices, area={sw.get('area', 0):.1f} m^2")

    # Stage 1: topology cleanup
    poly = topo_cleanup(outline, float(args.cleanup_eps_m))
    print(f"after cleanup: ext_verts={len(poly.exterior.coords)} holes={len(poly.interiors)} area={poly.area:.1f}")

    # Stage 2: boundary samples + Voronoi
    samples = sample_boundary(poly, float(args.boundary_sample_step_m))
    print(f"boundary samples: {len(samples)}")

    # Stage 3: medial axis + lambda prune
    raw_edges, kept_edges = medial_axis(poly, samples, float(args.lambda_min_m))
    print(f"raw MA edges: {len(raw_edges)}  kept (lambda>={args.lambda_min_m}m): {len(kept_edges)}")
    if kept_edges:
        lams = [e[2] for e in kept_edges]
        print(f"kept lambda: min={min(lams):.2f} med={np.median(lams):.2f} max={max(lams):.2f} m")

    # Stage 4 (思路 4): trajectory-direction pruning
    pose_xy = np.empty((0, 2)); pose_fwd = np.empty((0, 2))
    traj_dropped = []
    if args.pose:
        poses_arr = np.loadtxt(args.pose, delimiter=",")
        if poses_arr.ndim == 1:
            poses_arr = poses_arr.reshape(1, -1)
        pose_xy = poses_arr[:, :2]
        pose_fwd = np.array([pose_forward_xy(r) for r in poses_arr])
        traj_kept, traj_dropped = traj_prune_edges(
            kept_edges, pose_xy, pose_fwd, float(args.traj_angle_max_deg))
        print(f"traj prune (<= {args.traj_angle_max_deg}°): "
              f"keep {len(traj_kept)} / drop {len(traj_dropped)}")
        kept_edges = traj_kept

    # Stage 5 (思路 2): longest-path trunk extraction.
    trunk_poly_xy = []
    if args.extract_trunk and kept_edges:
        nodes, adj = build_graph(kept_edges)
        comps = connected_components(adj)
        print(f"graph: {sum(1 for c in comps)} components, sizes={sorted([len(c) for c in comps], reverse=True)[:5]}")
        path_idx, trunk_len = longest_path_dijkstra(adj)
        trunk_poly_xy = [nodes[i] for i in path_idx]
        print(f"trunk: {len(trunk_poly_xy)} nodes, length={trunk_len:.2f} m")
        if float(args.resample_spacing_m) > 0 and len(trunk_poly_xy) >= 2:
            trunk_poly_xy = resample_polyline_by_arclength(
                trunk_poly_xy, float(args.resample_spacing_m))
            print(f"trunk resampled @ {args.resample_spacing_m}m: {len(trunk_poly_xy)} nodes")

    # ---------- BEV canvas ----------
    xy = np.array(list(poly.exterior.coords))
    margin = float(args.margin_m)
    res = float(args.bev_res_m)
    x_min = xy[:, 0].min() - margin
    y_min = xy[:, 1].min() - margin
    x_max = xy[:, 0].max() + margin
    y_max = xy[:, 1].max() + margin
    grid_w = int(np.ceil((x_max - x_min) / res))
    grid_h = int(np.ceil((y_max - y_min) / res))
    img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Polygon fill (dark grey) + outline (light grey)
    ext_pix = world_to_canvas(np.array(poly.exterior.coords), x_min, y_min, res, grid_h)
    cv2.fillPoly(img, [ext_pix], (45, 45, 45))
    cv2.polylines(img, [ext_pix], True, (180, 180, 180), 1, cv2.LINE_AA)
    for h in poly.interiors:
        h_pix = world_to_canvas(np.array(h.coords), x_min, y_min, res, grid_h)
        cv2.fillPoly(img, [h_pix], (0, 0, 0))
        cv2.polylines(img, [h_pix], True, (120, 120, 120), 1, cv2.LINE_AA)

    # Optional LiDAR underlay
    if args.pkl:
        frames = []
        with open(args.pkl, "rb") as f:
            while True:
                try:
                    frames.append(np.asarray(pickle.load(f), dtype=np.float64))
                except EOFError:
                    break
        all_pts = []
        for arr in frames:
            if arr is None or len(arr) == 0:
                continue
            m = arr[:, 3].astype(np.int64) == int(args.sidewalk_class)
            if m.any():
                all_pts.append(arr[m, :2])
        if all_pts:
            pts = np.vstack(all_pts)
            pix = world_to_canvas(pts, x_min, y_min, res, grid_h)
            ok = ((pix[:, 0] >= 0) & (pix[:, 0] < grid_w)
                  & (pix[:, 1] >= 0) & (pix[:, 1] < grid_h))
            img[pix[ok, 1], pix[ok, 0]] = (80, 80, 0)
        print(f"underlay: {sum(len(p) for p in all_pts)} sidewalk LiDAR points")

    # Raw MA edges (faint grey)
    for a, b, _ in raw_edges:
        pa = world_to_canvas([a], x_min, y_min, res, grid_h)[0]
        pb = world_to_canvas([b], x_min, y_min, res, grid_h)[0]
        cv2.line(img, tuple(pa), tuple(pb), (90, 90, 90), 1, cv2.LINE_AA)

    # Trajectory pruning dropped edges (orange, thinner)
    for a, b, _ in traj_dropped:
        pa = world_to_canvas([a], x_min, y_min, res, grid_h)[0]
        pb = world_to_canvas([b], x_min, y_min, res, grid_h)[0]
        cv2.line(img, tuple(pa), tuple(pb), (0, 140, 255), 1, cv2.LINE_AA)

    # Kept MA edges colored by lambda
    if kept_edges:
        lam_max = float(max(e[2] for e in kept_edges))
        for a, b, lam in kept_edges:
            pa = world_to_canvas([a], x_min, y_min, res, grid_h)[0]
            pb = world_to_canvas([b], x_min, y_min, res, grid_h)[0]
            cv2.line(img, tuple(pa), tuple(pb), lam_color(lam, lam_max), 2, cv2.LINE_AA)

    # Trunk (bright yellow polyline + dots) drawn AFTER kept edges so it's on top
    if trunk_poly_xy and len(trunk_poly_xy) >= 2:
        trunk_pix = world_to_canvas(np.array(trunk_poly_xy), x_min, y_min, res, grid_h)
        for k in range(len(trunk_pix) - 1):
            cv2.line(img, tuple(trunk_pix[k]), tuple(trunk_pix[k+1]),
                     (0, 255, 255), 3, cv2.LINE_AA)
        for u, v in trunk_pix:
            cv2.circle(img, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)

    # Trajectory overlay (green dots + arrows every 10th pose)
    if len(pose_xy) > 0:
        pose_pix = world_to_canvas(pose_xy, x_min, y_min, res, grid_h)
        ok = ((pose_pix[:, 0] >= 0) & (pose_pix[:, 0] < grid_w)
              & (pose_pix[:, 1] >= 0) & (pose_pix[:, 1] < grid_h))
        for u, v in pose_pix[ok]:
            cv2.circle(img, (int(u), int(v)), 1, (0, 220, 0), -1, cv2.LINE_AA)
        arrow_step = max(1, len(pose_xy) // 30)
        for k in range(0, len(pose_xy), arrow_step):
            if not ok[k]:
                continue
            tip = pose_xy[k] + pose_fwd[k] * 1.0  # 1m arrow
            tip_pix = world_to_canvas([tip], x_min, y_min, res, grid_h)[0]
            cv2.arrowedLine(img, tuple(pose_pix[k]), tuple(tip_pix),
                            (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)

    traj_info = f"  trajprune<={args.traj_angle_max_deg}deg drop={len(traj_dropped)}" if args.pose else ""
    hdr = (f"polygon[{args.sidewalk_index}]  cleanup eps={args.cleanup_eps_m}m  "
           f"bsample={args.boundary_sample_step_m}m  lambda_min={args.lambda_min_m}m  "
           f"raw={len(raw_edges)} kept={len(kept_edges)}{traj_info}")
    cv2.putText(img, hdr, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, hdr, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    js = out_path.with_suffix(".json")
    js.write_text(json.dumps({
        "sidewalk_index": int(args.sidewalk_index),
        "cleanup_eps_m": float(args.cleanup_eps_m),
        "boundary_sample_step_m": float(args.boundary_sample_step_m),
        "lambda_min_m": float(args.lambda_min_m),
        "bev_res_m": float(res),
        "x_min": float(x_min), "y_min": float(y_min),
        "grid_w": int(grid_w), "grid_h": int(grid_h),
        "n_raw_edges": int(len(raw_edges)),
        "n_kept_edges": int(len(kept_edges)),
        "trunk_polyline": [[float(p[0]), float(p[1])] for p in trunk_poly_xy],
        "kept_edges": [
            {"a": [float(a[0]), float(a[1])], "b": [float(b[0]), float(b[1])],
             "lambda_m": float(lam)} for a, b, lam in kept_edges
        ],
    }, indent=2))
    print(f"wrote {out_path} ({grid_w}x{grid_h}); meta {js}")


if __name__ == "__main__":
    main()
