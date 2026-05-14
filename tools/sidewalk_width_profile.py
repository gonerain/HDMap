"""Raw width measurement along a sidewalk trunk via normal-ray casting.

For each trunk vertex p_i:
  1. Tangent t_i via centered difference (forward/backward at endpoints).
  2. Normal n_i = rotate(t_i, +90 deg)  ->  "left", -n_i  ->  "right".
  3. Cast ray from p_i along +n_i, intersect with polygon boundary,
     take the nearest hit -> L_i, d_L = |L_i - p_i|.
  4. Same with -n_i -> R_i, d_R.
  5. w_i = d_L + d_R.

Outputs:
  - PNG: left panel BEV with polygon + trunk + per-node L/R rays + hit
    points + width labels; right panel width profile chart (d_L, d_R, w
    as functions of arclength s).
  - JSON sidecar: per-node measurement records.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trunk-json", required=True,
                   help="Output JSON from sidewalk_lambda_skeleton.py (must contain trunk_polyline).")
    p.add_argument("--records", required=True,
                   help="sidewalk_records_*.json (uses sidewalks[i].outline).")
    p.add_argument("--sidewalk-index", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--ray-length-m", type=float, default=10.0,
                   help="Cap for normal-ray length.")
    p.add_argument("--bev-res-m", type=float, default=0.05)
    p.add_argument("--margin-m", type=float, default=2.0)
    p.add_argument("--label-every", type=int, default=3,
                   help="Show width label every Nth node.")
    return p.parse_args()


def load_polygon(records_path, idx):
    d = json.load(open(records_path))
    sw = d["sidewalks"][int(idx)]
    outline = np.asarray(sw["outline"], dtype=np.float64)
    poly = Polygon(outline)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def load_trunk(trunk_json_path):
    d = json.load(open(trunk_json_path))
    return np.asarray(d["trunk_polyline"], dtype=np.float64)


def compute_tangent(trunk):
    n = len(trunk)
    t = np.zeros_like(trunk)
    for i in range(n):
        if i == 0:
            d = trunk[i + 1] - trunk[i]
        elif i == n - 1:
            d = trunk[i] - trunk[i - 1]
        else:
            d = trunk[i + 1] - trunk[i - 1]
        norm = np.linalg.norm(d)
        t[i] = d / max(norm, 1e-9)
    return t


def measure_normal(p, n_vec, polygon, ray_length):
    """Cast a ray from p along n_vec, return (distance, hit_xy) or (None, None)."""
    p_end = p + n_vec * float(ray_length)
    ray = LineString([tuple(p), tuple(p_end)])
    inter = polygon.boundary.intersection(ray)
    if inter.is_empty:
        return None, None
    if isinstance(inter, Point):
        hit = np.array([inter.x, inter.y])
        return float(np.linalg.norm(hit - p)), hit
    coords = []
    geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    for g in geoms:
        if isinstance(g, Point):
            coords.append([g.x, g.y])
        elif isinstance(g, LineString):
            for c in g.coords:
                coords.append([c[0], c[1]])
    if not coords:
        return None, None
    coords = np.asarray(coords, dtype=np.float64)
    dists = np.linalg.norm(coords - p, axis=1)
    k = int(np.argmin(dists))
    return float(dists[k]), coords[k]


def world_to_canvas(pts_xy, x_min, y_min, res, h_px):
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    col = ((pts[:, 0] - x_min) / res).astype(np.int32)
    row = (h_px - 1 - (pts[:, 1] - y_min) / res).astype(np.int32)
    return np.column_stack([col, row])


def main():
    args = parse_args()
    trunk = load_trunk(args.trunk_json)
    poly = load_polygon(args.records, int(args.sidewalk_index))
    if len(trunk) < 2:
        raise SystemExit("trunk too short")

    tangents = compute_tangent(trunk)
    measurements = []
    cum_s = 0.0
    for i, (p, t) in enumerate(zip(trunk, tangents)):
        if i > 0:
            cum_s += float(np.linalg.norm(trunk[i] - trunk[i - 1]))
        n_left = np.array([-t[1], t[0]])
        n_right = -n_left
        d_L, hit_L = measure_normal(p, n_left, poly, args.ray_length_m)
        d_R, hit_R = measure_normal(p, n_right, poly, args.ray_length_m)
        w = (d_L + d_R) if (d_L is not None and d_R is not None) else None
        measurements.append({
            "i": int(i),
            "s_m": float(cum_s),
            "p_xy": [float(p[0]), float(p[1])],
            "tangent": [float(t[0]), float(t[1])],
            "d_L_m": d_L,
            "d_R_m": d_R,
            "w_m": w,
            "L_xy": [float(hit_L[0]), float(hit_L[1])] if hit_L is not None else None,
            "R_xy": [float(hit_R[0]), float(hit_R[1])] if hit_R is not None else None,
        })

    ws = [m["w_m"] for m in measurements if m["w_m"] is not None]
    dls = [m["d_L_m"] for m in measurements if m["d_L_m"] is not None]
    drs = [m["d_R_m"] for m in measurements if m["d_R_m"] is not None]
    print(f"trunk nodes: {len(trunk)}")
    if ws:
        print(f"width: min={min(ws):.2f} med={np.median(ws):.2f} max={max(ws):.2f} m  (N={len(ws)})")
        print(f"d_L:   min={min(dls):.2f} med={np.median(dls):.2f} max={max(dls):.2f} m")
        print(f"d_R:   min={min(drs):.2f} med={np.median(drs):.2f} max={max(drs):.2f} m")
    else:
        print("no valid widths")

    # ---------- BEV panel ----------
    xy = np.array(list(poly.exterior.coords))
    margin = float(args.margin_m)
    res = float(args.bev_res_m)
    x_min = xy[:, 0].min() - margin
    y_min = xy[:, 1].min() - margin
    x_max = xy[:, 0].max() + margin
    y_max = xy[:, 1].max() + margin
    grid_w = int(np.ceil((x_max - x_min) / res))
    grid_h = int(np.ceil((y_max - y_min) / res))

    bev = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    ext_pix = world_to_canvas(np.array(poly.exterior.coords), x_min, y_min, res, grid_h)
    cv2.fillPoly(bev, [ext_pix], (45, 45, 45))
    cv2.polylines(bev, [ext_pix], True, (200, 200, 200), 1, cv2.LINE_AA)
    for h in poly.interiors:
        h_pix = world_to_canvas(np.array(h.coords), x_min, y_min, res, grid_h)
        cv2.fillPoly(bev, [h_pix], (0, 0, 0))
        cv2.polylines(bev, [h_pix], True, (120, 120, 120), 1, cv2.LINE_AA)

    trunk_pix = world_to_canvas(trunk, x_min, y_min, res, grid_h)
    for k in range(len(trunk_pix) - 1):
        cv2.line(bev, tuple(trunk_pix[k]), tuple(trunk_pix[k + 1]),
                 (0, 220, 220), 2, cv2.LINE_AA)

    for m in measurements:
        if m["d_L_m"] is None or m["d_R_m"] is None:
            continue
        p_pix = world_to_canvas([m["p_xy"]], x_min, y_min, res, grid_h)[0]
        L_pix = world_to_canvas([m["L_xy"]], x_min, y_min, res, grid_h)[0]
        R_pix = world_to_canvas([m["R_xy"]], x_min, y_min, res, grid_h)[0]
        cv2.line(bev, tuple(p_pix), tuple(L_pix), (60, 60, 220), 1, cv2.LINE_AA)
        cv2.line(bev, tuple(p_pix), tuple(R_pix), (60, 220, 60), 1, cv2.LINE_AA)
        cv2.circle(bev, tuple(L_pix), 2, (60, 60, 220), -1, cv2.LINE_AA)
        cv2.circle(bev, tuple(R_pix), 2, (60, 220, 60), -1, cv2.LINE_AA)
        cv2.circle(bev, tuple(p_pix), 2, (255, 255, 255), -1, cv2.LINE_AA)
        if m["i"] % max(1, int(args.label_every)) == 0:
            cv2.putText(bev, f"{m['w_m']:.1f}",
                        (int(p_pix[0]) + 6, int(p_pix[1]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(bev, "L (left)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 220), 1, cv2.LINE_AA)
    cv2.putText(bev, "R (right)", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA)
    cv2.putText(bev, "trunk", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    # ---------- Chart panel ----------
    chart_w, chart_h = 560, grid_h
    chart = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)
    if ws:
        s_arr = np.array([m["s_m"] for m in measurements])
        d_L_arr = np.array([m["d_L_m"] if m["d_L_m"] is not None else np.nan for m in measurements])
        d_R_arr = np.array([m["d_R_m"] if m["d_R_m"] is not None else np.nan for m in measurements])
        w_arr = np.array([m["w_m"] if m["w_m"] is not None else np.nan for m in measurements])
        s_max = float(max(s_arr.max(), 1.0))
        y_max_val = float(max(np.nanmax(w_arr), 3.5))

        def sx(s):
            return int(60 + (chart_w - 90) * s / s_max)

        def sy(v):
            return int(chart_h - 40 - (chart_h - 70) * v / y_max_val)

        cv2.line(chart, (60, chart_h - 40), (chart_w - 30, chart_h - 40), (180, 180, 180), 1)
        cv2.line(chart, (60, 30), (60, chart_h - 40), (180, 180, 180), 1)
        for v in np.arange(0.5, y_max_val + 0.01, 0.5):
            y = sy(v)
            cv2.line(chart, (58, y), (62, y), (180, 180, 180), 1)
            cv2.putText(chart, f"{v:.1f}", (15, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        for s_tick in np.arange(0.0, s_max + 0.01, 5.0):
            x = sx(s_tick)
            cv2.line(chart, (x, chart_h - 42), (x, chart_h - 38), (180, 180, 180), 1)
            cv2.putText(chart, f"{s_tick:.0f}", (x - 8, chart_h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(chart, "s (m)", (chart_w - 80, chart_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(chart, "distance (m)", (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        def plot_series(arr, color, label, lx):
            prev = None
            for k in range(len(arr)):
                if np.isnan(arr[k]):
                    prev = None
                    continue
                px = sx(s_arr[k])
                py = sy(arr[k])
                cv2.circle(chart, (px, py), 2, color, -1, cv2.LINE_AA)
                if prev is not None:
                    cv2.line(chart, prev, (px, py), color, 1, cv2.LINE_AA)
                prev = (px, py)
            cv2.putText(chart, label, (lx, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        plot_series(d_L_arr, (60, 60, 220), "d_L", 130)
        plot_series(d_R_arr, (60, 220, 60), "d_R", 200)
        plot_series(w_arr, (255, 255, 255), "w = d_L + d_R", 270)

    gap = np.zeros((grid_h, 6, 3), dtype=np.uint8)
    gap[:, :] = (60, 60, 60)
    combined = np.hstack([bev, gap, chart])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)

    out_json = out_path.with_suffix(".json")
    out_json.write_text(json.dumps({
        "sidewalk_index": int(args.sidewalk_index),
        "ray_length_m": float(args.ray_length_m),
        "summary": {
            "trunk_nodes": int(len(trunk)),
            "n_valid": int(len(ws)),
            "width_min_m": float(min(ws)) if ws else None,
            "width_median_m": float(np.median(ws)) if ws else None,
            "width_max_m": float(max(ws)) if ws else None,
            "d_L_min_m": float(min(dls)) if dls else None,
            "d_L_max_m": float(max(dls)) if dls else None,
            "d_R_min_m": float(min(drs)) if drs else None,
            "d_R_max_m": float(max(drs)) if drs else None,
        },
        "measurements": measurements,
    }, indent=2))
    print(f"wrote {out_path} ({combined.shape[1]}x{combined.shape[0]}); json {out_json}")


if __name__ == "__main__":
    main()
