"""1D causal Kalman filter on sidewalk width along trunk arclength.

State:   w[i] (single scalar - the width)
Process: w[i] = w[i-1] + N(0, sigma_p * sqrt(delta_s))
Observ:  z[i] = w[i] + N(0, sigma_m)

Forward pass only (causal). Each new node updates state from previous
smoothed state + current raw measurement, weighted by the Kalman gain
(which adapts to the noise ratio). Steady-state gain approximates EWMA
alpha = sigma_p_per_step^2 / (sigma_p_per_step^2 + sigma_m^2).

Input:  width_sw0.json (output of sidewalk_width_profile.py).
Output: PNG chart with raw vs smoothed width curve + JSON sidecar with
        smoothed w per node.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="width_sw0.json")
    p.add_argument("--records", default=None,
                   help="sidewalk_records_fused.json for the polygon outline (BEV).")
    p.add_argument("--sidewalk-index", type=int, default=0)
    p.add_argument("--output", required=True, help="Output PNG path; JSON sidecar will be written alongside.")
    p.add_argument("--sigma-process-per-m", type=float, default=0.05,
                   help="Width process-noise stddev per meter of arclength.")
    p.add_argument("--sigma-measure", type=float, default=0.20,
                   help="Width measurement-noise stddev.")
    p.add_argument("--bev-res-m", type=float, default=0.05)
    p.add_argument("--margin-m", type=float, default=2.0)
    return p.parse_args()


def world_to_canvas(pts_xy, x_min, y_min, res, h_px):
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    col = ((pts[:, 0] - x_min) / res).astype(np.int32)
    row = (h_px - 1 - (pts[:, 1] - y_min) / res).astype(np.int32)
    return np.column_stack([col, row])


def render_bev(measurements, w_smooth, records_path, sidewalk_index,
               res, margin, target_h):
    """BEV panel: polygon outline + trunk + raw L/R rays (thin) + smoothed
    ribbon (yellow band). target_h is for height match with chart."""
    rec = json.load(open(records_path))
    outline = np.asarray(rec["sidewalks"][int(sidewalk_index)]["outline"], dtype=np.float64)

    # Trunk points + tangents from measurements
    P = np.array([m["p_xy"] for m in measurements], dtype=np.float64)
    T = np.array([m["tangent"] for m in measurements], dtype=np.float64)
    # Smoothed symmetric ribbon: distribute smoothed width equally L/R.
    # NOTE: we only filtered total w; asymmetry (d_L vs d_R) is the next step.
    n_left = np.column_stack([-T[:, 1], T[:, 0]])
    n_right = -n_left
    half_w = (w_smooth / 2.0).astype(np.float64)
    L_sm = P + n_left * half_w[:, None]
    R_sm = P + n_right * half_w[:, None]

    # Bounds
    all_xy = np.vstack([outline, P, L_sm[~np.isnan(half_w)], R_sm[~np.isnan(half_w)]])
    x_min = all_xy[:, 0].min() - margin
    y_min = all_xy[:, 1].min() - margin
    x_max = all_xy[:, 0].max() + margin
    y_max = all_xy[:, 1].max() + margin
    grid_w = int(np.ceil((x_max - x_min) / res))
    grid_h = int(np.ceil((y_max - y_min) / res))
    # Resize res so grid_h ~= target_h
    if grid_h > target_h * 1.1:
        scale = grid_h / target_h
        res = res * scale
        grid_w = int(np.ceil((x_max - x_min) / res))
        grid_h = int(np.ceil((y_max - y_min) / res))

    img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    ext_pix = world_to_canvas(outline, x_min, y_min, res, grid_h)
    cv2.fillPoly(img, [ext_pix], (45, 45, 45))
    cv2.polylines(img, [ext_pix], True, (200, 200, 200), 1, cv2.LINE_AA)

    # Raw L/R rays (thin gray)
    for m in measurements:
        if m["d_L_m"] is None or m["d_R_m"] is None:
            continue
        p_pix = world_to_canvas([m["p_xy"]], x_min, y_min, res, grid_h)[0]
        L_pix = world_to_canvas([m["L_xy"]], x_min, y_min, res, grid_h)[0]
        R_pix = world_to_canvas([m["R_xy"]], x_min, y_min, res, grid_h)[0]
        cv2.line(img, tuple(p_pix), tuple(L_pix), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(img, tuple(p_pix), tuple(R_pix), (90, 90, 90), 1, cv2.LINE_AA)

    # Smoothed ribbon: connect L_sm points (left boundary) and R_sm points (right boundary)
    L_sm_valid = []
    R_sm_valid = []
    for i in range(len(P)):
        if np.isnan(half_w[i]):
            L_sm_valid.append(None); R_sm_valid.append(None); continue
        L_sm_valid.append(L_sm[i])
        R_sm_valid.append(R_sm[i])
    def draw_seg(pts_list, color):
        prev = None
        for p in pts_list:
            if p is None:
                prev = None; continue
            cur = tuple(world_to_canvas([p], x_min, y_min, res, grid_h)[0])
            if prev is not None:
                cv2.line(img, prev, cur, color, 2, cv2.LINE_AA)
            cv2.circle(img, cur, 2, color, -1, cv2.LINE_AA)
            prev = cur
    draw_seg(L_sm_valid, (0, 255, 255))  # left  ribbon (yellow)
    draw_seg(R_sm_valid, (0, 255, 255))  # right ribbon (yellow)

    # Trunk (white)
    trunk_pix = world_to_canvas(P, x_min, y_min, res, grid_h)
    for k in range(len(trunk_pix) - 1):
        cv2.line(img, tuple(trunk_pix[k]), tuple(trunk_pix[k + 1]),
                 (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, "polygon", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "raw rays", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(img, "trunk", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "smoothed ribbon (sym w)", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Pad to target_h if needed
    if img.shape[0] < target_h:
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img = np.vstack([img, pad])
    elif img.shape[0] > target_h:
        img = img[:target_h]
    return img


def kalman_1d_forward(z_arr, sigma_p_per_step, sigma_m):
    """Forward 1D Kalman: state = w, process noise = N(0, sigma_p_per_step^2),
    measurement noise = N(0, sigma_m^2). NaN observations propagate state."""
    n = len(z_arr)
    smoothed = np.full(n, np.nan, dtype=np.float64)
    x = None
    P = sigma_m ** 2
    sp2 = sigma_p_per_step ** 2
    sm2 = sigma_m ** 2
    for i, z in enumerate(z_arr):
        if x is None and np.isnan(z):
            continue
        if x is None:
            x = float(z)
            P = sm2
            smoothed[i] = x
            continue
        # Predict
        P_pred = P + sp2
        if np.isnan(z):
            # No measurement; just propagate (state unchanged, covariance grows)
            P = P_pred
            smoothed[i] = x
            continue
        # Update
        K = P_pred / (P_pred + sm2)
        x = x + K * (float(z) - x)
        P = (1.0 - K) * P_pred
        smoothed[i] = x
    return smoothed


def main():
    args = parse_args()
    d = json.load(open(args.input))
    measurements = d["measurements"]

    s_arr = np.array([m["s_m"] for m in measurements], dtype=np.float64)
    w_arr = np.array([m["w_m"] if m["w_m"] is not None else np.nan
                      for m in measurements], dtype=np.float64)

    # Step size: use median of consecutive arclength deltas as the "unit"
    if len(s_arr) >= 2:
        median_step = float(np.median(np.diff(s_arr)))
    else:
        median_step = 1.0
    sigma_p_step = float(args.sigma_process_per_m) * np.sqrt(median_step)
    sigma_m = float(args.sigma_measure)
    w_smooth = kalman_1d_forward(w_arr, sigma_p_step, sigma_m)

    K_ss = sigma_p_step ** 2 / (sigma_p_step ** 2 + sigma_m ** 2) if sigma_p_step > 0 else 0.0
    print(f"step={median_step:.2f}m  sigma_p_per_step={sigma_p_step:.3f}m  sigma_m={sigma_m}m")
    print(f"steady-state gain alpha ~ {K_ss:.3f}")

    raw_valid = w_arr[~np.isnan(w_arr)]
    sm_valid = w_smooth[~np.isnan(w_smooth)]
    print(f"raw    width: min={raw_valid.min():.2f} med={np.median(raw_valid):.2f} "
          f"max={raw_valid.max():.2f} std={raw_valid.std():.3f}")
    print(f"smooth width: min={sm_valid.min():.2f} med={np.median(sm_valid):.2f} "
          f"max={sm_valid.max():.2f} std={sm_valid.std():.3f}")

    # ---------- Chart ----------
    chart_w, chart_h = 1000, 480
    chart = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)
    s_max = float(max(s_arr.max(), 1.0))
    combined = np.concatenate([raw_valid, sm_valid])
    y_max_val = float(max(combined.max() + 0.2, 3.5))

    def sx(s):
        return int(60 + (chart_w - 80) * s / s_max)

    def sy(v):
        return int(chart_h - 40 - (chart_h - 70) * v / y_max_val)

    cv2.line(chart, (60, chart_h - 40), (chart_w - 20, chart_h - 40),
             (180, 180, 180), 1)
    cv2.line(chart, (60, 30), (60, chart_h - 40), (180, 180, 180), 1)
    for v in np.arange(0.5, y_max_val + 0.01, 0.5):
        y = sy(v)
        cv2.line(chart, (58, y), (62, y), (180, 180, 180), 1)
        cv2.putText(chart, f"{v:.1f}", (15, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for st in np.arange(0.0, s_max + 0.01, 5.0):
        x = sx(st)
        cv2.line(chart, (x, chart_h - 42), (x, chart_h - 38),
                 (180, 180, 180), 1)
        cv2.putText(chart, f"{st:.0f}", (x - 8, chart_h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(chart, "distance (m)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(chart, "s (m)", (chart_w - 70, chart_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def plot(arr, color, lw=1, dot_r=2):
        prev = None
        for k in range(len(arr)):
            if np.isnan(arr[k]):
                prev = None
                continue
            px = sx(s_arr[k])
            py = sy(arr[k])
            if dot_r > 0:
                cv2.circle(chart, (px, py), dot_r, color, -1, cv2.LINE_AA)
            if prev is not None:
                cv2.line(chart, prev, (px, py), color, lw, cv2.LINE_AA)
            prev = (px, py)

    plot(w_arr, (130, 130, 130), 1, dot_r=2)
    plot(w_smooth, (0, 255, 255), 2, dot_r=3)

    cv2.putText(chart, "raw w", (130, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.putText(chart, "smoothed w (1D Kalman, forward)", (210, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(chart,
                f"sigma_p/m={args.sigma_process_per_m}  sigma_m={sigma_m}  alpha~{K_ss:.2f}",
                (chart_w - 380, chart_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 220, 220), 1, cv2.LINE_AA)

    # ---------- Optional BEV panel ----------
    bev = None
    if args.records:
        bev = render_bev(measurements, w_smooth, args.records,
                         int(args.sidewalk_index), float(args.bev_res_m),
                         float(args.margin_m), chart_h)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bev is not None:
        gap = np.zeros((chart_h, 6, 3), dtype=np.uint8)
        gap[:, :] = (60, 60, 60)
        combined = np.hstack([bev, gap, chart])
        cv2.imwrite(str(out_path), combined)
    else:
        cv2.imwrite(str(out_path), chart)

    # Sidecar JSON: copy measurements + add w_smooth_m
    for i, m in enumerate(measurements):
        m["w_smooth_m"] = float(w_smooth[i]) if not np.isnan(w_smooth[i]) else None
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "input": str(args.input),
        "sigma_process_per_m": float(args.sigma_process_per_m),
        "sigma_p_per_step": float(sigma_p_step),
        "sigma_measure": float(sigma_m),
        "median_step_m": float(median_step),
        "kalman_steady_gain": float(K_ss),
        "summary": {
            "raw_min": float(raw_valid.min()) if len(raw_valid) else None,
            "raw_med": float(np.median(raw_valid)) if len(raw_valid) else None,
            "raw_max": float(raw_valid.max()) if len(raw_valid) else None,
            "raw_std": float(raw_valid.std()) if len(raw_valid) else None,
            "smooth_min": float(sm_valid.min()) if len(sm_valid) else None,
            "smooth_med": float(np.median(sm_valid)) if len(sm_valid) else None,
            "smooth_max": float(sm_valid.max()) if len(sm_valid) else None,
            "smooth_std": float(sm_valid.std()) if len(sm_valid) else None,
        },
        "measurements": measurements,
    }, indent=2))
    print(f"wrote {out_path}; json {sidecar}")


if __name__ == "__main__":
    main()
