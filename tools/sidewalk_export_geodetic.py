"""Export sidewalk centerline + left/right boundary in geodetic (lat, lon, h).

Inputs:
  - width JSON (from sidewalk_width_smooth_1d.py) — has per-node trunk xy
    + tangent + smoothed w
  - ie txt — to extract the ENU origin (first pose's geodetic coord)

For each node:
  centerline  = p_xy                       (trunk point in ENU)
  left bound  = p_xy + n_left  * w/2
  right bound = p_xy + n_right * w/2
ENU -> ECEF -> (lat, lon, h) using the same conversion the pipeline used.

Output:
  - GeoJSON FeatureCollection with 3 LineStrings (centerline, left, right)
  - JSON sidecar with the per-node lat/lon arrays
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


# --- Geodetic conversions (copies of outdoor_livox_ie.py helpers) ---

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat_deg, lon_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [[-sin_lon, cos_lon, 0.0],
         [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
         [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]],
        dtype=np.float64,
    )


def ecef_to_geodetic(ecef):
    """Zhu 1994 closed-form ECEF → (lat_deg, lon_deg, height_m)."""
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    a, b = 6378137.0, 6356752.3142
    e2 = 1.0 - (b / a) ** 2
    ep2 = (a / b) ** 2 - 1.0
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(theta) ** 3,
                     p - e2 * a * math.cos(theta) ** 3)
    lon = math.atan2(y, x)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    cos_lat = math.cos(lat)
    h = (p / cos_lat - N) if abs(cos_lat) > 1e-6 else (abs(z) / abs(sin_lat) - N * (1.0 - e2))
    return math.degrees(lat), math.degrees(lon), h


def enu_to_geodetic_batch(pts_enu, ecef_origin, enu_from_ecef):
    """pts_enu: (N, 3). Returns (N, 3) lat/lon/h."""
    ecef_from_enu = enu_from_ecef.T
    pts_ecef = (ecef_from_enu @ np.asarray(pts_enu).T).T + ecef_origin
    out = np.zeros_like(pts_ecef)
    for i, e in enumerate(pts_ecef):
        out[i] = ecef_to_geodetic(e)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width-json", required=True,
                   help="Output of sidewalk_width_smooth_1d.py (has per-node p_xy, tangent, w_smooth_m).")
    p.add_argument("--ie-txt", required=True,
                   help="IE pose txt (first row's lat/lon/h defines the ENU origin).")
    p.add_argument("--sidewalk-z", type=float, default=None,
                   help="Override sidewalk z height (m) used for trunk + boundary. "
                        "Default: median of per-node L_xy / R_xy z if width-json carries them, "
                        "else 0.")
    p.add_argument("--output", required=True,
                   help="Output GeoJSON path. JSON sidecar will be written alongside.")
    p.add_argument("--label", default="sidewalk",
                   help="Identifier set on each GeoJSON Feature.")
    return p.parse_args()


def dms_to_deg(d, m, s):
    d = float(d); m = float(m); s = float(s)
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def load_ie_origin(ie_txt):
    """Read first valid IE data row to get (lat_deg, lon_deg, h_m) origin.
    IE.txt format: parts[3..5]=lat_dms, parts[6..8]=lon_dms, parts[9]=h."""
    import re
    with open(ie_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or not re.match(r"^[0-9]", line):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                lat = dms_to_deg(parts[3], parts[4], parts[5])
                lon = dms_to_deg(parts[6], parts[7], parts[8])
                h = float(parts[9])
                return lat, lon, h
            except (ValueError, TypeError):
                continue
    raise SystemExit("could not parse origin from " + ie_txt)


def main():
    args = parse_args()
    data = json.load(open(args.width_json))
    measurements = data["measurements"]

    P = np.array([m["p_xy"] for m in measurements], dtype=np.float64)
    T = np.array([m["tangent"] for m in measurements], dtype=np.float64)
    w_sm = np.array([m["w_smooth_m"] if m.get("w_smooth_m") is not None else np.nan
                     for m in measurements], dtype=np.float64)
    if np.isnan(w_sm).all():
        # Fall back to raw w if smoothing not available
        w_sm = np.array([m["w_m"] if m["w_m"] is not None else np.nan
                         for m in measurements], dtype=np.float64)

    # Pick z (height in ENU vertical) for the trunk
    if args.sidewalk_z is not None:
        z_enu = float(args.sidewalk_z)
    else:
        # Use median of any per-node L/R xy z carried in width-json (they're 2D in
        # our schema, so fall back to a config-provided sidewalk_z or 0).
        z_enu = 0.0
    print(f"trunk z (ENU vertical) = {z_enu:.3f} m")

    # Build centerline + boundary in ENU
    n_left = np.column_stack([-T[:, 1], T[:, 0]])
    n_right = -n_left
    half = np.where(np.isnan(w_sm), 0.0, w_sm / 2.0)
    centerline_enu = np.column_stack([P[:, 0], P[:, 1], np.full(len(P), z_enu)])
    left_enu = np.column_stack([
        P[:, 0] + n_left[:, 0] * half,
        P[:, 1] + n_left[:, 1] * half,
        np.full(len(P), z_enu),
    ])
    right_enu = np.column_stack([
        P[:, 0] + n_right[:, 0] * half,
        P[:, 1] + n_right[:, 1] * half,
        np.full(len(P), z_enu),
    ])
    # Drop boundary points where width is NaN
    valid = ~np.isnan(w_sm)
    left_enu = left_enu[valid]
    right_enu = right_enu[valid]

    # ENU -> geodetic
    lat0, lon0, h0 = load_ie_origin(args.ie_txt)
    print(f"ENU origin: lat={lat0:.9f} lon={lon0:.9f} h={h0:.3f}")
    ecef_origin = geodetic_to_ecef(lat0, lon0, h0)
    enu_from_ecef = ecef_to_enu_matrix(lat0, lon0)

    center_ll = enu_to_geodetic_batch(centerline_enu, ecef_origin, enu_from_ecef)
    left_ll = enu_to_geodetic_batch(left_enu, ecef_origin, enu_from_ecef)
    right_ll = enu_to_geodetic_batch(right_enu, ecef_origin, enu_from_ecef)

    # GeoJSON FeatureCollection
    def line_feature(name, coords_ll, color=None):
        # GeoJSON LineString: [[lon, lat, h], ...]
        coords = [[float(c[1]), float(c[0]), float(c[2])] for c in coords_ll]
        props = {"name": name, "label": args.label}
        if color:
            props["stroke"] = color
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }

    geojson = {
        "type": "FeatureCollection",
        "features": [
            line_feature("centerline", center_ll, "#ffffff"),
            line_feature("left_boundary", left_ll, "#ff5555"),
            line_feature("right_boundary", right_ll, "#55ff55"),
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    # JSON sidecar with raw arrays
    sidecar = out_path.with_suffix(".raw.json")
    with open(sidecar, "w") as f:
        json.dump({
            "label": args.label,
            "origin": {"lat_deg": lat0, "lon_deg": lon0, "height_m": h0},
            "sidewalk_z_enu_m": z_enu,
            "centerline_lla": [
                {"lat_deg": float(c[0]), "lon_deg": float(c[1]), "height_m": float(c[2])}
                for c in center_ll
            ],
            "left_boundary_lla": [
                {"lat_deg": float(c[0]), "lon_deg": float(c[1]), "height_m": float(c[2])}
                for c in left_ll
            ],
            "right_boundary_lla": [
                {"lat_deg": float(c[0]), "lon_deg": float(c[1]), "height_m": float(c[2])}
                for c in right_ll
            ],
        }, f, indent=2)
    print(f"wrote {out_path}  (GeoJSON, {len(center_ll)} center + {len(left_ll)} L + {len(right_ll)} R)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
