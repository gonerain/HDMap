#!/usr/bin/python3
"""Robust ground-plane denoising for sidewalk PCDs.

Problem: when we paint sidewalk-class points from segmentation, some human
bodies (and other tall objects mis-classified as sidewalk) leak in, producing
points scattered up to ~2 m above the actual ground. We want to keep only
points that lie on (or very close to) the local sidewalk surface.

Pipeline:
  1. (optional) voxel-downsample to a small fitting set so plane fitting
     is fast and isn't dominated by dense regions.
  2. Tile XY into cells (default 3 m). Per cell, run MSAC plane fit with a
     near-vertical normal constraint (|n . z_hat| >= 0.85) -- this rejects
     walls/poles even when they outnumber the ground.
  3. Reweighted SVD refit on inliers using a Tukey biweight loss; the
     resulting plane is robust to remaining outliers.
  4. Bilinear-interpolate the per-cell plane height z_plane(x, y) onto every
     ORIGINAL point (no smearing -- only the fit set was downsampled).
  5. Keep points with -tau_below <= z - z_plane <= tau_above.
  6. (optional) Statistical Outlier Removal pass (k-NN mean distance) to
     prune isolated stragglers.

Designed to run without Open3D / PCL. Only needs numpy + scipy.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np


# Mapillary Vistas label colormap (id -> rgb), copied from save_semantic_pcd.py
MAPILLARY_COLORMAP = np.asarray([
    [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
    [180, 165, 180], [102, 102, 156], [102, 102, 156], [128, 64, 255],
    [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
    [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
    [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
    [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128],
    [255, 255, 255], [64, 170, 64], [128, 64, 64], [70, 130, 180],
    [255, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
    [255, 255, 128], [250, 0, 30], [0, 0, 0], [220, 220, 220],
    [170, 170, 170], [222, 40, 40], [100, 170, 30], [40, 40, 40],
    [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170],
    [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 142],
    [250, 170, 30], [192, 192, 192], [220, 220, 0], [180, 165, 180],
    [119, 11, 32], [0, 0, 142], [0, 60, 100], [0, 0, 142],
    [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
    [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32],
    [0, 0, 0], [0, 0, 0],
], dtype=np.uint8)


def _class_to_rgb_packed(classes):
    """Class id ndarray -> PCL packed-float rgb ndarray (one float per point)."""
    table = MAPILLARY_COLORMAP
    safe = np.clip(classes.astype(np.int64), 0, len(table) - 1)
    rgb = table[safe]
    rgb_int = ((rgb[:, 0].astype(np.uint32) & 0xFF) << 16) | \
              ((rgb[:, 1].astype(np.uint32) & 0xFF) << 8)  | \
              (rgb[:, 2].astype(np.uint32) & 0xFF)
    return rgb_int.astype(np.uint32).view(np.float32)


def load_pkl_to_xyzrgb(pkl_path, keep_class=None):
    """Load multi-frame pkl, concatenate, optionally class-filter, return
    (xyz f32 [N,3], extras={'rgb': float32 packed [N]}, header_dict for save_pcd).
    """
    arrs = []
    with open(pkl_path, "rb") as f:
        while True:
            try:
                arrs.append(pickle.load(f))
            except EOFError:
                break
    if not arrs:
        raise RuntimeError(f"empty pkl: {pkl_path}")
    pts = np.vstack(arrs).astype(np.float32, copy=False)   # (N, 4) [x,y,z,class]
    n_total = len(pts)
    if keep_class is not None:
        pts = pts[pts[:, 3].astype(np.int64) == int(keep_class)]
    xyz = pts[:, :3].copy()
    rgb_f = _class_to_rgb_packed(pts[:, 3])
    extras = {"rgb": rgb_f}
    header = {
        "FIELDS": ["x", "y", "z", "rgb"],
        "SIZE":   [4, 4, 4, 4],
        "TYPE":   ["F", "F", "F", "F"],
    }
    print(f"  pkl: {len(arrs)} frames, {n_total:,} points "
          f"-> {len(xyz):,} after class={keep_class} filter", flush=True)
    return xyz, extras, header


# ------------------------------- PCD IO --------------------------------------

def _read_pcd_header(fh):
    header_lines = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unexpected EOF in PCD header")
        header_lines.append(line)
        if line.startswith(b"DATA"):
            break
    header = {"_lines": header_lines}
    for line in header_lines:
        parts = line.decode("ascii", errors="ignore").strip().split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            header["FIELDS"] = parts[1:]
        elif key == "SIZE":
            header["SIZE"] = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            header["TYPE"] = parts[1:]
        elif key == "COUNT":
            header["COUNT"] = [int(x) for x in parts[1:]]
        elif key == "POINTS":
            header["POINTS"] = int(parts[1])
        elif key == "DATA":
            header["DATA"] = parts[1].lower()
    return header


def load_pcd(path):
    """Returns (points_xyz: float32 [N,3], extras: dict of name->ndarray)."""
    with open(path, "rb") as f:
        header = _read_pcd_header(f)
        if header.get("DATA") != "binary":
            raise NotImplementedError(
                f"only binary PCD supported, got DATA={header.get('DATA')}"
            )
        n = header["POINTS"]
        fields = header["FIELDS"]
        sizes = header["SIZE"]
        types = header["TYPE"]
        counts = header["COUNT"]
        if any(c != 1 for c in counts):
            raise NotImplementedError("multi-count fields not supported")
        dtype = np.dtype({
            "names": fields,
            "formats": [_pcd_field_dtype(t, s) for t, s in zip(types, sizes)],
        })
        raw = np.frombuffer(f.read(), dtype=dtype, count=n)
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32)
    extras = {name: raw[name].copy() for name in fields if name not in ("x", "y", "z")}
    return xyz, extras, header


def _pcd_field_dtype(t, s):
    t = t.upper()
    if t == "F":
        return np.dtype(f"f{s}")
    if t == "U":
        return np.dtype(f"u{s}")
    if t == "I":
        return np.dtype(f"i{s}")
    raise ValueError(f"unknown PCD field type {t}{s}")


def save_pcd(path, xyz, extras, ref_header):
    """Write binary PCD that mirrors the reference field layout."""
    fields = ref_header["FIELDS"]
    sizes = ref_header["SIZE"]
    types = ref_header["TYPE"]
    n = len(xyz)
    dtype = np.dtype({
        "names": fields,
        "formats": [_pcd_field_dtype(t, s) for t, s in zip(types, sizes)],
    })
    out = np.empty(n, dtype=dtype)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    for name in fields:
        if name in ("x", "y", "z"):
            continue
        if name not in extras:
            raise KeyError(f"missing extra field {name!r} for output")
        if len(extras[name]) != n:
            raise ValueError(f"extra field {name!r} length mismatch")
        out[name] = extras[name]
    header = (
        f"# .PCD v0.7 - sidewalk-denoised\n"
        f"VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(s) for s in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join('1' for _ in fields)}\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes(order="C"))


# ------------------------------- Filtering -----------------------------------

def voxel_downsample_indices(xyz, voxel):
    """Return indices of one representative point per voxel (first hit)."""
    if voxel <= 0:
        return np.arange(len(xyz))
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    rng = keys.max(axis=0) + 1
    enc = (keys[:, 0] * rng[1] + keys[:, 1]) * rng[2] + keys[:, 2]
    order = np.argsort(enc, kind="stable")
    enc_sorted = enc[order]
    first_mask = np.concatenate([[True], np.diff(enc_sorted) != 0])
    return np.sort(order[first_mask])


def msac_plane_fit(pts, n_iter=200, thresh=0.05, n_min=0.85, rng=None):
    """MSAC plane fit (RANSAC variant scoring squared truncated residuals).

    Returns (n_unit, d, inlier_mask) for plane n.x = d, with n[2] >= n_min,
    or None if no acceptable consensus was found.
    """
    n_pts = len(pts)
    if n_pts < 3:
        return None
    rng = rng or np.random.default_rng(0)
    pts64 = np.asarray(pts, dtype=np.float64)
    best_score = np.inf
    best_inl = None
    best_plane = None
    t2 = thresh * thresh
    for _ in range(n_iter):
        idx = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = pts64[idx]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < n_min:
            continue
        d = float(n @ p1)
        r2 = (pts64 @ n - d) ** 2
        score = float(np.minimum(r2, t2).sum())
        if score < best_score:
            best_score = score
            best_plane = (n, d)
            best_inl = r2 < t2
    if best_plane is None:
        return None
    return best_plane[0], best_plane[1], best_inl


def tukey_refit_plane(pts, init_plane, c=0.10, n_iter=4):
    """IRLS plane refit with Tukey biweight, anchored on near-vertical normal."""
    n, d = init_plane
    pts64 = np.asarray(pts, dtype=np.float64)
    for _ in range(n_iter):
        r = pts64 @ n - d
        u = r / c
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if w.sum() < 3:
            break
        c_w = (w[:, None] * pts64).sum(axis=0) / w.sum()
        X = pts64 - c_w
        WX = X * np.sqrt(w[:, None])
        # Smallest singular vector of WX = plane normal
        try:
            _, _, Vt = np.linalg.svd(WX, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        n_new = Vt[-1]
        n_new /= np.linalg.norm(n_new) + 1e-12
        if n_new[2] < 0:
            n_new = -n_new
        n = n_new
        d = float(n @ c_w)
    return n, d


def fit_tile_planes(
    xyz_fit,
    cell_xy,
    msac_thresh,
    msac_iter,
    tukey_c,
    n_min_z,
    min_pts_per_cell,
    rng,
    log,
    ground_band=0.30,
):
    """Run MSAC + Tukey IRLS per XY tile, with a low-Z ground prior.

    Ground prior: in each tile, restrict the MSAC fit set to points within
    [z_min_robust, z_min_robust + ground_band] where z_min_robust is the
    5th-percentile z (robust against single-pixel depth bugs). This prevents
    a tall cluster of human-body points from outvoting the actual ground in
    tiles where humans dominate.

    Returns (z_plane_grid, x_edges, y_edges) where z_plane_grid[i, j] is the
    fitted z at the cell center (i, j); NaN where no plane was found.
    """
    x_min, y_min = xyz_fit[:, 0].min(), xyz_fit[:, 1].min()
    x_max, y_max = xyz_fit[:, 0].max(), xyz_fit[:, 1].max()
    nx = max(1, int(np.ceil((x_max - x_min) / cell_xy)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_xy)))
    x_edges = x_min + np.arange(nx + 1) * cell_xy
    y_edges = y_min + np.arange(ny + 1) * cell_xy
    z_grid = np.full((nx, ny), np.nan, dtype=np.float64)

    ix = np.clip(np.floor((xyz_fit[:, 0] - x_min) / cell_xy).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((xyz_fit[:, 1] - y_min) / cell_xy).astype(int), 0, ny - 1)
    cell_id = ix * ny + iy
    order = np.argsort(cell_id, kind="stable")
    cell_id_sorted = cell_id[order]
    bounds = np.searchsorted(cell_id_sorted, np.arange(nx * ny + 1))

    n_fit, n_skip = 0, 0
    for c in range(nx * ny):
        a, b = bounds[c], bounds[c + 1]
        if b - a < min_pts_per_cell:
            n_skip += 1
            continue
        pts_full = xyz_fit[order[a:b]]
        # Ground prior: only fit on the low-Z band of this tile. Robust
        # 5th-percentile floor + ground_band ceiling. This prevents human
        # bodies (which can be 50%+ of points in a busy tile) from being
        # selected as the dominant plane.
        z_floor = float(np.percentile(pts_full[:, 2], 5))
        ground_mask = pts_full[:, 2] <= z_floor + ground_band
        pts = pts_full[ground_mask]
        if len(pts) < max(min_pts_per_cell, 6):
            # Too few low points -- fall back to robust low-Z height
            z_grid[c // ny, c % ny] = z_floor
            n_skip += 1
            continue
        res = msac_plane_fit(pts, n_iter=msac_iter, thresh=msac_thresh,
                             n_min=n_min_z, rng=rng)
        if res is None:
            z_grid[c // ny, c % ny] = z_floor
            n_skip += 1
            continue
        normal, d, inl_mask = res
        if inl_mask.sum() >= max(min_pts_per_cell, 6):
            normal, d = tukey_refit_plane(pts[inl_mask], (normal, d), c=tukey_c)
        # Evaluate plane at cell center
        cx = (c // ny) + 0.5
        cy = (c % ny) + 0.5
        x_c = x_min + cx * cell_xy
        y_c = y_min + cy * cell_xy
        if abs(normal[2]) < 1e-6:
            z_grid[c // ny, c % ny] = float(np.percentile(pts[:, 2], 25))
        else:
            z_grid[c // ny, c % ny] = (d - normal[0] * x_c - normal[1] * y_c) / normal[2]
        n_fit += 1

    log(f"  tiles: {nx}x{ny} = {nx*ny}; fit={n_fit}, fallback/skip={n_skip}")
    # Fill NaNs with nearest finite value (cells that had too few points or
    # only outliers) -- prevents bilinear interp from producing NaN.
    if not np.isfinite(z_grid).all():
        z_grid = _fill_nan_nearest(z_grid)
    return z_grid, x_edges, y_edges


def _fill_nan_nearest(grid):
    """Fill NaNs by nearest finite neighbour (BFS)."""
    from collections import deque
    out = grid.copy()
    nx, ny = out.shape
    finite = np.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        out[~finite] = 0.0
        return out
    dq = deque()
    for i in range(nx):
        for j in range(ny):
            if finite[i, j]:
                dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and not finite[ni, nj]:
                out[ni, nj] = out[i, j]
                finite[ni, nj] = True
                dq.append((ni, nj))
    return out


def bilinear_sample_grid(z_grid, x_edges, y_edges, cell_xy, xyz):
    """Bilinearly sample z_grid (defined at cell CENTERS) at xyz[:, :2]."""
    nx, ny = z_grid.shape
    # cell centers
    xc = x_edges[0] + (np.arange(nx) + 0.5) * cell_xy
    yc = y_edges[0] + (np.arange(ny) + 0.5) * cell_xy
    fx = (xyz[:, 0] - xc[0]) / cell_xy
    fy = (xyz[:, 1] - yc[0]) / cell_xy
    i0 = np.clip(np.floor(fx).astype(int), 0, nx - 1)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny - 1)
    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    z00 = z_grid[i0, j0]
    z10 = z_grid[i1, j0]
    z01 = z_grid[i0, j1]
    z11 = z_grid[i1, j1]
    return (
        z00 * (1 - tx) * (1 - ty)
        + z10 * tx * (1 - ty)
        + z01 * (1 - tx) * ty
        + z11 * tx * ty
    )


def local_ground_z_knn(query_xy, source_xy, source_z, k=64, q=10):
    """For each query point, return the q-th percentile of source_z among its
    k nearest XY neighbours. This is the "local ground" estimator: small q
    (e.g. 5-15) returns the lower envelope of the local Z distribution, which
    is the actual sidewalk surface even when humans dominate the high-Z tail.

    Robust to sidewalk slope (uses local neighbourhood) and to sparse/diagonal
    coverage (no grid/tile assumption).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(source_xy)
    k_eff = min(k, len(source_xy))
    _, idx = tree.query(query_xy, k=k_eff)
    if k_eff == 1:
        return source_z[idx]
    return np.percentile(source_z[idx], q, axis=1)


def local_ground_z_knn_planefit(query_xy, source_xy, source_z, k=64, q=10,
                                 plane_band=0.20):
    """Like `local_ground_z_knn` but tightens the result by least-squares fitting
    a tilted plane to the bottom-Z neighbours. Returns the plane height at each
    query (x, y) — handles sloped sidewalks more accurately than a flat
    percentile, since neighbours uphill/downhill don't bias the estimate.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(source_xy)
    k_eff = min(k, len(source_xy))
    _, idx = tree.query(query_xy, k=k_eff)
    if k_eff == 1:
        return source_z[idx]
    z_neigh = source_z[idx]                # (Nq, k)
    x_neigh = source_xy[idx, 0]            # (Nq, k)
    y_neigh = source_xy[idx, 1]            # (Nq, k)
    z_floor = np.percentile(z_neigh, q, axis=1, keepdims=True)
    band_mask = z_neigh <= (z_floor + plane_band)
    out = z_floor.squeeze(-1).copy()
    # Vectorised per-query plane fit on the band points: solve [x, y, 1] [a, b, c] = z
    # Use weighted least squares with band_mask as the weight (0/1).
    w = band_mask.astype(np.float64)
    sx = (w * x_neigh).sum(1)
    sy = (w * y_neigh).sum(1)
    sw = w.sum(1)
    sxx = (w * x_neigh * x_neigh).sum(1)
    sxy = (w * x_neigh * y_neigh).sum(1)
    syy = (w * y_neigh * y_neigh).sum(1)
    sxz = (w * x_neigh * z_neigh).sum(1)
    syz = (w * y_neigh * z_neigh).sum(1)
    sz  = (w * z_neigh).sum(1)
    # Build 3x3 normal matrix per query and solve. Vectorise via a stack.
    A = np.empty((len(out), 3, 3))
    A[:, 0, 0] = sxx; A[:, 0, 1] = sxy; A[:, 0, 2] = sx
    A[:, 1, 0] = sxy; A[:, 1, 1] = syy; A[:, 1, 2] = sy
    A[:, 2, 0] = sx;  A[:, 2, 1] = sy;  A[:, 2, 2] = sw
    rhs = np.stack([sxz, syz, sz], axis=-1)[..., None]   # (Nq, 3, 1)
    qx, qy = query_xy[:, 0], query_xy[:, 1]
    valid = sw >= 4
    if valid.any():
        try:
            sol = np.linalg.solve(A[valid], rhs[valid])[..., 0]   # (Nv, 3)
            out[valid] = sol[:, 0] * qx[valid] + sol[:, 1] * qy[valid] + sol[:, 2]
        except np.linalg.LinAlgError:
            pass
    return out


def statistical_outlier_removal(xyz, k=12, std_ratio=2.0):
    """Drop points whose mean kNN distance exceeds mean + std_ratio * std."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [SOR] scipy missing -- skipping", file=sys.stderr)
        return np.ones(len(xyz), dtype=bool)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)
    # d[:, 0] == 0 (self); exclude it
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + std_ratio * md.std()
    return md <= thr


# ------------------------------- Main ----------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None, help="input PCD path (mutually exclusive with --input-pkl)")
    p.add_argument("--input-pkl", default=None,
                   help="input pkl produced by outdoor_livox_ie.py (each pickle "
                        "entry is a (N,4) [x,y,z,class] array). Combine with "
                        "--keep-class to filter to one semantic class.")
    p.add_argument("--output", required=True, help="output denoised PCD path")
    p.add_argument("--keep-class", type=int, default=None,
                   help="when reading from --input-pkl, keep only this class id "
                        "(e.g. 15 for sidewalk under Mapillary Vistas).")
    p.add_argument("--tile-size", type=float, default=3.0,
                   help="XY tile edge length in meters (default 3.0)")
    p.add_argument("--fit-voxel", type=float, default=0.05,
                   help="voxel size for downsampling the fit set (m); 0 disables")
    p.add_argument("--msac-thresh", type=float, default=0.05,
                   help="MSAC inlier residual (m). Sidewalk roughness ~3-5 cm.")
    p.add_argument("--msac-iter", type=int, default=200)
    p.add_argument("--n-min-z", type=float, default=0.85,
                   help="min |normal . z_hat| to accept plane (rejects walls)")
    p.add_argument("--tukey-c", type=float, default=0.10,
                   help="Tukey biweight scale (m); points beyond contribute 0")
    p.add_argument("--tau-above", type=float, default=0.20,
                   help="keep points up to this far ABOVE the local plane (m). "
                        "0.20 picks up ~95%% of the dz distribution on real "
                        "sidewalk data while still rejecting people; 0.12 was "
                        "too aggressive and lost real points.")
    p.add_argument("--tau-below", type=float, default=0.10,
                   help="keep points up to this far BELOW the local plane (m)")
    p.add_argument("--min-pts-per-cell", type=int, default=30)
    p.add_argument("--ground-band", type=float, default=0.30,
                   help="per-tile low-Z band (m above 5th-percentile z) used "
                        "as the MSAC fit set; ground prior that prevents tall "
                        "object clusters from hijacking the plane")
    p.add_argument("--sor-k", type=int, default=12,
                   help="k for final statistical outlier removal (0 to skip)")
    p.add_argument("--sor-std", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="knn-plane",
                   choices=["knn-plane", "knn-quantile", "tile"],
                   help="local-ground estimator: 'knn-plane' (recommended) "
                        "fits a tilted plane to the bottom-Z neighbours of "
                        "each query; 'knn-quantile' uses the q-th percentile "
                        "z directly; 'tile' is the original XY-grid method.")
    p.add_argument("--knn-k", type=int, default=64,
                   help="number of XY neighbours for knn methods")
    p.add_argument("--knn-q", type=float, default=10.0,
                   help="lower-percentile of neighbour Z used as ground prior")
    p.add_argument("--knn-plane-band", type=float, default=0.20,
                   help="for knn-plane: vertical band above the percentile-z "
                        "ground floor, used as the plane fit support set")
    p.add_argument("--save-debug-grid", default=None,
                   help="optional .npz path (tile method only)")
    p.add_argument("--save-rejected", default=None,
                   help="optional path: write the REJECTED points to this PCD "
                        "(useful for visually verifying what was removed)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    log = lambda s: print(s, flush=True)

    if (args.input is None) == (args.input_pkl is None):
        raise SystemExit("specify exactly one of --input or --input-pkl")

    t0 = time.time()
    if args.input_pkl is not None:
        log(f"loading {args.input_pkl} ...")
        xyz, extras, header = load_pkl_to_xyzrgb(args.input_pkl,
                                                  keep_class=args.keep_class)
    else:
        log(f"loading {args.input} ...")
        xyz, extras, header = load_pcd(args.input)
        if args.keep_class is not None:
            log(f"  (note) --keep-class only applies to --input-pkl; ignored")
    if len(xyz) == 0:
        raise SystemExit("no points to denoise after class filter")
    log(f"  {len(xyz):,} points; XYZ extents "
        f"[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}] x "
        f"[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}] x "
        f"[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    keep_mask = np.ones(len(xyz), dtype=bool)

    # 1) Downsample fitting set
    fit_idx = voxel_downsample_indices(xyz, args.fit_voxel)
    fit_xyz = xyz[fit_idx]
    log(f"voxel-fit set: {len(fit_idx):,} points (voxel={args.fit_voxel} m)")

    z_grid = xe = ye = None
    if args.method in ("knn-plane", "knn-quantile"):
        log(f"local ground via {args.method}: k={args.knn_k}, q={args.knn_q}"
            f"{' band='+str(args.knn_plane_band)+'m' if args.method=='knn-plane' else ''}")
        if args.method == "knn-plane":
            z_pred = local_ground_z_knn_planefit(
                xyz[:, :2].astype(np.float64),
                fit_xyz[:, :2].astype(np.float64),
                fit_xyz[:, 2].astype(np.float64),
                k=args.knn_k, q=args.knn_q,
                plane_band=args.knn_plane_band,
            )
        else:
            z_pred = local_ground_z_knn(
                xyz[:, :2].astype(np.float64),
                fit_xyz[:, :2].astype(np.float64),
                fit_xyz[:, 2].astype(np.float64),
                k=args.knn_k, q=args.knn_q,
            )
    else:
        log(f"fitting tiled planes (cell={args.tile_size} m) ...")
        z_grid, xe, ye = fit_tile_planes(
            xyz_fit=fit_xyz,
            cell_xy=args.tile_size,
            msac_thresh=args.msac_thresh,
            msac_iter=args.msac_iter,
            tukey_c=args.tukey_c,
            n_min_z=args.n_min_z,
            min_pts_per_cell=args.min_pts_per_cell,
            rng=rng,
            log=log,
            ground_band=args.ground_band,
        )
        log("evaluating local plane height at every input point ...")
        z_pred = bilinear_sample_grid(z_grid, xe, ye, args.tile_size, xyz)

    dz = xyz[:, 2] - z_pred
    plane_keep = (dz <= args.tau_above) & (dz >= -args.tau_below)
    log(f"  plane filter: kept {int(plane_keep.sum()):,} / {len(xyz):,} "
        f"({100*plane_keep.sum()/len(xyz):.1f} %)")
    log(f"  Δz percentiles dropped (above):  "
        f"{np.percentile(dz[~plane_keep & (dz>0)] if (~plane_keep & (dz>0)).any() else [0], [50,90,99])}")
    keep_mask &= plane_keep

    # 6) SOR cleanup
    if args.sor_k > 0 and keep_mask.sum() > args.sor_k + 1:
        log(f"statistical outlier removal (k={args.sor_k}, σ={args.sor_std}) ...")
        kept_xyz = xyz[keep_mask]
        sor_keep = statistical_outlier_removal(kept_xyz, k=args.sor_k,
                                                std_ratio=args.sor_std)
        log(f"  SOR kept {int(sor_keep.sum()):,} / {len(kept_xyz):,}")
        idx_in_full = np.where(keep_mask)[0]
        keep_mask[idx_in_full[~sor_keep]] = False

    log(f"final: {int(keep_mask.sum()):,} / {len(xyz):,} "
        f"({100*keep_mask.sum()/len(xyz):.2f} %)")

    out_xyz = xyz[keep_mask]
    out_extras = {k: v[keep_mask] for k, v in extras.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pcd(out_path, out_xyz, out_extras, header)
    log(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB) "
        f"in {time.time()-t0:.1f}s")

    if args.save_debug_grid and z_grid is not None:
        np.savez(args.save_debug_grid, z_plane_grid=z_grid,
                 x_edges=xe, y_edges=ye, tile_size=args.tile_size)
        log(f"saved debug grid -> {args.save_debug_grid}")

    if args.save_rejected:
        rej_xyz = xyz[~keep_mask]
        rej_extras = {k: v[~keep_mask] for k, v in extras.items()}
        rej_path = Path(args.save_rejected)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        save_pcd(rej_path, rej_xyz, rej_extras, header)
        log(f"wrote rejected points -> {rej_path} ({len(rej_xyz):,} pts)")


if __name__ == "__main__":
    main()
