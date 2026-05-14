#!/usr/bin/python3
"""In-place sidewalk-class noise removal on outdoor.pkl.

Backs the original up to <pkl>.bak (refusing to overwrite an existing backup),
then strips the human-body / above-plane points that leaked into the sidewalk
class (Mapillary id 15). All other classes are kept untouched, and the pkl is
re-written with the same per-frame layout.

Reuses the knn-plane denoiser from tools/denoise_sidewalk_pcd.py so both tools
stay in sync.
"""
import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise_sidewalk_pcd import (   # noqa: E402
    voxel_downsample_indices,
    local_ground_z_knn_planefit,
    local_ground_z_knn,
    statistical_outlier_removal,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="path to outdoor.pkl (will be rewritten in-place)")
    p.add_argument("--keep-class", type=int, default=15,
                   help="semantic class id to denoise (default 15 = sidewalk)")
    p.add_argument("--method", default="knn-plane",
                   choices=["knn-plane", "knn-quantile"])
    p.add_argument("--fit-voxel", type=float, default=0.05)
    p.add_argument("--knn-k", type=int, default=64)
    p.add_argument("--knn-q", type=float, default=10.0)
    p.add_argument("--knn-plane-band", type=float, default=0.20)
    p.add_argument("--tau-above", type=float, default=0.20)
    p.add_argument("--tau-below", type=float, default=0.10)
    p.add_argument("--sor-k", type=int, default=12)
    p.add_argument("--sor-std", type=float, default=2.0)
    p.add_argument("--backup-suffix", default=".bak",
                   help="suffix for backup file; refuses to clobber an existing one")
    p.add_argument("--dry-run", action="store_true",
                   help="run the analysis and report stats, but do NOT modify the pkl")
    return p.parse_args()


def denoise_pkl_class_inplace(
    pkl_path,
    keep_class=15,
    method="knn-plane",
    fit_voxel=0.05,
    knn_k=64,
    knn_q=10.0,
    knn_plane_band=0.20,
    tau_above=0.20,
    tau_below=0.10,
    sor_k=12,
    sor_std=2.0,
    backup_suffix=".bak",
    overwrite_backup=False,
    dry_run=False,
    log=print,
):
    """Strip noise points of `keep_class` from a per-frame-pickled pkl.

    Returns a dict with stats: {'frames', 'n_total_before', 'n_total_after',
    'n_target_before', 'n_target_after', 'n_dropped'}.

    Raises:
        FileNotFoundError if pkl_path is missing.
        FileExistsError if backup exists and overwrite_backup is False.
        RuntimeError on size mismatch / IO failures.

    `dry_run=True` runs the analysis (returning stats) without touching disk.
    Other classes in the pkl are passed through unchanged.
    """
    pkl = Path(pkl_path)
    if not pkl.is_file():
        raise FileNotFoundError(f"pkl not found: {pkl}")
    backup = Path(str(pkl) + backup_suffix)

    if not dry_run:
        if backup.exists() and not overwrite_backup:
            raise FileExistsError(
                f"backup exists: {backup}; pass overwrite_backup=True or "
                f"move it aside.")
        log(f"backing up {pkl} -> {backup}")
        shutil.copy2(pkl, backup)
        if backup.stat().st_size != pkl.stat().st_size:
            raise RuntimeError(
                f"backup size mismatch ({backup.stat().st_size} vs "
                f"{pkl.stat().st_size}); aborting before rewrite.")

    log(f"loading {pkl} ...")
    frames = []
    with open(pkl, "rb") as f:
        while True:
            try:
                frames.append(pickle.load(f))
            except EOFError:
                break
    if not frames:
        log("empty pkl; nothing to do")
        return {"frames": 0, "n_total_before": 0, "n_total_after": 0,
                "n_target_before": 0, "n_target_after": 0, "n_dropped": 0}
    sizes = np.asarray([len(a) for a in frames], dtype=np.int64)
    n_total = int(sizes.sum())
    log(f"  {len(frames)} frames, {n_total:,} points total")

    all_pts = np.vstack(frames).astype(np.float32, copy=False)   # (N, 4)
    cls = all_pts[:, 3].astype(np.int64)
    target_mask = cls == int(keep_class)
    n_target = int(target_mask.sum())
    if n_target == 0:
        log(f"  no points with class={keep_class}; nothing to do")
        return {"frames": len(frames), "n_total_before": n_total,
                "n_total_after": n_total, "n_target_before": 0,
                "n_target_after": 0, "n_dropped": 0}
    log(f"  class={keep_class}: {n_target:,} points "
        f"({100*n_target/n_total:.2f} %)")

    target_xyz = all_pts[target_mask, :3].astype(np.float64, copy=False)
    fit_idx = voxel_downsample_indices(target_xyz.astype(np.float32), fit_voxel)
    fit_xyz = target_xyz[fit_idx]
    log(f"  voxel-fit set: {len(fit_idx):,} points (voxel={fit_voxel} m)")
    if len(fit_idx) < max(knn_k, 16):
        log(f"  too few fit points ({len(fit_idx)}) to run knn (k={knn_k}); "
            f"skipping denoise.")
        return {"frames": len(frames), "n_total_before": n_total,
                "n_total_after": n_total, "n_target_before": n_target,
                "n_target_after": n_target, "n_dropped": 0}
    log(f"  local ground via {method}: k={knn_k}, q={knn_q}"
        f"{' band='+str(knn_plane_band)+'m' if method=='knn-plane' else ''}")
    if method == "knn-plane":
        z_pred = local_ground_z_knn_planefit(
            target_xyz[:, :2], fit_xyz[:, :2], fit_xyz[:, 2],
            k=knn_k, q=knn_q, plane_band=knn_plane_band,
        )
    else:
        z_pred = local_ground_z_knn(
            target_xyz[:, :2], fit_xyz[:, :2], fit_xyz[:, 2],
            k=knn_k, q=knn_q,
        )
    dz = target_xyz[:, 2] - z_pred
    plane_keep = (dz <= tau_above) & (dz >= -tau_below)
    log(f"  plane filter: kept {int(plane_keep.sum()):,} / {n_target:,} "
        f"({100*plane_keep.sum()/n_target:.1f} %)")

    target_keep = plane_keep.copy()
    if sor_k > 0 and target_keep.sum() > sor_k + 1:
        log(f"  SOR (k={sor_k}, sigma={sor_std}) ...")
        kept_xyz = target_xyz[target_keep]
        sor_keep = statistical_outlier_removal(
            kept_xyz.astype(np.float32), k=sor_k, std_ratio=sor_std,
        )
        idx_in_target = np.where(target_keep)[0]
        target_keep[idx_in_target[~sor_keep]] = False
        log(f"  SOR kept {int(target_keep.sum()):,} target points")

    global_keep = np.ones(n_total, dtype=bool)
    target_indices = np.where(target_mask)[0]
    global_keep[target_indices[~target_keep]] = False
    n_drop = int((~global_keep).sum())
    log(f"  dropping {n_drop:,} target-noise points "
        f"({100*n_drop/n_total:.3f} % of total pkl)")

    boundaries = np.concatenate([[0], np.cumsum(sizes)])
    new_frames = []
    per_frame_drop = []
    for i, arr in enumerate(frames):
        a, b = boundaries[i], boundaries[i + 1]
        m = global_keep[a:b]
        new_frames.append(np.asarray(arr)[m])
        per_frame_drop.append(int((~m).sum()))
    drops = np.asarray(per_frame_drop)
    nz = drops[drops > 0]
    if len(nz):
        log(f"  per-frame drops: {len(nz)}/{len(frames)} frames affected; "
            f"min={int(nz.min())}, median={int(np.median(nz))}, "
            f"max={int(nz.max())} drops/frame")

    stats = {
        "frames": len(frames),
        "n_total_before": n_total,
        "n_total_after": n_total - n_drop,
        "n_target_before": n_target,
        "n_target_after": int(target_keep.sum()),
        "n_dropped": n_drop,
    }

    if dry_run:
        log("[dry-run] not modifying pkl.")
        return stats

    tmp = Path(str(pkl) + ".tmp")
    log(f"writing {tmp} ...")
    t0 = time.time()
    with open(tmp, "wb") as f:
        for arr in new_frames:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  {tmp.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s")
    tmp.replace(pkl)
    log(f"renamed -> {pkl}")
    return stats


def main():
    args = parse_args()
    try:
        denoise_pkl_class_inplace(
            pkl_path=args.pkl,
            keep_class=args.keep_class,
            method=args.method,
            fit_voxel=args.fit_voxel,
            knn_k=args.knn_k,
            knn_q=args.knn_q,
            knn_plane_band=args.knn_plane_band,
            tau_above=args.tau_above,
            tau_below=args.tau_below,
            sor_k=args.sor_k,
            sor_std=args.sor_std,
            backup_suffix=args.backup_suffix,
            overwrite_backup=False,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
