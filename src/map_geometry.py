import json
from pathlib import Path

import numpy as np


def as_xyz_array(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=np.float64)
        arr = np.concatenate((arr, pad), axis=1)
    return arr[:, :3]


def as_xyz_array_with_default_z(points, default_z=0.0):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 3:
        return arr[:, :3]
    z_col = np.full((arr.shape[0], 1), float(default_z), dtype=np.float64)
    return np.concatenate((arr[:, :2], z_col), axis=1)


def load_json(input_path):
    input_path = Path(input_path)
    with input_path.open("r") as f:
        return json.load(f)


def load_fused_road_map_geometries(input_path):
    input_path = Path(input_path)
    data = load_json(input_path)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict-like fused road json in {input_path}, got {type(data).__name__}")

    items = []
    candidates = [
        ("left_edge", (255, 80, 80), 5),
        ("right_edge", (80, 220, 80), 5),
        ("center_line", (0, 255, 255), 5),
    ]
    for key, color, thickness in candidates:
        points_xyz = as_xyz_array(data.get(key, []))
        if len(points_xyz) == 0:
            continue
        items.append(
            {
                "geometry_type": "polyline",
                "name": key,
                "source_type": "fused_road",
                "points_xyz": points_xyz,
                "color": color,
                "thickness": thickness,
            }
        )

    return {
        "source_path": str(input_path),
        "source_type": "fused_road",
        "meta": data.get("meta", {}),
        "items": items,
    }


def load_fused_sidewalk_map_geometries(input_path):
    input_path = Path(input_path)
    data = load_json(input_path)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict-like fused sidewalk json in {input_path}, got {type(data).__name__}")

    items = []
    if "sidewalks" in data:
        for idx, record in enumerate(data.get("sidewalks", []), start=1):
            points = record.get("outline") or []
            points_xyz = as_xyz_array_with_default_z(points, default_z=record.get("sidewalk_z", 0.0))
            if len(points_xyz) == 0:
                continue
            item_name = record.get("id") or f"sidewalk_{idx}"
            items.append(
                {
                    "geometry_type": "polyline",
                    "name": str(item_name),
                    "source_type": "fused_sidewalk",
                    "points_xyz": points_xyz,
                    "color": (0, 220, 255),
                    "thickness": 4,
                }
            )
        return {
            "source_path": str(input_path),
            "source_type": "fused_sidewalk",
            "meta": data.get("meta", {}),
            "items": items,
        }

    side_specs = [
        ("left_sidewalks", "left_sidewalk", (0, 220, 255), 4),
        ("right_sidewalks", "right_sidewalk", (255, 180, 0), 4),
    ]
    for side_key, name_prefix, color, thickness in side_specs:
        for idx, record in enumerate(data.get(side_key, []), start=1):
            points = record.get("polyline") or record.get("outline") or []
            points_xyz = as_xyz_array_with_default_z(points, default_z=record.get("sidewalk_z", 0.0))
            if len(points_xyz) == 0:
                continue
            item_name = record.get("id") or f"{name_prefix}_{idx}"
            items.append(
                {
                    "geometry_type": "polyline",
                    "name": str(item_name),
                    "source_type": "fused_sidewalk",
                    "points_xyz": points_xyz,
                    "color": color,
                    "thickness": thickness,
                }
            )

    return {
        "source_path": str(input_path),
        "source_type": "fused_sidewalk",
        "meta": data.get("meta", {}),
        "items": items,
    }


def load_map_geometries(input_path):
    input_path = Path(input_path)
    data = load_json(input_path)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict-like map json in {input_path}, got {type(data).__name__}")
    if any(key in data for key in ("left_edge", "right_edge", "center_line")):
        return load_fused_road_map_geometries(input_path)
    if any(key in data for key in ("sidewalks", "left_sidewalks", "right_sidewalks")):
        return load_fused_sidewalk_map_geometries(input_path)
    raise ValueError(f"unsupported map geometry schema in {input_path}: keys={sorted(data.keys())[:10]}")


def split_polyline_by_mask(points_xyz, keep_mask, min_points=1):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must have shape (N, >=3), got {points_xyz.shape}")
    if keep_mask.shape != (len(points_xyz),):
        raise ValueError(f"keep_mask must have shape ({len(points_xyz)},), got {keep_mask.shape}")

    segments = []
    start = None
    for idx, keep in enumerate(keep_mask.tolist()):
        if keep and start is None:
            start = idx
            continue
        if not keep and start is not None:
            if idx - start >= int(min_points):
                segments.append(points_xyz[start:idx])
            start = None

    if start is not None and len(points_xyz) - start >= int(min_points):
        segments.append(points_xyz[start:])
    return segments


def filter_geometries_by_distance(items, origin_xyz, max_distance):
    if max_distance <= 0:
        return list(items)

    kept = []
    origin_xy = np.asarray(origin_xyz[:2], dtype=np.float64)
    for item in items:
        points_xyz = np.asarray(item["points_xyz"], dtype=np.float64)
        dist = np.linalg.norm(points_xyz[:, :2] - origin_xy[None, :], axis=1)
        mask = dist <= float(max_distance)
        for segment_idx, segment_points_xyz in enumerate(split_polyline_by_mask(points_xyz, mask)):
            segment_name = item["name"]
            if segment_idx > 0:
                segment_name = f"{segment_name}_segment_{segment_idx + 1}"
            kept.append({**item, "name": segment_name, "points_xyz": segment_points_xyz})
    return kept
