#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.9f}".rstrip("0").rstrip(".")
    return str(v)


def _as_xy(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    raise ValueError(f"expected 2D value, got {v}")


def _dist(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return math.hypot(dx, dy)


def _station(points_xy):
    s = [0.0]
    for i in range(1, len(points_xy)):
        s.append(s[-1] + _dist(points_xy[i - 1], points_xy[i]))
    return s


def _heading(p1, p2):
    return math.atan2(float(p2[1]) - float(p1[1]), float(p2[0]) - float(p1[0]))


def _median(values, default=0.0):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float(default)
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _collect_components(data):
    topo = data.get("topology", {})
    nodes = topo.get("nodes", [])
    boundary_pairs = topo.get("boundary_pairs", [])
    components = topo.get("components", [])
    if not nodes or not components:
        raise ValueError("fused file has no topology nodes/components")

    id_to_node_idx = {n["id"]: i for i, n in enumerate(nodes)}
    pair_map = {bp["node_id"]: bp for bp in boundary_pairs}

    out = []
    for road_id, comp in enumerate(components, start=1):
        sid = comp["start_node"]
        eid = comp["end_node"]
        if sid not in id_to_node_idx or eid not in id_to_node_idx:
            continue

        i0 = id_to_node_idx[sid]
        i1 = id_to_node_idx[eid]
        if i1 < i0:
            i0, i1 = i1, i0

        node_slice = nodes[i0 : i1 + 1]
        points_xy = [_as_xy(n["center"]) for n in node_slice]
        points_z = [float(n.get("center_z", n.get("road_z", 0.0))) for n in node_slice]

        widths = []
        for n in node_slice:
            bp = pair_map.get(n["id"])
            if bp is None:
                widths.append(float("nan"))
                continue
            lx, ly = _as_xy(bp["left_edge"])
            rx, ry = _as_xy(bp["right_edge"])
            widths.append(math.hypot(lx - rx, ly - ry))

        width_med = _median(widths, default=6.0)
        widths = [w if math.isfinite(w) else width_med for w in widths]

        s = _station(points_xy)
        if len(s) < 2 or s[-1] < 1e-3:
            continue

        out.append(
            {
                "road_id": road_id,
                "points_xy": points_xy,
                "points_z": points_z,
                "widths": widths,
                "s": s,
            }
        )

    if not out:
        raise ValueError("no valid road components after conversion")
    return out


def _build_road_links(components, link_dist_thresh):
    pred = {c["road_id"]: None for c in components}
    succ = {c["road_id"]: None for c in components}
    starts = {c["road_id"]: c["points_xy"][0] for c in components}
    ends = {c["road_id"]: c["points_xy"][-1] for c in components}

    for c in components:
        rid = c["road_id"]
        end_pt = ends[rid]
        best_id = None
        best_d = float("inf")
        for o in components:
            oid = o["road_id"]
            if oid == rid:
                continue
            d = _dist(starts[oid], end_pt)
            if d < best_d:
                best_d = d
                best_id = oid
        if best_id is not None and best_d <= link_dist_thresh:
            succ[rid] = best_id
            if pred[best_id] is None:
                pred[best_id] = rid
    return pred, succ


def _add_header(root, north, south, east, west):
    header = SubElement(
        root,
        "header",
        {
            "revMajor": "1",
            "revMinor": "4",
            "name": "HDMap Export",
            "version": "0.1",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "north": _fmt(north),
            "south": _fmt(south),
            "east": _fmt(east),
            "west": _fmt(west),
            "vendor": "HDMap",
        },
    )
    SubElement(header, "geoReference").text = "LOCAL_CS"


def _add_plan_view(road_elem, comp):
    plan = SubElement(road_elem, "planView")
    pts = comp["points_xy"]
    s = comp["s"]
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        length = _dist(p1, p2)
        if length < 1e-6:
            continue
        geom = SubElement(
            plan,
            "geometry",
            {
                "s": _fmt(s[i]),
                "x": _fmt(p1[0]),
                "y": _fmt(p1[1]),
                "hdg": _fmt(_heading(p1, p2)),
                "length": _fmt(length),
            },
        )
        SubElement(geom, "line")


def _add_elevation(road_elem, comp):
    elev = SubElement(road_elem, "elevationProfile")
    z = comp["points_z"]
    s = comp["s"]
    for i in range(len(z) - 1):
        ds = float(s[i + 1] - s[i])
        if ds < 1e-6:
            continue
        b = float((z[i + 1] - z[i]) / ds)
        SubElement(
            elev,
            "elevation",
            {
                "s": _fmt(s[i]),
                "a": _fmt(z[i]),
                "b": _fmt(b),
                "c": "0",
                "d": "0",
            },
        )


def _add_lanes(road_elem, comp):
    lanes = SubElement(road_elem, "lanes")
    SubElement(lanes, "laneOffset", {"s": "0", "a": "0", "b": "0", "c": "0", "d": "0"})
    section = SubElement(lanes, "laneSection", {"s": "0"})

    left = SubElement(section, "left")
    center = SubElement(section, "center")
    right = SubElement(section, "right")

    center_lane = SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
    SubElement(center_lane, "roadMark", {"sOffset": "0", "type": "none", "weight": "standard", "color": "standard", "width": "0.0"})

    half_width = max(_median(comp["widths"], default=6.0) * 0.5, 0.5)

    left_lane = SubElement(left, "lane", {"id": "1", "type": "driving", "level": "false"})
    SubElement(left_lane, "link")
    SubElement(left_lane, "width", {"sOffset": "0", "a": _fmt(half_width), "b": "0", "c": "0", "d": "0"})
    SubElement(left_lane, "roadMark", {"sOffset": "0", "type": "solid", "weight": "standard", "color": "standard", "width": "0.15"})

    right_lane = SubElement(right, "lane", {"id": "-1", "type": "driving", "level": "false"})
    SubElement(right_lane, "link")
    SubElement(right_lane, "width", {"sOffset": "0", "a": _fmt(half_width), "b": "0", "c": "0", "d": "0"})
    SubElement(right_lane, "roadMark", {"sOffset": "0", "type": "solid", "weight": "standard", "color": "standard", "width": "0.15"})


def export_opendrive(input_path, output_path, link_dist_thresh=12.0):
    data = json.loads(Path(input_path).read_text())
    comps = _collect_components(data)

    all_pts = [pt for c in comps for pt in c["points_xy"]]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    root = Element("OpenDRIVE")
    _add_header(root, north=max(ys), south=min(ys), east=max(xs), west=min(xs))

    pred_map, succ_map = _build_road_links(comps, link_dist_thresh=link_dist_thresh)

    for comp in comps:
        rid = comp["road_id"]
        road = SubElement(
            root,
            "road",
            {
                "name": f"road_{rid}",
                "length": _fmt(comp["s"][-1]),
                "id": str(rid),
                "junction": "-1",
            },
        )
        link = SubElement(road, "link")
        if pred_map.get(rid) is not None:
            SubElement(link, "predecessor", {"elementType": "road", "elementId": str(pred_map[rid]), "contactPoint": "end"})
        if succ_map.get(rid) is not None:
            SubElement(link, "successor", {"elementType": "road", "elementId": str(succ_map[rid]), "contactPoint": "start"})

        _add_plan_view(road, comp)
        _add_elevation(road, comp)
        _add_lanes(road, comp)

    xml_bytes = tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    Path(output_path).write_bytes(pretty)


def parse_args():
    p = argparse.ArgumentParser(description="Export fused road JSON to OpenDRIVE (.xodr)")
    p.add_argument("-i", "--input", required=True, help="Input fused road JSON")
    p.add_argument("-o", "--output", required=True, help="Output .xodr path")
    p.add_argument("--link-dist-thresh", type=float, default=12.0, help="Max distance for road predecessor/successor linking")
    return p.parse_args()


def main():
    args = parse_args()
    export_opendrive(args.input, args.output, link_dist_thresh=float(args.link_dist_thresh))
    print(f"saved OpenDRIVE to {args.output}")


if __name__ == "__main__":
    main()
