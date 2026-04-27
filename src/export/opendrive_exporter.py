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
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def _normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


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
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


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

        headings = []
        for i in range(len(points_xy)):
            if i == len(points_xy) - 1:
                headings.append(_heading(points_xy[i - 1], points_xy[i]))
            else:
                headings.append(_heading(points_xy[i], points_xy[i + 1]))

        out.append(
            {
                "road_id": road_id,
                "points_xy": points_xy,
                "points_z": points_z,
                "widths": widths,
                "s": s,
                "headings": headings,
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


def _fit_arc_k(p0, p1, p2):
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    a = _dist(p0, p1)
    b = _dist(p1, p2)
    c = _dist(p0, p2)
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    s = 0.5 * (a + b + c)
    area_sq = max(s * (s - a) * (s - b) * (s - c), 0.0)
    if area_sq < 1e-12:
        return 0.0
    area = math.sqrt(area_sq)
    k = 4.0 * area / (a * b * c)
    cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
    return -k if cross < 0.0 else k


def _arc_end_from_start(p0, hdg, k, length):
    x0, y0 = p0
    if abs(k) < 1e-9:
        return x0 + length * math.cos(hdg), y0 + length * math.sin(hdg)
    r = 1.0 / k
    hdg2 = hdg + k * length
    x = x0 + r * (math.sin(hdg2) - math.sin(hdg))
    y = y0 - r * (math.cos(hdg2) - math.cos(hdg))
    return x, y


def _make_geom_segments(comp, arc_fit_tol=0.6, arc_k_min=1e-4, arc_k_max=0.25):
    pts = comp["points_xy"]
    s = comp["s"]
    hdgs = comp["headings"]
    segs = []

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        ds = s[i + 1] - s[i]
        if ds < 1e-6:
            continue
        h0 = hdgs[i]

        use_arc = False
        k = 0.0
        if i + 2 < len(pts):
            k = _fit_arc_k(pts[i], pts[i + 1], pts[i + 2])
            if abs(k) >= arc_k_min and abs(k) <= arc_k_max:
                pred = _arc_end_from_start(p0, h0, k, ds)
                err = _dist(pred, p1)
                if err <= arc_fit_tol:
                    use_arc = True

        if use_arc:
            segs.append(
                {
                    "type": "arc",
                    "s": s[i],
                    "x": p0[0],
                    "y": p0[1],
                    "hdg": h0,
                    "length": ds,
                    "curvature": k,
                }
            )
            continue

        # paramPoly3 fallback in local frame, U(s)=s, V(s)=c*s^2+d*s^3
        h1 = hdgs[i + 1] if i + 1 < len(hdgs) else h0
        dh = _normalize_angle(h1 - h0)
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        x_local = dx * math.cos(h0) + dy * math.sin(h0)
        y_local = -dx * math.sin(h0) + dy * math.cos(h0)
        m1 = math.tan(dh)
        c_v = (3.0 * y_local - m1 * ds) / (ds * ds)
        d_v = (m1 * ds - 2.0 * y_local) / (ds * ds * ds)

        # If almost straight and aligned, keep line to reduce verbosity.
        if abs(y_local) < 0.02 and abs(dh) < 0.02 and abs(x_local - ds) < 0.05:
            segs.append(
                {
                    "type": "line",
                    "s": s[i],
                    "x": p0[0],
                    "y": p0[1],
                    "hdg": h0,
                    "length": ds,
                }
            )
        else:
            segs.append(
                {
                    "type": "paramPoly3",
                    "s": s[i],
                    "x": p0[0],
                    "y": p0[1],
                    "hdg": h0,
                    "length": ds,
                    "aU": 0.0,
                    "bU": 1.0,
                    "cU": 0.0,
                    "dU": 0.0,
                    "aV": 0.0,
                    "bV": 0.0,
                    "cV": c_v,
                    "dV": d_v,
                    "pRange": "arcLength",
                }
            )
    return segs


def _cluster_endpoints_for_junctions(components, radius=8.0):
    endpoints = []
    for c in components:
        rid = c["road_id"]
        endpoints.append({"road_id": rid, "end": "start", "pt": c["points_xy"][0]})
        endpoints.append({"road_id": rid, "end": "end", "pt": c["points_xy"][-1]})

    n = len(endpoints)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _dist(endpoints[i]["pt"], endpoints[j]["pt"]) <= radius:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(endpoints[i])

    clusters = []
    for g in groups.values():
        roads = {e["road_id"] for e in g}
        if len(roads) < 2:
            continue
        starts = [e for e in g if e["end"] == "start"]
        ends = [e for e in g if e["end"] == "end"]
        if not starts or not ends:
            continue
        if len(g) < 3:
            continue
        clusters.append({"items": g, "starts": starts, "ends": ends})
    return clusters


def _build_junctions_and_maps(components, direct_pred, direct_succ, junction_radius=8.0):
    clusters = _cluster_endpoints_for_junctions(components, radius=junction_radius)
    if not clusters:
        return [], {}, {}

    road_end_to_junction = {}
    junctions = []
    jid_base = 1000

    for idx, cl in enumerate(clusters):
        jid = str(jid_base + idx)
        connections = []
        for e_in in cl["ends"]:
            for e_out in cl["starts"]:
                if e_in["road_id"] == e_out["road_id"]:
                    continue
                connections.append(
                    {
                        "incomingRoad": str(e_in["road_id"]),
                        "connectingRoad": str(e_out["road_id"]),
                        "contactPoint": "start",
                    }
                )

        if not connections:
            continue

        for it in cl["items"]:
            road_end_to_junction[(it["road_id"], it["end"])] = jid

        junctions.append({"id": jid, "connections": connections})

    # Override direct links only at ends participating in junction.
    pred = dict(direct_pred)
    succ = dict(direct_succ)
    for (rid, end), _jid in road_end_to_junction.items():
        if end == "start":
            pred[rid] = None
        else:
            succ[rid] = None

    return junctions, road_end_to_junction, {"pred": pred, "succ": succ}


def _add_header(root, north, south, east, west):
    header = SubElement(
        root,
        "header",
        {
            "revMajor": "1",
            "revMinor": "4",
            "name": "HDMap Export",
            "version": "0.2",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "north": _fmt(north),
            "south": _fmt(south),
            "east": _fmt(east),
            "west": _fmt(west),
            "vendor": "HDMap",
        },
    )
    SubElement(header, "geoReference").text = "LOCAL_CS"


def _add_plan_view(road_elem, comp, arc_fit_tol, arc_k_min, arc_k_max):
    plan = SubElement(road_elem, "planView")
    geoms = _make_geom_segments(comp, arc_fit_tol=arc_fit_tol, arc_k_min=arc_k_min, arc_k_max=arc_k_max)
    for g in geoms:
        geom = SubElement(
            plan,
            "geometry",
            {
                "s": _fmt(g["s"]),
                "x": _fmt(g["x"]),
                "y": _fmt(g["y"]),
                "hdg": _fmt(g["hdg"]),
                "length": _fmt(g["length"]),
            },
        )
        if g["type"] == "line":
            SubElement(geom, "line")
        elif g["type"] == "arc":
            SubElement(geom, "arc", {"curvature": _fmt(g["curvature"])})
        else:
            SubElement(
                geom,
                "paramPoly3",
                {
                    "aU": _fmt(g["aU"]),
                    "bU": _fmt(g["bU"]),
                    "cU": _fmt(g["cU"]),
                    "dU": _fmt(g["dU"]),
                    "aV": _fmt(g["aV"]),
                    "bV": _fmt(g["bV"]),
                    "cV": _fmt(g["cV"]),
                    "dV": _fmt(g["dV"]),
                    "pRange": g["pRange"],
                },
            )


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
            {"s": _fmt(s[i]), "a": _fmt(z[i]), "b": _fmt(b), "c": "0", "d": "0"},
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


def export_opendrive(
    input_path,
    output_path,
    link_dist_thresh=12.0,
    junction_radius=8.0,
    arc_fit_tol=0.6,
    arc_k_min=1e-4,
    arc_k_max=0.25,
):
    data = json.loads(Path(input_path).read_text())
    comps = _collect_components(data)

    all_pts = [pt for c in comps for pt in c["points_xy"]]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    root = Element("OpenDRIVE")
    _add_header(root, north=max(ys), south=min(ys), east=max(xs), west=min(xs))

    direct_pred, direct_succ = _build_road_links(comps, link_dist_thresh=link_dist_thresh)
    junctions, end2junc, links = _build_junctions_and_maps(
        comps,
        direct_pred,
        direct_succ,
        junction_radius=junction_radius,
    )
    pred_map = links.get("pred", direct_pred)
    succ_map = links.get("succ", direct_succ)

    for comp in comps:
        rid = comp["road_id"]
        road = SubElement(
            root,
            "road",
            {
                "name": f"road_{rid}",
                "length": _fmt(comp["s"][-1]),
                "id": str(rid),
                "junction": end2junc.get((rid, "start"), end2junc.get((rid, "end"), "-1")),
            },
        )

        link = SubElement(road, "link")

        if (rid, "start") in end2junc:
            SubElement(link, "predecessor", {"elementType": "junction", "elementId": end2junc[(rid, "start")]})
        elif pred_map.get(rid) is not None:
            SubElement(link, "predecessor", {"elementType": "road", "elementId": str(pred_map[rid]), "contactPoint": "end"})

        if (rid, "end") in end2junc:
            SubElement(link, "successor", {"elementType": "junction", "elementId": end2junc[(rid, "end")]})
        elif succ_map.get(rid) is not None:
            SubElement(link, "successor", {"elementType": "road", "elementId": str(succ_map[rid]), "contactPoint": "start"})

        _add_plan_view(road, comp, arc_fit_tol=arc_fit_tol, arc_k_min=arc_k_min, arc_k_max=arc_k_max)
        _add_elevation(road, comp)
        _add_lanes(road, comp)

    for j in junctions:
        je = SubElement(root, "junction", {"id": j["id"], "name": f"junction_{j['id']}"})
        for cid, conn in enumerate(j["connections"]):
            ce = SubElement(
                je,
                "connection",
                {
                    "id": str(cid),
                    "incomingRoad": conn["incomingRoad"],
                    "connectingRoad": conn["connectingRoad"],
                    "contactPoint": conn["contactPoint"],
                },
            )
            SubElement(ce, "laneLink", {"from": "1", "to": "1"})
            SubElement(ce, "laneLink", {"from": "-1", "to": "-1"})

    xml_bytes = tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    Path(output_path).write_bytes(pretty)


def parse_args():
    p = argparse.ArgumentParser(description="Export fused road JSON to OpenDRIVE (.xodr)")
    p.add_argument("-i", "--input", required=True, help="Input fused road JSON")
    p.add_argument("-o", "--output", required=True, help="Output .xodr path")
    p.add_argument("--link-dist-thresh", type=float, default=12.0, help="Max distance for direct road predecessor/successor linking")
    p.add_argument("--junction-radius", type=float, default=8.0, help="Endpoint clustering radius for junction detection")
    p.add_argument("--arc-fit-tol", type=float, default=0.6, help="Arc fitting endpoint tolerance")
    p.add_argument("--arc-k-min", type=float, default=1e-4, help="Minimum curvature magnitude for arc")
    p.add_argument("--arc-k-max", type=float, default=0.25, help="Maximum curvature magnitude for arc")
    return p.parse_args()


def main():
    args = parse_args()
    export_opendrive(
        args.input,
        args.output,
        link_dist_thresh=float(args.link_dist_thresh),
        junction_radius=float(args.junction_radius),
        arc_fit_tol=float(args.arc_fit_tol),
        arc_k_min=float(args.arc_k_min),
        arc_k_max=float(args.arc_k_max),
    )
    print(f"saved OpenDRIVE to {args.output}")


if __name__ == "__main__":
    main()
