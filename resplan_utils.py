"""
resplan_utils.py â€” Nonâ€“deep-learning helpers for the ResPlan-style floorplan datasets.

Dependencies (install as needed):
    pip install shapely geopandas matplotlib networkx numpy opencv-python

Contents:
    - Color maps and constants
    - Geometry utilities (get_geometries, centroid, perturb, noise)
    - Mask conversion (geometry_to_mask)
    - Augmentations (rotate/flip/scale)
    - Buffer helpers (shrinkâ†’expand, expandâ†’shrink)
    - Plan plotting (plot_plan)
    - Planâ†’graph (plan_to_graph) + graph overlay plotting (plot_plan_and_graph)
    - Dataset helpers (normalize_keys, get_plan_width)
"""

from __future__ import annotations
import math
from typing import Iterable, List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection, base, box
)
from shapely.ops import unary_union
from shapely import affinity

# -----------------------------
# Colors & constants
# -----------------------------

CATEGORY_COLORS: Dict[str, str] = {
    "living": "#d9d9d9",     # light gray
    "bedroom": "#66c2a5",    # greenish
    "bathroom": "#fc8d62",   # orange
    "kitchen": "#8da0cb",    # blue
    "door": "#e78ac3",       # pink
    "window": "#a6d854",     # lime
    "wall": "#ffd92f",       # yellow
    "front_door": "#a63603", # dark reddish-brown
    "balcony": "#b3b3b3"     # dark gray
}

DEFAULT_CANVAS_SIZE = (256, 256)  # (H, W)

# -----------------------------
# Dataset helpers
# -----------------------------

def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common key typos / variations in-place (balaconyâ†’balcony)."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan

def get_plan_width(plan: Dict[str, Any]) -> float:
    """Returns the max(width, height) of the inner polygon bounds."""
    inner = plan.get("inner")
    if inner is None or inner.is_empty:
        return 0.0
    x1, y1, x2, y2 = inner.bounds
    return max(x2 - x1, y2 - y1)

# -----------------------------
# Geometry utilities
# -----------------------------

def get_geometries(geom_data: Any) -> List[Any]:
    """Safely extract individual geometries from single/multi/collections."""
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []

def centroid(poly: Union[Polygon, MultiPolygon]) -> Point:
    """Centroid for Polygon/MultiPolygon (largest part if multi)."""
    if isinstance(poly, Polygon):
        return poly.centroid
    if isinstance(poly, MultiPolygon) and len(poly.geoms) > 0:
        largest = max(poly.geoms, key=lambda p: p.area)
        return largest.centroid
    return Point(-1e6, -1e6)

def perturb_polygon(polygon: Polygon, x_range: Tuple[float, float]=(-2, 2),
                    y_range: Tuple[float, float]=(-2, 2)) -> Polygon:
    """Apply random per-vertex perturbation to a polygon."""
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    dx = np.random.uniform(x_range[0], x_range[1], size=len(coords))
    dy = np.random.uniform(y_range[0], y_range[1], size=len(coords))
    perturbed = np.column_stack([coords[:,0] + dx, coords[:,1] + dy])
    return Polygon(perturbed)

def noise(point: Point, noise_scale: float = 10.0) -> Point:
    """Jitter a point by uniform noise within Â±noise_scale."""
    x, y = point.x, point.y
    return Point(x + np.random.uniform(-noise_scale, noise_scale),
                 y + np.random.uniform(-noise_scale, noise_scale))

# -----------------------------
# Augmentations
# -----------------------------

def augment_geom(geom: base.BaseGeometry,
                 degree: float = 0.0,
                 flip_vertical: bool = False,
                 scale: float = 1.0,
                 size: int = 256) -> base.BaseGeometry:
    """Rotate around image center, optional vertical flip (via negative y-scale), and scale."""
    if geom is None:
        return Point(-1e6, -1e6)
    g = affinity.rotate(geom, degree, origin=(size/2, size/2))
    flip = -1.0 if flip_vertical else 1.0
    return affinity.scale(g, xfact=scale, yfact=scale * flip, origin=(size/2, size/2))

# -----------------------------
# Buffer helpers
# -----------------------------

def buffer_shrink_expand(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Shrink then expand by w (useful for cleaning)."""
    return geom.buffer(-w, join_style=join_style, cap_style=cap_style)                   .buffer(+w, join_style=join_style, cap_style=cap_style)

def buffer_expand_shrink(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Expand then shrink by w (useful for filling tiny gaps)."""
    return geom.buffer(+w, join_style=join_style, cap_style=cap_style)                   .buffer(-w, join_style=join_style, cap_style=cap_style)

# -----------------------------
# Geometry â†’ mask
# -----------------------------

def _poly_to_mask(poly: Polygon, shape: Tuple[int, int], line_thickness: int = 0) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    if line_thickness > 0:
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=line_thickness)
    else:
        cv2.fillPoly(img, [pts], color=255)
    for interior in poly.interiors:
        pts_in = np.array(interior.coords, dtype=np.int32)
        if line_thickness > 0:
            cv2.polylines(img, [pts_in], isClosed=True, color=0, thickness=line_thickness)
        else:
            cv2.fillPoly(img, [pts_in], color=0)
    return img

def geometry_to_mask(geom: Any,
                     shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
                     point_radius: int = 5,
                     line_thickness: int = 0) -> np.ndarray:
    """Rasterize Polygon/MultiPolygon/LineString/Point/iterables to a binary mask [0,255]."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Single geometry
    if isinstance(geom, Polygon):
        return _poly_to_mask(geom, shape, line_thickness)
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out = np.maximum(out, _poly_to_mask(p, shape, line_thickness))
        return out
    if isinstance(geom, LineString):
        pts = np.array(geom.coords, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, MultiLineString):
        for ls in geom.geoms:
            pts = np.array(ls.coords, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, Point):
        cx, cy = int(round(geom.x)), int(round(geom.y))
        cv2.circle(out, (cx, cy), point_radius, 255, -1)
        return out
    if isinstance(geom, Iterable):
        for g in geom:
            out = np.maximum(out, geometry_to_mask(g, shape, point_radius, line_thickness))
        return out
    # Unrecognized â†’ empty
    return out

# -----------------------------
# Plotting
# -----------------------------

def plot_plan(plan: Dict[str, Any],
              categories: Optional[List[str]] = None,
              colors: Dict[str, str] = CATEGORY_COLORS,
              ax: Optional[plt.Axes] = None,
              legend: bool = True,
              title: Optional[str] = None,
              tight: bool = True) -> plt.Axes:
    """Plot a single plan with colored layers."""
    plan = normalize_keys(plan)
    if categories is None:
        categories = ["living","bedroom","bathroom","kitchen","door","window","wall","front_door","balcony"]

    geoms, color_list, present = [], [], []
    for key in categories:
        geom = plan.get(key)
        if geom is None:
            continue
        parts = get_geometries(geom)
        if not parts:
            continue
        geoms.extend(parts)
        color_list.extend([colors.get(key, "#000000")] * len(parts))
        present.append(key)

    if not geoms:
        raise ValueError("No geometries to plot.")

    gseries = gpd.GeoSeries(geoms)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    gseries.plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if legend:
        from matplotlib.patches import Patch
        uniq_present = list(dict.fromkeys(present))  # preserve order
        handles = [Patch(facecolor=colors.get(k, "#000000"), edgecolor="black", label=k.replace("_"," ")) for k in uniq_present]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1,1), frameon=False)

    if tight:
        plt.tight_layout()
    return ax

# -----------------------------
# Plan â†’ Graph
# -----------------------------

def plan_to_graph(plan: Dict[str, Any],
                  buffer_factor: float = 0.75) -> nx.Graph:
    """Create a simple room graph: nodes are room parts; edges denote adjacency or connections via door/window."""
    plan = normalize_keys(plan)
    G = nx.Graph()
    ww = float(plan.get("wall_width", 0.1) or 0.1)
    buf = max(ww * buffer_factor, 0.01)

    nodes_by_type: Dict[str, List[str]] = {k: [] for k in ["living","kitchen","bedroom","bathroom","balcony","front_door"]}

    # rooms
    for room_type in ["living","kitchen","bedroom","bathroom","balcony"]:
        parts = get_geometries(plan.get(room_type))
        # for living, keep separate parts; user can union beforehand if desired
        for i, geom in enumerate(parts):
            if isinstance(geom, Polygon) and not geom.is_empty:
                nid = f"{room_type}_{i}"
                G.add_node(nid, geometry=geom, type=room_type, area=geom.area)
                nodes_by_type[room_type].append(nid)

    # front door (may be line/polygon)
    for i, geom in enumerate(get_geometries(plan.get("front_door"))):
        nid = f"front_door_{i}"
        G.add_node(nid, geometry=geom, type="front_door", area=getattr(geom, "area", 0.0))
        nodes_by_type["front_door"].append(nid)

    doors  = get_geometries(plan.get("door"))
    wins   = get_geometries(plan.get("window"))
    conns  = [(d, "via_door") for d in doors] + [(w, "via_window") for w in wins]

    # front_door â†’ living
    for fd in nodes_by_type["front_door"]:
        fd_geom = G.nodes[fd]["geometry"]
        for gen in nodes_by_type["living"]:
            gen_geom = G.nodes[gen]["geometry"]
            if fd_geom.intersects(gen_geom.buffer(buf)):
                G.add_edge(fd, gen, type="direct")

    # adjacency: kitchen/bedroom â†” living
    for room_type in ["kitchen","bedroom"]:
        for rn in nodes_by_type[room_type]:
            rgeom = G.nodes[rn]["geometry"].buffer(buf)
            for gen in nodes_by_type["living"]:
                gen_geom = G.nodes[gen]["geometry"]
                if rgeom.buffer(buf).intersects(gen_geom.buffer(buf)):
                    G.add_edge(rn, gen, type="adjacency")

    # bathroom & balcony connections via door/window to living/bedroom
    for room_type in ["bathroom","balcony"]:
        for rn in nodes_by_type[room_type]:
            rgeom = G.nodes[rn]["geometry"].buffer(buf)
            for cgeom, ctype in conns:
                if not cgeom.intersects(rgeom):
                    continue
                for target_type in ["living","bedroom"]:
                    for tn in nodes_by_type[target_type]:
                        tgeom = G.nodes[tn]["geometry"].buffer(buf)
                        if cgeom.intersects(tgeom):
                            if not G.has_edge(rn, tn):
                                G.add_edge(rn, tn, type=ctype)
    return G

# -----------------------------
# Graph overlay on plan
# -----------------------------

def plot_plan_and_graph(plan: Dict[str, Any],
                        ax: Optional[plt.Axes] = None,
                        node_scale: Tuple[float,float]=(150, 1000),
                        title: Optional[str] = None) -> plt.Axes:
    """Plot plan and overlay the room graph (node size scaled by room area)."""
    G = plan["graph"] if "graph" in plan else plan_to_graph(plan)
    ax = plot_plan(plan, legend=True, ax=ax, title=title)

    # node positions = centroids
    pos = {}
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        pos[n] = (c.x, c.y)

    # style maps
    node_style = {
        "living":    dict(color="white",     shape="o", size=400, edgecolor="black"),
        "bedroom":    dict(color="cyan",      shape="s", size=300, edgecolor="black"),
        "bathroom":   dict(color="magenta",   shape="D", size=260, edgecolor="black"),
        "kitchen":    dict(color="yellow",    shape="^", size=300, edgecolor="black"),
        "balcony":    dict(color="lightgray", shape="X", size=260, edgecolor="black"),
        "front_door": dict(color="red",       shape="*", size=420, edgecolor="black"),
    }

    # draw nodes per type for shapes
    nodes_plotted = set()
    min_size, max_size = node_scale
    # area-based scaling
    areas = [G.nodes[n].get("area", 0.0) for n in G.nodes]
    a_min = min(areas) if areas else 0.0
    a_max = max(areas) if areas else 1.0
    def scale_size(a):
        if a_max <= a_min:
            return (min_size + max_size) / 2
        t = (a - a_min) / (a_max - a_min)
        return min_size + t * (max_size - min_size)

    for t, style in node_style.items():
        nlist = [n for n, d in G.nodes(data=True) if d.get("type")==t and n in pos]
        if not nlist:
            continue
        sizes = [scale_size(G.nodes[n].get("area", 0.0)) for n in nlist]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist, node_size=sizes,
            node_shape=style["shape"], node_color=style["color"],
            edgecolors=style["edgecolor"], linewidths=1.0, ax=ax, alpha=0.9
        )
        nodes_plotted.update(nlist)

    # edges by type
    edge_style = {
        "direct":     dict(color="darkred",   width=2.0,  style="-"),
        "adjacency":  dict(color="darkgreen", width=1.5,  style="--"),
        "via_door":   dict(color="darkblue",  width=1.2,  style="-"),
        "via_window": dict(color="orange",    width=1.0,  style=":"),
    }
    for etype, style in edge_style.items():
        elist = [(u,v) for u,v,d in G.edges(data=True) if d.get("type")==etype and u in pos and v in pos]
        if not elist:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=elist,
                               width=style["width"], edge_color=style["color"],
                               style=style["style"], ax=ax, alpha=0.8)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax

# def get_2d_plan(plan: Dict[str, Any],
#                 shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
#                 line_thickness: int = 0,
#                 wall_color: int = 0,
#                 opening_color: int = 140,
#                 bg_color: int = 255,
#                 ax: Optional[plt.Axes] = None,
#                 show: bool = True,
#                 title: Optional[str] = None,
#                 diff: bool=True) -> np.ndarray:
#     """
#     Render a grayscale 2D plan showing walls, windows, doors and front_door.
#     Walls are drawn dark (black by default); doors / windows / front_door are
#     drawn in a lighter gray on top so each element is clearly identifiable
#     in a monochrome image.

#     Returns:
#         np.ndarray of shape (H, W), dtype uint8.
#     """
#     plan = normalize_keys(plan)
#     h, w = shape
#     img = np.full((h, w), bg_color, dtype=np.uint8)

#     # 1) Walls first (dark)
#     wall_geom = plan.get("wall")
#     if wall_geom is not None:
#         mask = geometry_to_mask(wall_geom, shape=shape, line_thickness=line_thickness)
#         img[mask > 0] = wall_color

#     # 2) Openings on top (gray) so they appear as breaks in the wall
#     for key in ["door", "window", "front_door"]:
#         geom = plan.get(key)
#         if geom is None:
#             continue
#         mask = geometry_to_mask(geom, shape=shape, line_thickness=line_thickness)
#         if diff:
#             img[mask > 0] = opening_color
#         else:
#             img[mask > 0] = wall_color

#     img = np.flipud(img)
#     if show:
#         if ax is None:
#             _, ax = plt.subplots(figsize=(6, 6))
#         ax.imshow(img, cmap="gray", vmin=0, vmax=255)
#         ax.set_axis_off()
#         if title:
#             ax.set_title(title)
#         plt.tight_layout()

#     return img

def _oriented_opening(geom, default_thickness: float = 6.0):
    """
    Return (p1, p2, length, thickness) for a door/window geometry.
    p1, p2 = endpoints of the opening CENTERLINE (long axis).
    thickness = short-axis width of the rect (≈ wall thickness).
    """
    if isinstance(geom, LineString):
        coords = np.array(geom.coords, dtype=float)
        p1, p2 = coords[0], coords[-1]
        length = float(np.linalg.norm(p2 - p1))
        return p1, p2, length, float(default_thickness)

    if isinstance(geom, (Polygon, MultiPolygon)):
        rect = geom.minimum_rotated_rectangle
        if not isinstance(rect, Polygon):
            return None
        coords = np.array(rect.exterior.coords[:-1], dtype=float)
        # edge lengths of the rect
        e0 = np.linalg.norm(coords[1] - coords[0])
        e1 = np.linalg.norm(coords[2] - coords[1])
        if max(e0, e1) <= 0:
            return None
        if e0 >= e1:   # long axis along edge 0
            long_vec, length, thickness = coords[1] - coords[0], e0, e1
            a, b, c, d = coords[0], coords[1], coords[3], coords[2]
        else:          # long axis along edge 1
            long_vec, length, thickness = coords[2] - coords[1], e1, e0
            a, b, c, d = coords[1], coords[2], coords[0], coords[3]
        # centerline endpoints = midpoints of the two short edges
        p1 = (a + c) / 2.0
        p2 = (b + d) / 2.0
        return p1, p2, float(length), float(thickness)

    return None


# def _draw_window_symbol(img: np.ndarray,
#                         geom,
#                         wall_color: int = 0,
#                         bg_color: int = 255,
#                         default_thickness: float = 6.0) -> np.ndarray:
#     """
#     Classic architectural window:
#       erase opening → frame rectangle spanning EXACTLY p1→p2 across the full
#       wall thickness (end caps sit at p1/p2 → flush with wall stubs)
#       → one centre line along the opening.
#     """
#     result = _oriented_opening(geom, default_thickness)
#     if result is None:
#         return img
#     p1, p2, length, wall_t = result
#     if length < 2:
#         return img

#     u = (p2 - p1) / length
#     perp = np.array([-u[1], u[0]])
#     half_t = wall_t / 2.0

#     def pt(p):
#         return (int(round(p[0])), int(round(p[1])))

#     # 1) erase opening across full wall thickness
#     quad = np.array([pt(p1 + perp * half_t), pt(p1 - perp * half_t),
#                      pt(p2 - perp * half_t), pt(p2 + perp * half_t)], np.int32)
#     cv2.fillPoly(img, [quad], bg_color)

#     # 2) frame rectangle, corners exactly at the opening ends → connected
#     frame = np.array([pt(p1 + perp * half_t), pt(p2 + perp * half_t),
#                       pt(p2 - perp * half_t), pt(p1 - perp * half_t)], np.int32)
#     cv2.polylines(img, [frame], isClosed=True, color=wall_color, thickness=1)

#     # 3) centre line along the opening
#     cv2.line(img, pt(p1), pt(p2), wall_color, 1)
#     return img


def _draw_window_symbol(img, geom,
                        wall_color=0,
                        window_color=90,
                        bg_color=255,
                        default_thickness=6.0):
    result = _oriented_opening(geom, default_thickness)
    if result is None:
        return img

    p1, p2, length, wall_t = result
    if length < 2:
        return img

    u = (p2 - p1) / length
    perp = np.array([-u[1], u[0]])
    half_t = wall_t / 2.0

    def pt(p):
        return (int(round(p[0])), int(round(p[1])))

    # erase full opening
    quad = np.array([
        pt(p1 + perp * half_t),
        pt(p1 - perp * half_t),
        pt(p2 - perp * half_t),
        pt(p2 + perp * half_t)
    ], np.int32)
    cv2.fillPoly(img, [quad], bg_color)

    # draw softer, centered window lines
    inset = 1.0
    a1 = p1 + perp * (half_t - inset)
    a2 = p2 + perp * (half_t - inset)
    b1 = p1 - perp * (half_t - inset)
    b2 = p2 - perp * (half_t - inset)

    cv2.line(img, pt(a1), pt(a2), window_color, 1)
    cv2.line(img, pt(b1), pt(b2), window_color, 1)
    cv2.line(img, pt(p1), pt(p2), window_color, 1)

    return img

def _draw_door_symbol(img: np.ndarray,
                      geom,
                      wall_color: int = 0,
                      bg_color: int = 255,
                      swing_color: int = 120,
                      default_thickness: float = 6.0) -> np.ndarray:
    """
    Architectural door symbol with automatic swing-side AND hinge-end choice:
      1. BEFORE erasing, sample free space on both perpendicular sides → swing side.
      2. BEFORE erasing, sample the quarter-disc the arc would occupy for BOTH
         hinge ends → pick the hinge whose swing region is most free (prevents
         collisions with walls and other doors).
      3. Erase opening, draw a wall-colored hinge block (wall thickness),
         thin leaf line, smooth quarter arc.
    """
    result = _oriented_opening(geom, default_thickness)
    if result is None:
        return img
    p1, p2, length, wall_t = result
    if length < 3:
        return img

    u = (p2 - p1) / length
    perp = np.array([-u[1], u[0]])
    half_t = wall_t / 2.0
    h_img, w_img = img.shape[:2]

    def pt(p):
        return (int(round(p[0])), int(round(p[1])))

    def sample_free(points):
        free, tot = 0, 0
        for sp in points:
            ix, iy = int(round(sp[0])), int(round(sp[1]))
            if 0 <= ix < w_img and 0 <= iy < h_img:
                tot += 1
                if img[iy, ix] == bg_color:
                    free += 1
        return free / max(tot, 1)

    # ── 1) swing side (sampled BEFORE erase) ──
    def side_points(sign):
        pts = []
        for t in np.linspace(0.1, 0.9, 7):
            base = p1 + u * (length * t)
            for d in np.linspace(half_t + 3, half_t + length, 6):
                pts.append(base + sign * perp * d)
        return pts

    s = +1 if sample_free(side_points(+1)) >= sample_free(side_points(-1)) else -1

    # ── 2) hinge end: test the quarter-disc swept by the arc for both ends ──
    def quarter_free(hinge, other):
        d_open = (other - hinge) / length
        pts = []
        for rr in np.linspace(length * 0.3, length * 0.95, 5):
            for aa in np.linspace(0.08, math.pi / 2 - 0.08, 7):
                dirv = d_open * math.cos(aa) + s * perp * math.sin(aa)
                pts.append(hinge + dirv * rr)
        return sample_free(pts)

    if quarter_free(p1, p2) >= quarter_free(p2, p1):
        hinge, other = p1, p2
    else:
        hinge, other = p2, p1

    # ── 3) erase opening ──
    quad = np.array([pt(p1 + perp * half_t), pt(p1 - perp * half_t),
                     pt(p2 - perp * half_t), pt(p2 + perp * half_t)], np.int32)
    cv2.fillPoly(img, [quad], bg_color)

    # ── 4) hinge block: wall-colored, full wall thickness, flush with stub ──
    hs = max(2, int(round(wall_t / 2)))
    d_open = (other - hinge) / length
    hq = np.array([pt(hinge + perp * half_t),
                   pt(hinge + d_open * hs + perp * half_t),
                   pt(hinge + d_open * hs - perp * half_t),
                   pt(hinge - perp * half_t)], np.int32)
    cv2.fillPoly(img, [hq], wall_color)

    # ── 5) leaf + smooth arc (pivot at inner face of hinge block) ──
    leaf = length - hs
    leaf_base = hinge + d_open * hs
    open_tip = leaf_base + s * perp * leaf
    closed_tip = other

    cv2.line(img, pt(leaf_base), pt(open_tip), swing_color, 1)

    a0 = math.atan2(closed_tip[1] - leaf_base[1], closed_tip[0] - leaf_base[0])
    a1 = math.atan2(open_tip[1] - leaf_base[1], open_tip[0] - leaf_base[0])
    delta = a1 - a0
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi

    steps = max(40, int(leaf * 2))
    arc = np.array([pt(leaf_base + leaf * np.array([math.cos(a0 + delta * t / steps),
                                                    math.sin(a0 + delta * t / steps)]))
                    for t in range(steps + 1)], np.int32)
    cv2.polylines(img, [arc], isClosed=False, color=swing_color, thickness=1)
    return img


def get_2d_plan(plan: Dict[str, Any],
                shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
                line_thickness: int = 0,
                wall_color: int = 0,
                opening_color: int = 140,
                bg_color: int = 255,
                ax: Optional[plt.Axes] = None,
                show: bool = True,
                title: Optional[str] = None,
                diff: bool = True,
                draw_door_swing: bool = True) -> np.ndarray:

    plan = normalize_keys(plan)

    wall_width = float(plan.get("wall_width", 6) or 6)
    pad = int(round(wall_width)) + 4

    h, w = shape
    padded_shape = (h + 2 * pad, w + 2 * pad)

    plan = dict(plan)
    for key in ["wall", "door", "window", "front_door"]:
        if plan.get(key) is not None:
            plan[key] = affinity.translate(plan[key], xoff=pad, yoff=pad)

    img = np.full(padded_shape, bg_color, dtype=np.uint8)

    # 1) walls
    wall_geom = plan.get("wall")
    if wall_geom is not None:
        mask = geometry_to_mask(wall_geom, shape=padded_shape,
                                line_thickness=line_thickness)
        img[mask > 0] = wall_color

    # 2) windows + front door (window-style symbol)
    for key in ["window", "front_door"]:
        geom = plan.get(key)
        if geom is None:
            continue
        for g in get_geometries(geom):
            # img = _draw_window_symbol(img, g,
            #                           wall_color=wall_color,
            #                           bg_color=bg_color,
            #                           default_thickness=wall_width)
            img = _draw_window_symbol(
                    img, g,
                    wall_color=wall_color,
                    window_color=90,
                    bg_color=bg_color,
                    default_thickness=wall_width
                )

    # 3) interior doors — swing-side and hinge-end auto-detected BEFORE erase,
    #    so draw them one at a time on the live image
    door_geom = plan.get("door")
    if door_geom is not None:
        if draw_door_swing:
            for g in get_geometries(door_geom):
                img = _draw_door_symbol(img, g,
                                        wall_color=wall_color,
                                        bg_color=bg_color,
                                        swing_color=opening_color,
                                        default_thickness=wall_width)
        else:
            mask = geometry_to_mask(door_geom, shape=padded_shape,
                                    line_thickness=line_thickness)
            img[mask > 0] = opening_color if diff else wall_color

    img = np.flipud(img)

    if show:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        plt.tight_layout()

    return img

