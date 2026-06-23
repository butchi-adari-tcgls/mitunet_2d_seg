"""
Floorplan mask -> room-connectivity graph  (4-class: 0 bg, 1 wall, 2 door, 3 window)

Rooms are enclosed pockets of free space (everything that is NOT wall/door/
window). Predicted walls are usually leaky, so we SEAL the wall+door+window
barrier (morphological close) before carving rooms, and use 4-connectivity so
diagonal pinholes don't let the interior leak to the exterior.

The exported JSON follows the hierarchical layout-graph schema:
    building (root) -> rooms          (relationship: "contains")
    door  between two rooms           (relationship: "connected_to", door)
    door  to outside                  (relationship: "entry_to",     door)
    window to outside                 (relationship: "opens_to",     window)

NOTE: semantic room types (bedroom/kitchen/...), apartment grouping, lift/
stair/corridor, scale, and sliding/front-door distinctions are NOT derivable
from a 4-class mask, so those fields are left null / generic on purpose.
"""

import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


# ----------------------------------------------------------------------
# LABELS  (matches your test pipeline's COLOR_MAP)
# ----------------------------------------------------------------------
BACKGROUND_ID = 0
WALL_ID       = 1
DOOR_ID       = 2
WINDOW_ID     = 3

LABELS = {0: "background", 1: "wall", 2: "door", 3: "window"}
CONNECTOR_IDS = {DOOR_ID, WINDOW_ID}
BARRIER_IDS   = [WALL_ID, DOOR_ID, WINDOW_ID]
OUTSIDE_ID    = -1                      # internal marker for "outside"

# RGB palette from your test_floorplan_segmentation.py
COLOR_MAP = {
    0: (200, 200, 200),
    1: (40,  40,  40),
    2: (160, 82,  45),
    3: (135, 206, 250),
}


# ======================================================================
# LOADERS
# ======================================================================
def load_label_mask(path):
    """For a mask already storing 0..3 ids (single channel, or grey-RGB)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.int32)


def load_color_mask_exact(path, color_map=COLOR_MAP):
    """Decode the COLOURISED png back to 0..3 ids (exact, lossless)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    label = np.zeros(rgb.shape[:2], np.int32)        # background by default
    for cls, color in color_map.items():
        hit = np.all(rgb == np.array(color, np.uint8), axis=-1)
        label[hit] = cls
    return label


def print_class_histogram(mask, tag="classes"):
    ids, counts = np.unique(mask, return_counts=True)
    total = mask.size
    parts = [f"{LABELS.get(int(i), i)}={c} ({100*c/total:.1f}%)"
             for i, c in zip(ids, counts)]
    print(f"[{tag}] " + "  ".join(parts))


# ======================================================================
# CLEAN
# ======================================================================
def _connect_connectors_to_walls(cleaned, max_gap=20, touch_radius=2,
                                  min_pixels=6, verbose=False):
    """
    Grow every door/window blob along ITS OWN axis until it reaches the wall.

    For each connector component with at least `min_pixels` pixels:
      * find its principal axis (PCA) and its two tips,
      * for each tip not already touching a wall, march outward ALONG THE AXIS
        up to `max_gap` px, scanning a small perpendicular band so a slightly
        off-axis wall is still detected,
      * when a wall is hit, draw a stub of the connector's own thickness from
        the tip to that wall, filling only background pixels with the
        connector's class.
    Tips already touching a wall, tiny specks, and tips with no wall within
    `max_gap` are left untouched. Growth happens only along the axis direction.
    """
    H, W = cleaned.shape
    wall = (cleaned == WALL_ID).astype(np.uint8)
    bridged = 0
    for cid in CONNECTOR_IDS:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(
            (cleaned == cid).astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_pixels:      # need a few real pixels
                continue
            ys, xs = np.where(lbl == i)
            pts = np.column_stack([xs, ys]).astype(np.float64)   # (N,2) as (x,y)
            center = pts.mean(0)

            # principal (long) axis + minor extent (-> stub thickness)
            cov = np.cov((pts - center).T)
            evals, evecs = np.linalg.eigh(cov)
            axis  = evecs[:, int(np.argmax(evals))]
            axis  = axis / (np.linalg.norm(axis) + 1e-9)
            minor = evecs[:, int(np.argmin(evals))]
            proj_major = (pts - center) @ axis
            proj_minor = (pts - center) @ minor
            thick = max(1, int(round(proj_minor.max() - proj_minor.min())))
            half  = thick // 2 + 2                           # perpendicular search half-width
            perp  = np.array([-axis[1], axis[0]])

            tips = [(pts[int(np.argmax(proj_major))], +1.0),
                    (pts[int(np.argmin(proj_major))], -1.0)]
            for tip, sign in tips:
                u = axis * sign
                tx, ty = int(round(tip[0])), int(round(tip[1]))
                y0, y1 = max(0, ty - touch_radius), min(H, ty + touch_radius + 1)
                x0, x1 = max(0, tx - touch_radius), min(W, tx + touch_radius + 1)
                if wall[y0:y1, x0:x1].any():                 # already connected -> skip
                    continue
                hit = None
                for step in range(1, max_gap + 1):
                    base = tip + u * step
                    for t in range(-half, half + 1):         # perpendicular band scan
                        px = int(round(base[0] + perp[0] * t))
                        py = int(round(base[1] + perp[1] * t))
                        if 0 <= px < W and 0 <= py < H and wall[py, px]:
                            hit = (px, py)
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    line = np.zeros((H, W), np.uint8)
                    cv2.line(line, (tx, ty), hit, 1, thickness=thick)
                    cleaned[(line > 0) & (cleaned == BACKGROUND_ID)] = cid
                    bridged += 1
    if verbose:
        print(f"  [bridge] connected {bridged} floating connector end(s) to walls")
    return cleaned


def clean_mask(mask, connector_min_area=2,
               bridge_to_walls=True, bridge_max_gap=20,
               bridge_touch_radius=2, bridge_min_pixels=6,
               verbose=False):
    cleaned = mask.astype(np.int32).copy()

    # =========================================================
    # NEW STEP: Fuse broken door/window pixels (Morphological Close)
    # =========================================================
    # Adjust this kernel size. (5, 5) or (7, 7) usually works well 
    # depending on how far apart the broken door chunks are.
    fuse_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    for cid in CONNECTOR_IDS:
        # Isolate the class (door or window)
        bin_mask = (cleaned == cid).astype(np.uint8)
        
        # Apply closing (dilates to connect broken pieces, erodes to shrink back)
        fused_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, fuse_kernel)
        
        # Apply the fused pixels back to the map. 
        # Crucial: Only overwrite background pixels so we don't accidentally eat into walls.
        cleaned[(fused_mask > 0) & (cleaned == BACKGROUND_ID)] = cid
    # =========================================================

    # 1) drop tiny door/window specks (segmentation noise)
    for cid in CONNECTOR_IDS:
        layer = (cleaned == cid).astype(np.uint8)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(layer, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < connector_min_area:
                cleaned[lbl == i] = BACKGROUND_ID

    # 2) grow each door/window along its axis to connect with the nearby wall
    if bridge_to_walls:
        cleaned = _connect_connectors_to_walls(
            cleaned, max_gap=bridge_max_gap, touch_radius=bridge_touch_radius,
            min_pixels=bridge_min_pixels, verbose=verbose)

    return cleaned


# ======================================================================
# CONTOURS
# ======================================================================
def _bbox4(x, y, w, h):
    return [[int(x), int(y)], [int(x + w), int(y)],
            [int(x + w), int(y + h)], [int(x), int(y + h)]]


def _bbox_contour(blob):
    ys, xs = np.where(blob)
    if xs.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _poly_contour(blob, max_pts=24, min_pts=4):
    cnts, _ = cv2.findContours(blob.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return None
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_pts and eps < 0.05 * peri:
        eps *= 1.3
        nxt = cv2.approxPolyDP(c, eps, True)
        if len(nxt) < min_pts:
            break
        approx = nxt
    return approx.reshape(-1, 2).tolist()


# ======================================================================
# ROOMS  =  sealed free-space components that don't touch the border
# ======================================================================
def extract_rooms(cleaned, min_area=120, contour_mode="poly",
                  seal_ksize=7, verbose=False):
    H, W = cleaned.shape

    barrier = np.isin(cleaned, BARRIER_IDS).astype(np.uint8)
    if seal_ksize and seal_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal_ksize, seal_ksize))
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, k)

    free = (barrier == 0).astype(np.uint8)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(free, connectivity=4)

    outside_cc = set()
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            outside_cc.add(i)
    if not outside_cc and n > 1:
        outside_cc.add(max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA]))

    contour_fn = _bbox_contour if contour_mode == "bbox" else _poly_contour
    nodes = []
    room_label_map = np.full((H, W), -1, np.int32)
    nid = 0
    for i in range(1, n):
        if i in outside_cc:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blob = (lbl == i)
        contour = contour_fn(blob)
        if contour is None or len(contour) < 3:
            continue
        cx, cy = cent[i]
        nodes.append({
            "id": nid,
            "name": f"room_{nid}",
            "area_px": area,
            "centroid_px": [float(cx), float(cy)],
            "bbox_px": _bbox4(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
            "contour": contour,
        })
        room_label_map[blob] = nid
        nid += 1

    outside_mask = np.isin(lbl, list(outside_cc))
    if verbose:
        print(f"  [extract_rooms] seal={seal_ksize} free_components={n-1} "
              f"border/outside={len(outside_cc)} rooms={len(nodes)}")
    return nodes, room_label_map, outside_mask


# ======================================================================
# CONNECTORS
# ======================================================================
def extract_connectors(cleaned, min_area=2):
    conns, cnt = [], 0
    for cls in sorted(CONNECTOR_IDS):
        layer = (cleaned == cls).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(layer, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            blob = (lbl == i)
            contour = _bbox_contour(blob)
            if contour is None:
                continue
            cx, cy = cent[i]
            conns.append({
                "id": cnt, "type": LABELS[cls], "label_id": int(cls),
                "centroid_px": [float(cx), float(cy)],
                "contour": contour, "_mask": blob,
            })
            cnt += 1
    return conns


# ======================================================================
# EDGES (internal representation)
# ======================================================================
def _touching_rooms(conn_mask, room_label_map, outside_mask, tolerance):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * tolerance + 1, 2 * tolerance + 1))
    grown = cv2.dilate(conn_mask.astype(np.uint8), k).astype(bool)
    touches_outside = bool((grown & outside_mask).any())
    region = room_label_map[grown]
    region = region[region >= 0]
    if region.size == 0:
        return [], touches_outside
    ids, counts = np.unique(region, return_counts=True)
    return ids[np.argsort(-counts)].tolist(), touches_outside


def build_edges(room_label_map, outside_mask, connectors, tolerance):
    edges = []
    for eid, c in enumerate(connectors):
        rooms, outside = _touching_rooms(c["_mask"], room_label_map,
                                         outside_mask, tolerance)
        edge = {"id": eid, "type": c["type"], "label_id": c["label_id"],
                "centroid_px": c["centroid_px"], "contour": c["contour"],
                "from": None, "to": None}
        if len(rooms) >= 2:
            edge["from"], edge["to"] = int(rooms[0]), int(rooms[1])
        elif len(rooms) == 1:
            edge["from"] = int(rooms[0])
            edge["to"] = OUTSIDE_ID if outside else None
        edges.append(edge)
    return edges


# ======================================================================
# TOP LEVEL  (auto-escalates the seal until rooms appear)
# ======================================================================
def mask_to_graph(mask, room_min_area=120, connector_min_area=2,
                  match_tolerance=8, contour_mode="poly",
                  seal_ksize="auto", verbose=True):
    cleaned = clean_mask(mask, connector_min_area=connector_min_area,
                         verbose=verbose)
    if verbose:
        print_class_histogram(cleaned)

    # seal_list = [seal_ksize] if seal_ksize != "auto" else [3, 5, 7, 9, 13, 17, 21]
    # nodes = room_label_map = outside_mask = None
    # used_seal = seal_list[-1]
    # for s in seal_list:
    #     nodes, room_label_map, outside_mask = extract_rooms(
    #         cleaned, min_area=room_min_area, contour_mode=contour_mode,
    #         seal_ksize=s, verbose=verbose)
    #     used_seal = s
    #     if len(nodes) > 0:
    #         break

    seal_list = [seal_ksize] if seal_ksize != "auto" else [3, 5, 7, 9, 13, 17, 21]
    best_nodes = []
    best_room_label_map = None
    best_outside_mask = None
    used_seal = seal_list[0]

    for s in seal_list:
        nodes, room_label_map, outside_mask = extract_rooms(
            cleaned, min_area=room_min_area, contour_mode=contour_mode,
            seal_ksize=s, verbose=verbose)
        
        # If this kernel size found MORE rooms, it means it sealed a leak!
        if len(nodes) > len(best_nodes):
            best_nodes = nodes
            best_room_label_map = room_label_map
            best_outside_mask = outside_mask
            used_seal = s

    # Reassign the best results back to your variables
    nodes = best_nodes
    room_label_map = best_room_label_map
    outside_mask = best_outside_mask

    if verbose:
        print(f"[result] seal_ksize={used_seal} -> {len(nodes)} rooms")
        if len(nodes) == 0:
            print("[WARN] still 0 rooms. If a class is ~100% in the histogram the "
                  "decode is wrong; otherwise the walls are too broken to enclose "
                  "anything.")

    eff_tol = max(match_tolerance, used_seal // 2 + 4)
    connectors = extract_connectors(cleaned, min_area=connector_min_area)
    edges = build_edges(room_label_map, outside_mask, connectors, tolerance=eff_tol)
    for c in connectors:
        c.pop("_mask", None)

    return {"nodes": nodes, "edges": edges,
            "cleaned_mask": cleaned, "outside_mask": outside_mask}


def clean_graph(graph, keep_outside=True):
    if keep_outside:
        graph["edges"] = [e for e in graph["edges"] if e["from"] is not None]
    else:
        graph["edges"] = [e for e in graph["edges"]
                          if e["from"] is not None and e["to"] is not None
                          and e["to"] != OUTSIDE_ID]
    return graph


# ======================================================================
# EXPORT  ->  hierarchical layout-graph schema
# ======================================================================
def _round_pt(p):
    return [int(round(p[0])), int(round(p[1]))]


def to_export_schema(graph, image_size, sample_id="sample",
                     scale_px_per_ft=None):
    """
    Convert the internal graph into the hierarchical JSON schema:
    building root contains rooms; doors/windows are connectivity edges.
    """
    H, W = image_size
    rooms = graph["nodes"]
    edges = graph["edges"]
    N = len(rooms)

    BUILDING_ID = 0
    room_eid = {r["id"]: r["id"] + 1 for r in rooms}      # rooms -> 1..N
    needs_exterior = any(e.get("to") == OUTSIDE_ID for e in edges)
    EXTERIOR_ID = (N + 1) if needs_exterior else None

    full_poly = [[0, 0], [W, 0], [W, H], [0, H]]

    # ---- nodes ----
    nodes_out = [{
        "id": BUILDING_ID, "hierarchy_id": "0", "name": "building",
        "label_id": None, "type": "building", "parent_id": None,
        "bbox_px": full_poly, "polygon_px": full_poly,
    }]
    for k, r in enumerate(rooms, start=1):
        nodes_out.append({
            "id": room_eid[r["id"]],
            "hierarchy_id": f"0.{k}",
            "name": r.get("name", f"room_{k}"),
            "label_id": None,                 # room type unknown from 4 classes
            "type": "room",
            "parent_id": BUILDING_ID,
            "area_px": int(r.get("area_px", 0)),
            "centroid_px": _round_pt(r["centroid_px"]),
            "bbox_px": r.get("bbox_px"),
            "polygon_px": r.get("contour"),
        })
    if needs_exterior:
        nodes_out.append({
            "id": EXTERIOR_ID, "hierarchy_id": f"0.{N + 1}", "name": "exterior",
            "label_id": BACKGROUND_ID, "type": "exterior", "parent_id": None,
            "bbox_px": full_poly, "polygon_px": None,
        })

    # ---- edges ----
    edges_out, warnings = [], []
    for r in rooms:                                       # containment
        edges_out.append({"from_id": BUILDING_ID,
                          "to_id": room_eid[r["id"]],
                          "relationship": "contains"})

    for e in edges:                                       # connectivity
        if e["from"] is None:
            warnings.append("connector touched no room (dropped)")
            continue
        ctype = LABELS.get(e["label_id"], "connector")    # 'door' | 'window'
        center = _round_pt(e["centroid_px"])
        if e["to"] == OUTSIDE_ID:
            rel = "entry_to" if e["label_id"] == DOOR_ID else "opens_to"
            efrom, eto = room_eid[e["from"]], EXTERIOR_ID
        elif e["to"] is None:
            continue
        else:
            rel = "connected_to"
            efrom, eto = room_eid[e["from"]], room_eid[e["to"]]
        edges_out.append({
            "from_id": efrom, "to_id": eto,
            "relationship": rel, "connection_type": ctype,
            "center_px": center,
        })

    present = sorted({int(v) for v in np.unique(graph["cleaned_mask"])})

    return {
        "sample_id": sample_id,
        "graph_type": "room_connectivity_graph_4class",
        "apartment_mix": [],                  # not derivable from 4 classes
        "image_size_wh": [int(W), int(H)],
        "scale_px_per_ft": scale_px_per_ft,   # unknown -> null
        "nodes": nodes_out,
        "edges": edges_out,
        "unique_label_ids": present,
        "unique_classes": [LABELS.get(i, str(i)) for i in present],
        "rule_validation": {
            "valid": N > 0,
            "errors": [] if N > 0 else ["no rooms detected"],
            "warnings": sorted(set(warnings)),
        },
        "project_type": "layout_graph",
        "units": [],
        "rooms": [n["id"] for n in nodes_out if n["type"] == "room"],
        "unit_masks": [],
        "access_points": [e for e in edges_out
                          if e.get("relationship") in ("entry_to", "opens_to")],
    }


# ======================================================================
# SAVE
# ======================================================================
def mask_to_rgb(mask, color_map=COLOR_MAP):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for cls, color in color_map.items():
        rgb[mask == cls] = color
    return rgb


def save_cleaned_mask(graph, out_base, save_npy=True, save_png=True):
    cleaned = graph["cleaned_mask"]
    out = {}
    if save_npy:
        p = f"{out_base}_cleaned.npy"
        np.save(p, cleaned.astype(np.uint8)); out["npy"] = p
        print(f"Cleaned labels saved -> {p}")
    if save_png:
        p = f"{out_base}_cleaned.png"
        Image.fromarray(mask_to_rgb(cleaned)).save(p); out["png"] = p
        print(f"Cleaned mask png saved -> {p}")
    return out


def save_graph(graph, out_json_path, image_size, sample_id="sample",
               scale_px_per_ft=None):
    data = to_export_schema(graph, image_size, sample_id, scale_px_per_ft)
    with open(out_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved -> {out_json_path}")
    return out_json_path


# ======================================================================
# PLOT
# ======================================================================
def plot_graph(graph, image_size=None, figsize=(9, 9), show_mask=True, save_path=None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    try:
        cmap = matplotlib.colormaps["tab20"]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20")

    nodes, edges = graph["nodes"], graph["edges"]
    cleaned = graph.get("cleaned_mask")
    if image_size is None and cleaned is not None:
        image_size = cleaned.shape
    H, W = image_size if image_size else (512, 512)

    edge_colors = {DOOR_ID: "#ff7f0e", WINDOW_ID: "#1f78ff"}
    edge_names  = {DOOR_ID: "door", WINDOW_ID: "window"}

    fig, ax = plt.subplots(figsize=figsize)
    if show_mask and cleaned is not None:
        ax.imshow(cleaned, cmap="Greys", alpha=0.18, interpolation="nearest")

    outside_xy = (W * 0.04, H * 0.04)
    node_xy = {OUTSIDE_ID: outside_xy}
    for n in nodes:
        cx, cy = n["centroid_px"]
        node_xy[n["id"]] = (cx, cy)
        col = cmap(n["id"] % 20)
        if n.get("contour") and len(n["contour"]) >= 3:
            ax.add_patch(MplPolygon(n["contour"], closed=True, facecolor=col,
                                    edgecolor="#333", alpha=0.45, linewidth=1.2))
        ax.scatter([cx], [cy], s=240, color=col, edgecolors="black",
                   zorder=5, linewidths=1.3)
        ax.text(cx, cy, str(n["id"]), ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=6)

    for e in edges:
        a, b = node_xy.get(e.get("from")), node_xy.get(e.get("to"))
        col = edge_colors.get(e["label_id"], "#555")
        if a is None:
            continue
        if b is None:
            mx, my = e["centroid_px"]
            ax.scatter([mx], [my], s=50, marker="x", c=col, zorder=7)
            continue
        is_out = (e.get("to") == OUTSIDE_ID)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=2.2, zorder=4,
                solid_capstyle="round", linestyle="--" if is_out else "-", alpha=0.9)
        mx, my = e["centroid_px"]
        ax.scatter([mx], [my], s=55, marker="D", c=col,
                   edgecolors="white", linewidths=1, zorder=7)

    handles = [Line2D([0], [0], color=c, lw=3, label=edge_names[k])
               for k, c in edge_colors.items()]
    ax.legend(handles=handles, title="connectors", loc="upper right", fontsize=8)
    ax.set_xlim(-20, W + 20); ax.set_ylim(H + 20, -20)
    ax.set_aspect("equal")
    ax.set_title(f"Floorplan graph  ({len(nodes)} rooms, {len(edges)} edges)")
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"Plot saved -> {save_path}")
    else:
        plt.show()
    return fig


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    MASK_PATH = "test_outputs/plan_6_mask.png"     # <-- your coloured mask png

    mask = load_color_mask_exact(MASK_PATH)        # decode colours -> 0..3
    # mask = load_label_mask(MASK_PATH)            # use this if you saved labels

    graph = mask_to_graph(
        mask,
        room_min_area=120,
        match_tolerance=8,
        contour_mode="poly",
        seal_ksize="auto",
        verbose=True,
    )
    graph = clean_graph(graph, keep_outside=True)
    print(f"{len(graph['nodes'])} rooms, {len(graph['edges'])} edges")

    stem   = Path(MASK_PATH).stem
    outdir = Path(MASK_PATH).with_suffix("")       # folder named like the file
    outdir.mkdir(parents=True, exist_ok=True)
    base   = str(outdir / stem)

    save_graph(graph, base + "_graph.json", image_size=mask.shape, sample_id=stem)
    save_cleaned_mask(graph, base)
    plot_graph(graph, image_size=mask.shape, show_mask=True,
               save_path=base + "_graph.png")