import numpy as np
from PIL import Image, ImageDraw
import random, os, json
from collections import Counter

# ═══════════════════════════════════════════════════════════════
#  SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════
RULES = {
    "apartment_types": {
        "1HK": {
            "living_space":   1,
            "kitchen":        1,
            "bedroom":        0,
            "master_bedroom": 0,
            "bathroom":       {"min": 1, "max": 1},
            "balcony":        {"min": 0, "max": 1},
            "corridor":       0,
            "lift":           0,
            "front_door_on":  "living_space",
        },
        "1BHK": {
            "living_space":   1,
            "kitchen":        1,
            "bedroom":        1,
            "master_bedroom": 0,
            "bathroom":       {"min": 1, "max": 1},
            "balcony":        {"min": 0, "max": 1},
            "corridor":       0,
            "lift":           0,
            "front_door_on":  "living_space",
        },
        "2BHK": {
            "living_space":                1,
            "kitchen":                     1,
            "bedroom":                     2,
            "master_bedroom":              0,
            "bathroom":                    {"min": 1, "max": 2},
            "balcony":                     {"min": 0, "max": 1},
            "corridor":                    0,
            "lift":                        0,
            "front_door_on":               "living_space",
            "bedroom_sizes_equal_allowed": True,
            "bedroom_one_larger_allowed":  True,
        },
        "3BHK": {
            "living_space":                1,
            "kitchen":                     1,
            "bedroom":                     2,
            "master_bedroom":              1,
            "bathroom":                    {"min": 1, "max": 3},
            "balcony":                     {"min": 0, "max": 2},
            "corridor":                    1,
            "lift":                        0,
            "front_door_on":               "corridor",
            "master_area_gt_each_bedroom": True,
            "master_position_randomised":  True,
            "master_uses_wider_column":    True,
        },
        "CLUSTER": {
            "corridor":             1,
            "lift":                 {"min": 0, "max": 2},
            "stair":                {"min": 0, "max": 1},
            "front_door_on":        "living_space",
            "front_door_faces":     "corridor",
            "supported_unit_types": ["1BHK", "2BHK", "3BHK"],
            "configurations": [
                "CLUSTER_2x1BHK",
                "CLUSTER_2x2BHK",
                "CLUSTER_2x3BHK",
                # "CLUSTER_3x1BHK",
                # "CLUSTER_3x2BHK",
                # "CLUSTER_3x3BHK",
            ],
        },
    },

    "room_types": {
        "living_space": {
            "dimensions_ft": {
                "width":  {"min": 10, "max": 24},
                "height": {"min": 10, "max": 20},
                "area":   {"min": 120, "max": 420},
            },
            "aspect_ratio_max": 2.4,
            "count_per_unit":   {"1HK": 1, "1BHK": 1, "2BHK": 1, "3BHK": 1},
            "area_must_exceed": ["kitchen", "bathroom"],
            "doors": {
                "max_doors":               None,
                "front_door":              True,
                "front_door_wall":         "exterior",
                "front_door_count":        1,
                "allowed_connections":     ["bedroom","master_bedroom","kitchen","bathroom","balcony","corridor"],
                "forbidden_connections":   ["lift"],
                "sliding_door_to_balcony": True,
            },
            "windows": {
                "allowed":                       True,
                "at_flat_boundary":              True,
                "probability":                   1.0,
                "forbidden_on_shared_wall_with": [],
            },
        },

        "bedroom": {
            "dimensions_ft": {
                "width":  {"min": 9,  "max": 20},
                "height": {"min": 10, "max": 18},
                "area":   {"min": 90, "max": 350},
            },
            "aspect_ratio_max": 2.0,
            "count_per_unit":   {"1HK": 0, "1BHK": 1, "2BHK": 2, "3BHK": 2},
            "area_must_exceed": ["bathroom"],
            "doors": {
                "max_doors":               None,
                "front_door":              False,
                "allowed_connections":     ["living_space","bathroom","balcony","corridor","bedroom"],
                "forbidden_connections":   ["kitchen","lift","master_bedroom"],
                "sliding_door_to_balcony": True,
            },
            "windows": {
                "allowed":                       True,
                "at_flat_boundary":              True,
                "probability":                   1.0,
                "forbidden_on_shared_wall_with": ["bathroom"],
            },
        },

        "master_bedroom": {
            "applies_to":    ["3BHK"],
            "dimensions_ft": {
                "width":  {"min": 12, "max": 20},
                "height": {"min": 12, "max": 20},
                "area":   {"min": 144, "max": 360},
            },
            "aspect_ratio_max":      2.0,
            "count_per_unit":        {"3BHK": 1},
            "area_must_exceed":      ["bedroom","bathroom"],
            "area_gt_every_bedroom": True,
            "ensuite": {
                "required":      True,
                "count":         1,
                "room_type":     "bathroom",
                "placement":     "horizontal_touch_below_master",
                "connects_to":   "master_bedroom_only",
                "door_explicit": True,
                "dimensions_ft": {
                    "width":  {"min": 4, "max": 7},
                    "height": {"min": 5, "max": 12},
                    "area":   {"min": 20, "max": 84},
                },
            },
            "doors": {
                "max_doors":               None,
                "front_door":              False,
                "allowed_connections":     ["living_space","balcony","corridor"],
                "forbidden_connections":   ["kitchen","lift","bedroom","bathroom"],
                "sliding_door_to_balcony": True,
            },
            "windows": {
                "allowed":                       True,
                "at_flat_boundary":              True,
                "probability":                   1.0,
                "forbidden_on_shared_wall_with": ["bathroom"],
            },
        },

        "kitchen": {
            "dimensions_ft": {
                "width":  {"min": 6,  "max": 14},
                "height": {"min": 7,  "max": 14},
                "area":   {"min": 42, "max": 160},
            },
            "aspect_ratio_max": 2.2,
            "count_per_unit":   {"1HK": 1, "1BHK": 1, "2BHK": 1, "3BHK": 1},
            "area_must_exceed": ["bathroom"],
            "area_less_than":   ["living_space"],
            "doors": {
                "max_doors":               None,
                "front_door":              False,
                "allowed_connections":     ["living_space","corridor"],
                "forbidden_connections":   ["bathroom","bedroom","master_bedroom","balcony","lift"],
                "sliding_door_to_balcony": False,
            },
            "windows": {
                "allowed":                       True,
                "at_flat_boundary":              True,
                "probability":                   0.6,
                "forbidden_on_shared_wall_with": ["bathroom"],
            },
        },

        "bathroom": {
            "dimensions_ft": {
                "width":  {"min": 4,  "max": 10},
                "height": {"min": 5,  "max": 12},
                "area":   {"min": 20, "max": 100},
            },
            "aspect_ratio_max": 2.5,
            "count_per_unit": {
                "1HK":  {"min": 1, "max": 1},
                "1BHK": {"min": 1, "max": 1},
                "2BHK": {"min": 1, "max": 1},
                "3BHK": {"min": 2, "max": 2},
            },
            "area_less_than": ["living_space","bedroom","master_bedroom","kitchen"],
            "doors": {
                "max_doors":               1,
                "front_door":              False,
                "allowed_connections":     ["living_space","bedroom","master_bedroom"],
                "forbidden_connections":   ["corridor","kitchen","balcony","lift","bathroom"],
                "sliding_door_to_balcony": False,
            },
            "windows": {
                "allowed":                       True,
                "at_flat_boundary":              True,
                "probability":                   0.55,
                "forbidden_on_shared_wall_with": ["bedroom","master_bedroom","kitchen"],
            },
        },

        "balcony": {
            "dimensions_ft": {
                "width":  {"min": 4,  "max": 30},
                "height": {"min": 3,  "max":  7},
                "area":   {"min": 12, "max": 200},
            },
            "aspect_ratio_max": 5.0,
            "count_per_unit": {
                "1HK":  {"min": 0, "max": 1},
                "1BHK": {"min": 0, "max": 1},
                "2BHK": {"min": 0, "max": 1},
                "3BHK": {"min": 0, "max": 2},
            },
            "doors": {
                "max_doors":                None,
                "front_door":               False,
                "allowed_connections":      ["living_space","bedroom","master_bedroom"],
                "forbidden_connections":    ["kitchen","bathroom","corridor","lift"],
                "sliding_door_to_balcony":  False,
                "incoming_door_is_sliding": True,
            },
            "windows": {
                "allowed":                       False,
                "at_flat_boundary":              False,
                "probability":                   0.0,
                "forbidden_on_shared_wall_with": [],
            },
        },

        "corridor": {
            "dimensions_ft": {
                "width":  {"min": 5, "max":  8},
                "height": {"min": 8, "max": 80},
                "area":   None,
            },
            "aspect_ratio_max": None,
            "count_per_unit":   {"1HK": 0, "1BHK": 0, "2BHK": 0, "3BHK": 1, "CLUSTER": 1},
            "doors": {
                "max_doors":               None,
                "front_door":              True,
                "front_door_count":        1,
                "front_door_wall":         "exterior",
                "allowed_connections":     ["living_space","kitchen","bedroom","master_bedroom"],
                "forbidden_connections":   ["bathroom","balcony"],
                "sliding_door_to_balcony": False,
            },
            "windows": {
                "allowed":                       False,
                "at_flat_boundary":              False,
                "probability":                   0.0,
                "forbidden_on_shared_wall_with": [],
            },
        },

        "lift": {
            "dimensions_ft": {
                "width":  {"min": 5, "max":  9},
                "height": {"min": 5, "max": 10},
                "area":   None,
            },
            "aspect_ratio_max":        None,
            "count_per_cluster":       {"min": 1, "max": 2},
            "sits_at_top_of_corridor": True,
            "same_x_as_corridor":      True,
            "doors": {
                "max_doors":               1,
                "front_door":              False,
                "door_wall":               "bottom",
                "door_explicit":           True,
                "allowed_connections":     ["corridor"],
                "forbidden_connections":   ["bedroom","master_bedroom","bathroom","kitchen","living_space","balcony"],
                "sliding_door_to_balcony": False,
            },
            "windows": {
                "allowed":                       False,
                "at_flat_boundary":              False,
                "probability":                   0.0,
                "forbidden_on_shared_wall_with": [],
            },
        },
    },

    "area_hierarchy": [
        ("living_space",   ">", "kitchen"),
        ("living_space",   ">", "bathroom"),
        ("bedroom",        ">", "bathroom"),
        ("master_bedroom", ">", "bathroom"),
        ("master_bedroom", ">", "bedroom"),
        ("kitchen",        ">", "bathroom"),
    ],

    "global_door_rules": {
        "allowed_pairs": [
            ("living_space",   "bedroom"),
            ("living_space",   "master_bedroom"),
            ("living_space",   "kitchen"),
            ("living_space",   "bathroom"),
            ("bedroom",        "bathroom"),
            ("bedroom",        "bedroom"),
            ("corridor",       "living_space"),
            ("corridor",       "kitchen"),
            ("corridor",       "bedroom"),
            ("corridor",       "master_bedroom"),
        ],
        "balcony_pairs": [
            ("living_space",   "balcony"),
            ("bedroom",        "balcony"),
            ("master_bedroom", "balcony"),
        ],
        "explicit_door_pairs": [
            ("corridor",       "lift"),
        ],
        "forbidden_pairs": [
            ("kitchen",        "bathroom"),
            ("kitchen",        "kitchen"),
            ("bathroom",       "bathroom"),
            ("balcony",        "kitchen"),
            ("balcony",        "bathroom"),
            ("corridor",       "bathroom"),
            ("lift",           "bathroom"),
            ("lift",           "bedroom"),
            ("lift",           "master_bedroom"),
            ("lift",           "kitchen"),
            ("lift",           "living_space"),
            ("lift",           "balcony"),
            ("master_bedroom", "bedroom"),
            ("kitchen",        "master_bedroom"),
        ],
        "per_room_door_limits": {
            "bathroom": {"max_doors": 1},
            "lift":     {"max_doors": 1, "allowed_neighbor": "corridor"},
        },
    },

    "global_window_rules": {
        "at_flat_boundary": True,
        "interior_wall":    False,
        "window_probability_by_room": {
            "living_space":   1.0,
            "bedroom":        1.0,
            "master_bedroom": 1.0,
            "kitchen":        0.6,
            "bathroom":       0.55,
        },
        "side_selection": "random_from_available_boundary_sides",
    },
}


# ═══════════════════════════════════════════════════════════════
#  DERIVE ALL WORKING CONSTANTS FROM RULES
# ═══════════════════════════════════════════════════════════════
_STRICT_ROOMS = {"living_space", "bedroom", "master_bedroom", "kitchen"}

def _bhk_rules():
    out = {}
    for apt, d in RULES["apartment_types"].items():
        if apt == "CLUSTER": continue
        out[apt] = {k: v for k, v in d.items()
                    if k in _STRICT_ROOMS and isinstance(v, int)}
    return out

def _max_bathrooms():
    out = {}
    for apt, d in RULES["apartment_types"].items():
        if apt == "CLUSTER": continue
        b = d.get("bathroom", {})
        if isinstance(b, dict): out[apt] = b["max"]
    return out

def _room_size_ranges():
    out = {}
    for name, rd in RULES["room_types"].items():
        dim = rd.get("dimensions_ft", {})
        out[name] = {
            "w":    (dim["width"]["min"],  dim["width"]["max"])  if "width"  in dim else None,
            "h":    (dim["height"]["min"], dim["height"]["max"]) if "height" in dim else None,
            "area": (dim["area"]["min"],   dim["area"]["max"])
                    if isinstance(dim.get("area"), dict) else None,
        }
    return out

def _aspect_limits():
    return {n: rd["aspect_ratio_max"]
            for n, rd in RULES["room_types"].items()
            if rd.get("aspect_ratio_max") is not None}

def _interior_door():
    skip = ({frozenset(p) for p in RULES["global_door_rules"]["explicit_door_pairs"]} |
            {frozenset(p) for p in RULES["global_door_rules"]["balcony_pairs"]})
    return {frozenset(p) for p in RULES["global_door_rules"]["allowed_pairs"]
            if frozenset(p) not in skip}

def _no_door():
    return {frozenset(p) for p in RULES["global_door_rules"]["forbidden_pairs"]}

def _balcony_connects():
    return {frozenset(p) for p in RULES["global_door_rules"]["balcony_pairs"]}

def _win_rooms_prob():
    prob_map = RULES["global_window_rules"]["window_probability_by_room"]
    rooms = {n for n, rd in RULES["room_types"].items()
             if rd.get("windows", {}).get("allowed")}
    prob  = {n: prob_map[n] for n in rooms if n in prob_map}
    return rooms, prob

BHK_RULES        = _bhk_rules()
MAX_BATHROOMS    = _max_bathrooms()
ROOM_SIZE_RANGES = _room_size_ranges()
ASPECT_LIMITS    = _aspect_limits()
INTERIOR_DOOR    = _interior_door()
NO_DOOR          = _no_door()
BALCONY_CONNECTS = _balcony_connects()
WIN_ROOMS, WIN_PROB = _win_rooms_prob()
AREA_HIERARCHY   = RULES["area_hierarchy"]
DOOR_LIMITS      = RULES["global_door_rules"]["per_room_door_limits"]


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
NUM_SAMPLES  = 10000
BASE_DIR     = "plans"
SCALE        = 15
WALL_T       = 7
PAD          = 50
CORR_W_FT    = 7
LIFT_H_FT    = 5
DOOR_W_RANGE = (18, 26)
WIN_W_RANGE  = (20, 28)
# Window openings should occupy most of the available exterior wall,
# leaving the same small wall gap on both ends.
WINDOW_SIDE_GAP_PX = 6
MIN_WINDOW_LEN_PX = 14

APARTMENT_TYPES = [
    "1HK","1BHK","2BHK","3BHK",
    "CLUSTER_2x1BHK",
    "CLUSTER_2x2BHK",
    "CLUSTER_2x3BHK",
    "CLUSTER_3xRANDOM",
    "CLUSTER_4xRANDOM",
    "CLUSTER_4xMIXED",
]

# LABELS = {
#     0:  ("background",     (200, 200, 200)),
#     1:  ("bedroom",        (255, 165,  80)),
#     2:  ("bathroom",       ( 70, 130, 220)),
#     3:  ("kitchen",        (100, 200, 100)),
#     4:  ("wall",           ( 40,  40,  40)),
#     5:  ("door",           (160,  82,  45)),
#     6:  ("window",         (135, 206, 250)),
#     7:  ("front_door",     (255,  50,  50)),
#     8:  ("balcony",        (144, 238, 144)),
#     9:  ("living_space",   (255, 245, 130)),
#     10: ("lift",           (190,   0, 220)),
#     11: ("corridor",       (210, 180, 140)),
#     12: ("apartment",      (220, 170, 255)),
#     13: ("master_bedroom", (255, 140,   0)),
# }

LABELS = {
    0:  ("background",   (200, 200, 200)),
    1:  ("living",       (255, 245, 130)),
    2:  ("bedroom",      (255, 165,  80)),
    3:  ("bathroom",     ( 70, 130, 220)),
    4:  ("kitchen",      (100, 200, 100)),
    5:  ("door",         (160,  82,  45)),
    6:  ("window",       (135, 206, 250)),
    7:  ("wall",         ( 40,  40,  40)),
    8:  ("front_door",   (255,  50,  50)),
    9:  ("balcony",      (144, 238, 144)),
    10: ("lift",         (190,   0, 220)),
    11: ("corridor",     (210, 180, 140)),
    12: ("stair",        (235, 235, 235)),
}


LID = {n: i for i, (n, _) in LABELS.items()}
LCOL = {i: c for i, (_, c) in LABELS.items()}

# aliases
LID["living_space"] = LID["living"]
LID["master_bedroom"] = LID["bedroom"]

# ═══════════════════════════════════════════════════════════════
#  LAYOUT GENERATORS
# ═══════════════════════════════════════════════════════════════
def rv(b, lo=-2, hi=2): return max(4, b + random.randint(lo, hi))
def _lw(): return rv(14, -2, 2)
def _rw(): return rv(7,  -1, 1)
def _lh(): return rv(13, -1, 1)
def _bh(): return rv(11, -1, 1)

def _split_kb(h):
    b = min(8, max(5, h // 3))
    return h - b, b

def _mirror_rooms_lr(rooms, W):
    return [(n, W - x - w, y, w, h) for (n, x, y, w, h) in rooms]

def _mirror_rooms_tb(rooms, H):
    return [(n, x, H - y - h, w, h) for (n, x, y, w, h) in rooms]

def _transform_layout(rooms, W, H, allow_vflip=True):
    if random.random() < 0.5:
        rooms = _mirror_rooms_lr(rooms, W)
    if allow_vflip and random.random() < 0.35:
        rooms = _mirror_rooms_tb(rooms, H)
    return rooms, W, H

def _make_flat_side(n_bed, n_bath, bhs=None):
    lw, rw, lh = _lw(), _rw(), _lh()
    W = lw + rw
    if n_bed == 0:
        kh, bth = _split_kb(lh)
        rooms = [
            ("living_space", 0,  0,  lw, lh),
            ("kitchen",      lw, 0,  rw, kh),
            ("bathroom",     lw, kh, rw, bth),
            ("balcony",      0,  lh, W,  4),
        ]
        return _transform_layout(rooms, W, lh + 4)
    if bhs is None:
        bhs = [_bh() for _ in range(n_bed)]
    rooms = [("living_space", 0, 0, lw, lh), ("kitchen", lw, 0, rw, lh)]
    y = lh
    for i, bh in enumerate(bhs):
        rooms.append(("bedroom", 0, y, lw, bh))
        if i < n_bath:
            rooms.append(("bathroom", lw, y, rw, bh))
        y += bh
    rooms.append(("balcony", 0, y, W, 4))
    return _transform_layout(rooms, W, y + 4)

def _make_1hk_alt():
    kw = rv(11, -1, 2)
    bw = rv(5, 0, 1)
    top_h = rv(7, 0, 1)
    lh = rv(13, -1, 2)
    W = kw + bw
    rooms = [
        ("kitchen",      0,   0, kw, top_h),
        ("bathroom",     kw,  0, bw, top_h),
        ("living_space", 0, top_h, W,  lh),
        ("balcony",      0, top_h + lh, W, 4),
    ]
    return _transform_layout(rooms, W, top_h + lh + 4)

def _make_1bhk_alt():
    left_w = rv(13, -1, 2)
    serv_w = rv(7, -1, 1)
    living_h = rv(12, -1, 1)
    k_h = rv(7, -1, 1)
    b_h = rv(6, -1, 1)
    bed_h = k_h + b_h
    W = left_w + serv_w
    rooms = [
        ("living_space", 0, 0, W, living_h),
        ("bedroom",      0, living_h, left_w, bed_h),
        ("kitchen",      left_w, living_h, serv_w, k_h),
        ("bathroom",     left_w, living_h + k_h, serv_w, b_h),
        ("balcony",      0, living_h + bed_h, left_w, 4),
    ]
    return _transform_layout(rooms, W, living_h + bed_h + 4)

def _make_2bhk_alt():
    left_w = rv(13, -1, 2)
    right_w = rv(10, -1, 1)
    top_h = rv(12, -1, 1)
    bed2_h = rv(10, 0, 1)
    bath_h = rv(6, -1, 1)
    bed1_h = bed2_h + bath_h
    W = left_w + right_w
    rooms = [
        ("living_space", 0, 0, left_w, top_h),
        ("kitchen",      left_w, 0, right_w, top_h),
        ("bedroom",      0, top_h, left_w, bed1_h),
        ("bedroom",      left_w, top_h, right_w, bed2_h),
        ("bathroom",     left_w, top_h + bed2_h, right_w, bath_h),
        ("balcony",      0, top_h + bed1_h, left_w, 4),
    ]
    return _transform_layout(rooms, W, top_h + bed1_h + 4)

def _make_flat(n_bed, n_bath, bhs=None):
    if n_bed == 0:
        maker = random.choice([_make_flat_side, _make_1hk_alt])
        if maker is _make_flat_side:
            return maker(n_bed, n_bath, bhs)
        return maker()
    if n_bed == 1:
        maker = random.choice([_make_flat_side, _make_1bhk_alt])
        if maker is _make_flat_side:
            return maker(n_bed, n_bath, bhs)
        return maker()
    if n_bed == 2:
        maker = random.choice([_make_flat_side, _make_2bhk_alt])
        if maker is _make_flat_side:
            return maker(n_bed, n_bath, bhs)
        return maker()
    return _make_flat_side(n_bed, n_bath, bhs)

def _make_3bhk_grid_base(variant=0):
    cw = CORR_W_FT
    lw = rv(15, -1, 1)
    rw = rv(10, -1, 1)
    top_h = rv(12, -1, 1)
    bed_h1 = 10
    bed_h2 = 10
    bed_total_h = bed_h1 + bed_h2
    master_h = bed_total_h
    bal_h = 4
    en_w, en_h = 6, 7
    bath_w, bath_h = 4, 5
    W = lw + cw + rw
    H = top_h + bed_total_h + bal_h
    right_x = lw + cw

    if variant == 0:
        rooms = [
            ("living_space",   0,       0,              lw, top_h),
            ("kitchen",        right_x, 0,              rw, top_h),
            ("corridor",       lw,      0,              cw, H - bal_h),
            ("master_bedroom", 0,       top_h,          lw, master_h),
            ("bathroom",       0,       top_h + master_h - en_h, en_w, en_h),
            ("bedroom",        right_x, top_h,          rw, bed_h1),
            ("bedroom",        right_x, top_h + bed_h1, rw, bed_h2),
            ("bathroom",       right_x + rw - bath_w, top_h + bed_total_h - bath_h, bath_w, bath_h),
            ("balcony",        0,       H - bal_h,      lw, bal_h),
            ("balcony",        right_x, H - bal_h,      rw, bal_h),
        ]
    else:
        rooms = [
            ("kitchen",        0,       0,              rw, top_h),
            ("living_space",   rw + cw, 0,              lw, top_h),
            ("corridor",       rw,      0,              cw, H - bal_h),
            ("bedroom",        0,       top_h,          rw, bed_h1),
            ("bedroom",        0,       top_h + bed_h1, rw, bed_h2),
            ("bathroom",       0,       top_h + bed_total_h - bath_h, bath_w, bath_h),
            ("master_bedroom", rw + cw, top_h,          lw, master_h),
            ("bathroom",       rw + cw + lw - en_w, top_h + master_h - en_h, en_w, en_h),
            ("balcony",        0,       H - bal_h,      rw, bal_h),
            ("balcony",        rw + cw, H - bal_h,      lw, bal_h),
        ]
    return rooms, W, H

def _make_3bhk_grid():
    rooms, W, H = _make_3bhk_grid_base(random.choice([0, 1]))
    return _transform_layout(rooms, W, H, allow_vflip=False)

def tmpl_1hk():  return _make_flat(0, 1)
def tmpl_1bhk(): return _make_flat(1, 1)
def tmpl_2bhk():
    bhs = [rv(11, -1, 1), max(10, min(15, rv(11, -3, 4)))]
    return _make_flat(2, 1, bhs)
def tmpl_3bhk(): return _make_3bhk_grid()

FLAT_TEMPLATES = {
    "1HK": tmpl_1hk, "1BHK": tmpl_1bhk,
    "2BHK": tmpl_2bhk, "3BHK": tmpl_3bhk,
}
MAX_ATTEMPTS = 150


# ═══════════════════════════════════════════════════════════════
#  VALIDATION  (uses RULES-derived constants)
# ═══════════════════════════════════════════════════════════════
def _asp(w, h): return max(w/h, h/w) if w and h else 99

def validate(apt_type, rooms_ft, tw=None, th=None, strict=True):
    err = []
    counts = Counter(r[0] for r in rooms_ft)
    for name, x, y, w, h in rooms_ft:
        if name not in LID: err.append(f"unknown:{name}"); continue
        if not strict and name == "corridor":
            if tw and (x < 0 or x+w > tw+0.01): err.append(f"{name} x OOB")
            if th and (y < 0 or y+h > th+0.01): err.append(f"{name} y OOB")
            continue

        # In cluster layouts, a 3BHK's internal corridor is intentionally renamed
        # to living_space so it becomes part of the hall. That merged hall strip is
        # narrow/long by design, so do not validate it using normal living-room
        # width/height/aspect rules. Still check that it stays inside bounds.
        is_merged_hall_strip = (
            not strict and
            name == "living_space" and
            (w <= CORR_W_FT + 1 or h <= CORR_W_FT + 1)
        )
        if is_merged_hall_strip:
            if tw and (x < 0 or x+w > tw+0.01): err.append(f"{name} x OOB")
            if th and (y < 0 or y+h > th+0.01): err.append(f"{name} y OOB")
            continue

        sz = ROOM_SIZE_RANGES.get(name, {})
        if sz.get("w"):
            mw, xw = sz["w"]
            if not mw <= w <= xw: err.append(f"{name} w={w:.1f}∉[{mw},{xw}]")
        if sz.get("h"):
            mh, xh = sz["h"]
            if not mh <= h <= xh: err.append(f"{name} h={h:.1f}∉[{mh},{xh}]")
        if sz.get("area"):
            ma, xa = sz["area"]
            if not ma <= w*h <= xa: err.append(f"{name} area={w*h:.0f}∉[{ma},{xa}]")
        lim = ASPECT_LIMITS.get(name)
        if lim and _asp(w, h) > lim: err.append(f"{name} asp={_asp(w,h):.2f}>{lim}")
        if tw and (x < 0 or x+w > tw+0.01): err.append(f"{name} x OOB")
        if th and (y < 0 or y+h > th+0.01): err.append(f"{name} y OOB")
    base = apt_type.split("x",1)[-1] if apt_type.startswith("CLUSTER_") else apt_type
    if strict:
        for rn, exp in (BHK_RULES.get(base) or {}).items():
            if counts.get(rn, 0) != exp:
                err.append(f"{base}:{rn} need {exp} got {counts.get(rn,0)}")
        mb = MAX_BATHROOMS.get(base)
        if mb and counts.get("bathroom", 0) > mb:
            err.append(f"{base}: max {mb} bath got {counts.get('bathroom',0)}")
    if strict:
        for big, _, sml in AREA_HIERARCHY:
            ba = [w*h for n,_,_,w,h in rooms_ft if n==big]
            sa = [w*h for n,_,_,w,h in rooms_ft if n==sml]
            if ba and sa and min(ba) <= max(sa): err.append(f"{big} not > {sml}")
    return {"valid": not err, "errors": err, "warnings": []}


def _merge_internal_corridor_into_living(rooms_ft):
    """Convert corridors that belong to an individual flat into hall/living space.

    The cluster-level/building corridor is added later by the cluster placement
    functions, so any corridor present inside a generated flat template is an
    internal passage. Treating it as living_space removes the extra corridor
    label and lets the front door connect directly to the hall.
    """
    return [
        ("living_space" if n == "corridor" else n, x, y, w, h)
        for (n, x, y, w, h) in rooms_ft
    ]

def _make_flat_side_balcony(n_bed, n_bath, side="right", bhs=None):
    """Generate a flat with the balcony as a 4ft-wide vertical strip on the exterior wall.
    Rooms keep their normal sizes; balcony is an extra column that doesn't shrink anything.

    side="right": corridor on left  → living touches x=0 (left=corridor), balcony on right
    side="left":  corridor on right → living touches x=W (right=corridor), balcony on left
    """
    BAL = 4
    lw, rw, lh = _lw(), _rw(), _lh()
    if bhs is None:
        bhs = [_bh() for _ in range(n_bed)]
    H = lh + sum(bhs)
    bal_h = min(7, lh)   # balcony alongside living_space height, capped at 7ft

    if side == "right":
        # Rooms in normal positions; living at x=0 faces corridor (left)
        if n_bed == 0:
            kh, bth = _split_kb(lh)
            rooms = [("living_space", 0, 0, lw, lh),
                     ("kitchen",      lw, 0, rw, kh),
                     ("bathroom",     lw, kh, rw, bth)]
        else:
            rooms = [("living_space", 0, 0, lw, lh), ("kitchen", lw, 0, rw, lh)]
            y = lh
            for i, bh in enumerate(bhs):
                rooms.append(("bedroom", 0, y, lw, bh))
                if i < n_bath:
                    rooms.append(("bathroom", lw, y, rw, bh))
                y += bh
        W_rooms = lw + rw
        rooms.append(("balcony", W_rooms, 0, BAL, bal_h))   # right exterior
        W = W_rooms + BAL

    else:  # side == "left": balcony on left, rooms shifted right, living faces right (corridor)
        if n_bed == 0:
            kh, bth = _split_kb(lh)
            rooms = [("living_space", BAL + rw, 0, lw, lh),
                     ("kitchen",      BAL,      0, rw, kh),
                     ("bathroom",     BAL,      kh, rw, bth)]
        else:
            rooms = [("living_space", BAL + rw, 0, lw, lh),
                     ("kitchen",      BAL,      0, rw, lh)]
            y = lh
            for i, bh in enumerate(bhs):
                rooms.append(("bedroom",  BAL + rw, y, lw, bh))
                if i < n_bath:
                    rooms.append(("bathroom", BAL, y, rw, bh))
                y += bh
        W_rooms = lw + rw
        rooms.append(("balcony", 0, 0, BAL, bal_h))          # left exterior
        W = W_rooms + BAL

    return rooms, W, H


def _gen_flat_side_balcony(btype, side="right"):
    """gen_flat equivalent that produces a flat with a side-facing balcony."""
    if btype == "3BHK":
        # Keep the correct 3BHK composition: 2 bedrooms + 1 master_bedroom.
        # Do not force a left/right balcony by shrinking rooms; that can make
        # master_bedroom narrower than its rule minimum. 3BHK already has
        # top/bottom balconies when those sides are exterior. If only a side is
        # exterior, balcony is optional, so safely remove it.
        r, tw, th = tmpl_3bhk()
        r = _merge_internal_corridor_into_living(r)
        if side in ("left", "right"):
            r = _remove_balconies(r)
        else:
            r = _orient_balcony(r, tw, th, side)
        return r, tw, th

    n_bed  = {"1HK": 0, "1BHK": 1, "2BHK": 2}.get(btype, 1)
    n_bath = {"1HK": 1, "1BHK": 1, "2BHK": 1}.get(btype, 1)
    best = None
    for _ in range(MAX_ATTEMPTS):
        r, tw, th = _make_flat_side_balcony(n_bed, n_bath, side=side)
        v = validate(btype, r, tw, th)
        if v["valid"]:
            return r, tw, th
        if best is None:
            best = (r, tw, th)
    return best


def gen_flat(btype):
    best = None
    for _ in range(MAX_ATTEMPTS):
        r, tw, th = FLAT_TEMPLATES[btype]()
        v = validate(btype, r, tw, th)
        if v["valid"]:
            return _merge_internal_corridor_into_living(r), tw, th, v
        best = (r, tw, th, v)
    r, tw, th, v = best
    raise ValueError(f"{btype}: {'; '.join(v['errors'][:3])}")

def _exterior_sides_of_flat(x0, y0, w, h, total_w, total_h, tol=0.5):
    """Return which sides of a flat touch the building exterior boundary."""
    sides = []
    if x0 <= tol:                    sides.append("left")
    if x0 + w >= total_w - tol:      sides.append("right")
    if y0 <= tol:                    sides.append("top")
    if y0 + h >= total_h - tol:      sides.append("bottom")
    return sides


def _pick_balcony_side(ext_sides):
    """Pick an actual exterior side for a balcony.

    Returns None when the flat has no confirmed building-exterior side.
    This prevents balconies from being forced between two flats.
    """
    for s in ("bottom", "top", "right", "left"):
        if s in ext_sides:
            return s
    return None


def _orient_balcony(rooms, W, H, side, depth=4):
    """
    Orient the balcony to face `side` (the exterior wall).

    - top / bottom: use mirrors — the horizontal strip already fits.
    - right / left:  the horizontal strip can't face these walls via mirroring alone,
                     so physically replace it with a vertical strip and shrink the
                     rooms that previously occupied that edge by `depth` feet.
    """
    if side in ("top", "bottom"):
        return _force_balcony_side(rooms, W, H, side)

    # --- side balcony (right or left) ---
    # Separate existing balcony from main rooms.
    main = [(n, x, y, w, h) for (n, x, y, w, h) in rooms if n != "balcony"]
    # H_main = height of the main room area (total H minus the horizontal strip at top/bottom)
    H_main = H - depth

    result = []
    if side == "right":
        for (n, x, y, rw, rh) in main:
            if abs((x + rw) - W) < 0.5:          # room touches the right wall
                result.append((n, x, y, max(4, rw - depth), rh))
            else:
                result.append((n, x, y, rw, rh))
        result.append(("balcony", W - depth, 0, depth, min(7, H_main)))
    else:  # left
        for (n, x, y, rw, rh) in main:
            if abs(x) < 0.5:                       # room touches the left wall
                result.append((n, depth, y, max(4, rw - depth), rh))
            else:
                result.append((n, x, y, rw, rh))
        result.append(("balcony", 0, 0, depth, min(7, H_main)))

    return result


def _remove_balconies(rooms):
    return [(n, x, y, w, h) for (n, x, y, w, h) in rooms if n != "balcony"]


def _balcony_has_building_exterior(room, x0, y0, total_w, total_h, tol=0.5):
    n, x, y, w, h = room
    if n != "balcony":
        return True
    gx1, gy1 = x0 + x, y0 + y
    gx2, gy2 = gx1 + w, gy1 + h
    return (
        gx1 <= tol or
        gy1 <= tol or
        gx2 >= total_w - tol or
        gy2 >= total_h - tol
    )


def _keep_only_exterior_balconies(rooms, x0, y0, total_w, total_h):
    """Drop any balcony that is not on the building exterior boundary."""
    return [
        r for r in rooms
        if r[0] != "balcony" or _balcony_has_building_exterior(r, x0, y0, total_w, total_h)
    ]


def _force_balcony_side(rooms, W, H, side):
    def _balcony_touches(rs, s):
        for n, x, y, w, h in rs:
            if n != "balcony":
                continue
            if s == "top"    and abs(y) < 1e-6: return True
            if s == "bottom" and abs((y + h) - H) < 1e-6: return True
            if s == "left"   and abs(x) < 1e-6: return True
            if s == "right"  and abs((x + w) - W) < 1e-6: return True
        return False

    opts = [
        rooms,
        _mirror_rooms_lr(rooms, W),
        _mirror_rooms_tb(rooms, H),
        _mirror_rooms_tb(_mirror_rooms_lr(rooms, W), H),
    ]
    for rs in opts:
        if _balcony_touches(rs, s=side):
            return rs
    return rooms


def _force_living_side(rooms, W, H, side):
    def _living_touches(rs, s):
        for n,x,y,w,h in rs:
            if n != "living_space":
                continue
            if s == "left"   and abs(x) < 1e-6: return True
            if s == "right"  and abs((x+w) - W) < 1e-6: return True
            if s == "top"    and abs(y) < 1e-6: return True
            if s == "bottom" and abs((y+h) - H) < 1e-6: return True
        return False

    opts = [
        rooms,
        _mirror_rooms_lr(rooms, W),
        _mirror_rooms_tb(rooms, H),
        _mirror_rooms_tb(_mirror_rooms_lr(rooms, W), H),
    ]
    for rs in opts:
        if _living_touches(rs, side):
            return rs
    return rooms


def _choose_service_spec(orientation, corridor_side=None):
    """Randomize lift/stair service core attached to the corridor outer edge.

    Cases produced:
      - lift only
      - stair only
      - lift + stair

    Stairs use at least 3 visual styles at render time; here we only place the
    bounding box. When both lift and stair are present, they are placed on
    opposite ends/sides when possible so the shared corridor reads clearly.
    """
    mode = random.choice(["lift_only", "stair_only", "lift_and_stair"])

    lift_count = 0
    if mode in ("lift_only", "lift_and_stair"):
        lift_count = random.choice([1, 2])

    stair_count = 1 if mode in ("stair_only", "lift_and_stair") else 0

    lift_w = 7
    lift_d = LIFT_H_FT

    stair_style = random.choice(["straight", "l_shaped", "u_shaped"])
    if stair_style == "straight":
        stair_w = 10
        stair_d = max(CORR_W_FT, 8)
    elif stair_style == "l_shaped":
        stair_w = 12
        stair_d = max(CORR_W_FT, 9)
    else:
        stair_w = 14
        stair_d = max(CORR_W_FT, 10)

    # For horizontal corridors, service blocks must always sit on the OUTER wall.
    if corridor_side in ("top", "bottom"):
        outer_side = "bottom" if corridor_side == "top" else "top"
    else:
        outer_side = random.choice(["top", "bottom"])

    lift_side = outer_side if lift_count else None
    stair_side = outer_side if stair_count else None

    lift_anchor = "center"
    stair_anchor = "center"
    if mode == "lift_and_stair":
        if orientation == "vertical":
            # Put them on opposite ends of the corridor when possible.
            stair_side = "bottom" if lift_side == "top" else "top"
        else:
            # Same outer wall, opposite horizontal ends.
            if random.random() < 0.5:
                lift_anchor, stair_anchor = "left", "right"
            else:
                lift_anchor, stair_anchor = "right", "left"

    corr_span = CORR_W_FT
    if orientation == "vertical":
        need_w = 0
        if lift_count:
            need_w = max(need_w, lift_count * lift_w)
        if stair_count:
            need_w = max(need_w, stair_w)
        corr_span = max(CORR_W_FT, need_w)
    else:
        need_h = 0
        if lift_count:
            need_h = max(need_h, lift_d)
        if stair_count:
            need_h = max(need_h, stair_d)
        corr_span = max(CORR_W_FT, need_h)

    return {
        "mode": mode,
        "orientation": orientation,
        "corr_span": corr_span,
        "lift_spec": None if lift_count == 0 else {
            "count": lift_count,
            "side": lift_side,
            "lift_w": lift_w,
            "lift_d": lift_d,
            "anchor": lift_anchor,
        },
        "stair_spec": None if stair_count == 0 else {
            "count": stair_count,
            "side": stair_side,
            "stair_w": stair_w,
            "stair_d": stair_d,
            "anchor": stair_anchor,
            "style": stair_style,
        },
    }


def _anchored_x(corr_x, corr_w, block_w, anchor):
    if anchor == "left":
        return corr_x
    if anchor == "right":
        return corr_x + max(0, corr_w - block_w)
    return corr_x + max(0, (corr_w - block_w) / 2)


def _build_vertical_lifts(corr_x, corr_y, corr_w, corr_h, lift_spec):
    count = int(lift_spec.get("count", 1))
    lift_d = min(LIFT_H_FT, max(5, lift_spec.get("lift_d", LIFT_H_FT)))
    lift_w = max(5, min(9, lift_spec.get("lift_w", 7)))
    total_lift_w = count * lift_w
    x0 = _anchored_x(corr_x, corr_w, total_lift_w, lift_spec.get("anchor", "center"))
    y0 = corr_y if lift_spec.get("side") == "top" else corr_y + max(0, corr_h - lift_d)
    return [("lift", x0 + i * lift_w, y0, lift_w, lift_d) for i in range(count)]


def _build_horizontal_lifts(corr_x, corr_y, corr_w, corr_h, lift_spec):
    count = int(lift_spec.get("count", 1))
    lift_d = min(LIFT_H_FT, max(5, lift_spec.get("lift_d", LIFT_H_FT)))
    lift_w = max(5, min(9, lift_spec.get("lift_w", 7)))
    total_lift_w = count * lift_w
    x0 = _anchored_x(corr_x, corr_w, total_lift_w, lift_spec.get("anchor", "center"))
    y0 = corr_y if lift_spec.get("side") == "top" else corr_y + max(0, corr_h - lift_d)
    return [("lift", x0 + i * lift_w, y0, lift_w, lift_d) for i in range(count)]


def _build_vertical_stairs(corr_x, corr_y, corr_w, corr_h, stair_spec):
    stair_w = max(8, stair_spec.get("stair_w", 10))
    stair_d = max(7, stair_spec.get("stair_d", CORR_W_FT))
    x0 = _anchored_x(corr_x, corr_w, stair_w, stair_spec.get("anchor", "center"))
    y0 = corr_y if stair_spec.get("side") == "top" else corr_y + max(0, corr_h - stair_d)
    return [("stair", x0, y0, stair_w, stair_d)]


def _build_horizontal_stairs(corr_x, corr_y, corr_w, corr_h, stair_spec):
    stair_w = max(8, stair_spec.get("stair_w", 10))
    stair_d = max(7, stair_spec.get("stair_d", CORR_W_FT))
    x0 = _anchored_x(corr_x, corr_w, stair_w, stair_spec.get("anchor", "center"))
    y0 = corr_y if stair_spec.get("side") == "top" else corr_y + max(0, corr_h - stair_d)
    return [("stair", x0, y0, stair_w, stair_d)]


def _place_vertical_corridor(unit_defs, left_ids, right_ids):
    left_raw  = [unit_defs[i] for i in left_ids]
    right_raw = [unit_defs[i] for i in right_ids]
    service_spec = _choose_service_spec("vertical")
    corr_w = service_spec["corr_span"]

    def _tentative_positions(raw_defs, x_offset_fn, col_w):
        positions, y = [], 0
        for (_, _, w, h) in raw_defs:
            positions.append((x_offset_fn(w, col_w), y))
            y += h
        return positions

    left_w_tent  = max([w for (_, _, w, _) in left_raw],  default=0)
    right_w_tent = max([w for (_, _, w, _) in right_raw], default=0)
    left_h_sum   = sum(h for (_, _, _, h) in left_raw)
    right_h_sum  = sum(h for (_, _, _, h) in right_raw)
    total_h      = max(left_h_sum, right_h_sum,
                       max([h for (*_, h) in left_raw]  + [0]),
                       max([h for (*_, h) in right_raw] + [0]))
    total_w_tent = left_w_tent + corr_w + right_w_tent

    left_pos_tent  = _tentative_positions(left_raw,  lambda w, cw: cw - w, left_w_tent)
    right_pos_tent = _tentative_positions(right_raw, lambda w, cw: left_w_tent + corr_w, right_w_tent)

    def _resolve_column(raw_defs, positions, col_side, living_side):
        out = []
        for (btype, rooms, w, h), (x0, y0) in zip(raw_defs, positions):
            ext = _exterior_sides_of_flat(x0, y0, w, h, total_w_tent, total_h)
            best = _pick_balcony_side(ext)
            if best is None:
                rooms = _remove_balconies(rooms)
            elif best in ("left", "right"):
                result = _gen_flat_side_balcony(btype, side=best)
                rooms, w, h = result
            rooms = _force_living_side(rooms, w, h, living_side)
            out.append((rooms, btype, w, h, best))
        return out

    left_defs  = _resolve_column(left_raw,  left_pos_tent,  "left",  "right")
    right_defs = _resolve_column(right_raw, right_pos_tent, "right", "left")

    left_w   = max([w for (_, _, w, _, _) in left_defs],  default=0)
    right_w  = max([w for (_, _, w, _, _) in right_defs], default=0)
    left_h_final  = sum(h for (_, _, _, h, _) in left_defs)
    right_h_final = sum(h for (_, _, _, h, _) in right_defs)
    total_h  = max(left_h_final, right_h_final,
                   max([h for (*_, h, _) in left_defs]  + [0]),
                   max([h for (*_, h, _) in right_defs] + [0]))
    total_w  = left_w + corr_w + right_w

    left_positions, right_positions = [], []
    y = 0
    for (_, _, w, h, _) in left_defs:
        left_positions.append((left_w - w, y)); y += h
    x_base = left_w + corr_w; y = 0
    for (_, _, w, h, _) in right_defs:
        right_positions.append((x_base, y)); y += h

    def _blit_column(col_defs, col_positions):
        for (rooms, btype, w, h, bal_side), (x0, y0) in zip(col_defs, col_positions):
            if bal_side in ("top", "bottom"):
                rooms = _force_balcony_side(rooms, w, h, bal_side)
            rooms = _keep_only_exterior_balconies(rooms, x0, y0, total_w, total_h)
            all_rooms.extend([(nm, x0+x, y0+y, rw, rh) for (nm, x, y, rw, rh) in rooms])
            flat_bounds.append((x0, y0, w, h))

    all_rooms, flat_bounds = [], []
    _blit_column(left_defs,  left_positions)
    _blit_column(right_defs, right_positions)

    all_rooms.append(("corridor", left_w, 0, corr_w, total_h))
    if service_spec.get("lift_spec"):
        all_rooms.extend(_build_vertical_lifts(left_w, 0, corr_w, total_h, service_spec["lift_spec"]))
    if service_spec.get("stair_spec"):
        all_rooms.extend(_build_vertical_stairs(left_w, 0, corr_w, total_h, service_spec["stair_spec"]))
    return all_rooms, total_w, total_h, flat_bounds, {"orientation": "vertical", "service_spec": service_spec}


def _place_one_side_vertical(unit_defs, side="right"):
    target_side = "left" if side == "right" else "right"
    service_spec = _choose_service_spec("vertical")
    corr_w = service_spec["corr_span"]
    col_w_tent = max(w for (_, _, w, _) in unit_defs)
    total_h    = sum(h for (_, _, _, h) in unit_defs)
    total_w_tent = corr_w + col_w_tent
    x_unit = corr_w if side == "right" else 0

    tent_pos, y = [], 0
    for (_, _, w, h) in unit_defs:
        x0 = x_unit if side == "right" else col_w_tent - w
        tent_pos.append((x0, y)); y += h

    defs = []
    for (btype, rooms, w, h), (x0, y0) in zip(unit_defs, tent_pos):
        ext  = _exterior_sides_of_flat(x0, y0, w, h, total_w_tent, total_h)
        best = _pick_balcony_side(ext)
        if best is None:
            rooms = _remove_balconies(rooms)
        elif best in ("left", "right"):
            result = _gen_flat_side_balcony(btype, side=best)
            rooms, w, h = result
        rooms = _force_living_side(rooms, w, h, target_side)
        defs.append((rooms, btype, w, h, best))

    col_w   = max(w for (_, _, w, _, _) in defs)
    total_h = sum(h for (_, _, _, h, _) in defs)
    total_w = corr_w + col_w
    corr_x  = 0 if side == "right" else col_w

    positions, y = [], 0
    for (_, _, w, h, _) in defs:
        x0 = x_unit if side == "right" else col_w - w
        positions.append((x0, y)); y += h

    all_rooms, flat_bounds = [], []
    for (rooms, btype, w, h, bal_side), (x0, y0) in zip(defs, positions):
        if bal_side in ("top", "bottom"):
            rooms = _force_balcony_side(rooms, w, h, bal_side)
        rooms = _keep_only_exterior_balconies(rooms, x0, y0, total_w, total_h)
        all_rooms.extend([(nm, x0+x, y0+y, rw, rh) for (nm, x, y, rw, rh) in rooms])
        flat_bounds.append((x0, y0, w, h))

    all_rooms.append(("corridor", corr_x, 0, corr_w, total_h))
    if service_spec.get("lift_spec"):
        all_rooms.extend(_build_vertical_lifts(corr_x, 0, corr_w, total_h, service_spec["lift_spec"]))
    if service_spec.get("stair_spec"):
        all_rooms.extend(_build_vertical_stairs(corr_x, 0, corr_w, total_h, service_spec["stair_spec"]))
    return all_rooms, total_w, total_h, flat_bounds, {"orientation": "vertical", "service_spec": service_spec}


def _place_one_side_horizontal(unit_defs, side="bottom"):
    target_side = "top" if side == "bottom" else "bottom"
    service_spec = _choose_service_spec("horizontal", corridor_side=side)
    corr_h = service_spec["corr_span"]
    total_w = sum(w for (_, _, w, _) in unit_defs)
    row_h   = max(h for (_, _, _, h) in unit_defs)
    total_h = corr_h + row_h
    corr_y  = 0 if side == "bottom" else row_h

    def _y0(h):
        return corr_h if side == "bottom" else row_h - h

    tent_pos, x = [], 0
    for (_, _, w, h) in unit_defs:
        tent_pos.append((x, _y0(h))); x += w

    defs = []
    for (btype, rooms, w, h), (x0, y0) in zip(unit_defs, tent_pos):
        ext  = _exterior_sides_of_flat(x0, y0, w, h, total_w, total_h)
        best = _pick_balcony_side(ext)
        if best is None:
            rooms = _remove_balconies(rooms)
        elif best in ("left", "right"):
            result = _gen_flat_side_balcony(btype, side=best)
            rooms, w, h = result
        rooms = _force_living_side(rooms, w, h, target_side)
        defs.append((rooms, btype, w, h, best))

    total_w = sum(w for (_, _, w, _, _) in defs)
    row_h = max(h for (_, _, _, h, _) in defs)
    total_h = corr_h + row_h
    corr_y = 0 if side == "bottom" else row_h
    positions, x = [], 0
    for (_, _, w, h, _) in defs:
        positions.append((x, _y0(h))); x += w

    all_rooms, flat_bounds = [], []
    for (rooms, btype, w, h, bal_side), (x0, y0) in zip(defs, positions):
        if bal_side in ("top", "bottom"):
            rooms = _force_balcony_side(rooms, w, h, bal_side)
        rooms = _keep_only_exterior_balconies(rooms, x0, y0, total_w, total_h)
        all_rooms.extend([(nm, x0+x, y0+y, rw, rh) for (nm, x, y, rw, rh) in rooms])
        flat_bounds.append((x0, y0, w, h))

    all_rooms.append(("corridor", 0, corr_y, total_w, corr_h))
    if service_spec.get("lift_spec"):
        all_rooms.extend(_build_horizontal_lifts(0, corr_y, total_w, corr_h, service_spec["lift_spec"]))
    if service_spec.get("stair_spec"):
        all_rooms.extend(_build_horizontal_stairs(0, corr_y, total_w, corr_h, service_spec["stair_spec"]))
    return all_rooms, total_w, total_h, flat_bounds, {"orientation": "horizontal", "service_spec": service_spec}


def make_cluster_from_type(apt_type):
    if apt_type in ("CLUSTER_3xRANDOM", "CLUSTER_4xRANDOM"):
        n_units = 3 if "3x" in apt_type else 4
        choices = ["1BHK", "2BHK", "3BHK"]
        unit_types = [random.choice(choices) for _ in range(n_units)]
    elif apt_type == "CLUSTER_4xMIXED":
        unit_types = ["3BHK", "2BHK", "2BHK", "2BHK"]
        random.shuffle(unit_types)
        n_units = 4
    else:
        n_str, bkey = apt_type.replace("CLUSTER_","").split("x",1)
        n_units = int(n_str)
        unit_types = [bkey] * n_units

    unit_defs = []
    for btype in unit_types:
        rooms, w, h, _ = gen_flat(btype)
        unit_defs.append((btype, rooms, w, h))

    # Layout variety
    if n_units == 2:
        if random.random() < 0.5:
            return _place_vertical_corridor(unit_defs, [0], [1])
        return _place_one_side_vertical(unit_defs, side=random.choice(["left", "right"]))

    if n_units == 3:
        variant = random.choice(["one_vs_two", "all_vertical", "all_horizontal"])
        if variant == "one_vs_two":
            if random.random() < 0.5:
                return _place_vertical_corridor(unit_defs, [0], [1,2])
            return _place_vertical_corridor(unit_defs, [0,1], [2])
        if variant == "all_vertical":
            return _place_one_side_vertical(unit_defs, side=random.choice(["left", "right"]))
        return _place_one_side_horizontal(unit_defs, side=random.choice(["top", "bottom"]))

    # n_units == 4
    if apt_type == "CLUSTER_4xMIXED":
        # Prefer 1 vs 3 composition to match requested example.
        idx3 = next(i for i, (t, *_rest) in enumerate(unit_defs) if t == "3BHK")
        other_ids = [i for i in range(4) if i != idx3]
        if random.random() < 0.75:
            if random.random() < 0.5:
                return _place_vertical_corridor(unit_defs, [idx3], other_ids)
            return _place_vertical_corridor(unit_defs, other_ids, [idx3])

    variant = random.choice(["two_two", "all_vertical", "all_horizontal", "one_three"])
    if variant == "two_two":
        return _place_vertical_corridor(unit_defs, [0,1], [2,3])
    if variant == "all_vertical":
        return _place_one_side_vertical(unit_defs, side=random.choice(["left", "right"]))
    if variant == "all_horizontal":
        return _place_one_side_horizontal(unit_defs, side=random.choice(["top", "bottom"]))
    if random.random() < 0.5:
        return _place_vertical_corridor(unit_defs, [0], [1,2,3])
    return _place_vertical_corridor(unit_defs, [0,1,2], [3])


def make_cluster(n_units, bkey):
    return make_cluster_from_type(f"CLUSTER_{n_units}x{bkey}")


# ═══════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def f2p(v): return int(v * SCALE)

def to_px(rooms_ft):
    return [{"idx":i,"name":n,"label":LID[n],
             "x":f2p(x)+PAD,"y":f2p(y)+PAD,"w":f2p(w),"h":f2p(h)}
            for i,(n,x,y,w,h) in enumerate(rooms_ft)]

def blit(fd, md, lbl, rect, col, lid):
    x1,y1,x2,y2 = (int(v) for v in rect)
    fd.rectangle([x1,y1,x2,y2], fill=col)
    md.rectangle([x1,y1,x2,y2], fill=col)
    H,W = lbl.shape
    lbl[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = lid

def blit_bw(bd, rect, wall):
    x1,y1,x2,y2 = (int(v) for v in rect)
    bd.rectangle([x1,y1,x2,y2], fill=(0,0,0) if wall else (255,255,255))

def _clip_rect(lbl, rect):
    x1,y1,x2,y2 = (int(v) for v in rect)
    H,W = lbl.shape
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

def draw_wall_outline(fd, bd, md, lbl, rect, col, lid, wall_mode="filled_black"):
    x1,y1,x2,y2 = (int(v) for v in rect)
    if x2 <= x1 or y2 <= y1:
        return
    if wall_mode == "hollow":
        fd.rectangle([x1,y1,x2,y2], outline=col, width=1)
        bd.rectangle([x1,y1,x2,y2], outline=(0,0,0), width=1)
        md.rectangle([x1,y1,x2,y2], fill=col)
    else:
        fill_col = (120,120,120) if wall_mode == "filled_gray" else col
        fd.rectangle([x1,y1,x2,y2], fill=fill_col)
        bd.rectangle([x1,y1,x2,y2], fill=(0,0,0))
        md.rectangle([x1,y1,x2,y2], fill=col)
    H,W = lbl.shape
    yy1,yy2 = max(0,y1), min(H,y2)
    xx1,xx2 = max(0,x1), min(W,x2)
    if xx2 <= xx1 or yy2 <= yy1:
        return
    lbl[yy1:yy2, xx1:xx2] = lid

def _erase_opening(fd, bd, md, lbl, rect):
    x1,y1,x2,y2 = _clip_rect(lbl, rect)
    if x2 <= x1 or y2 <= y1:
        return x1,y1,x2,y2
    fd.rectangle([x1,y1,x2,y2], fill=(255,255,255))
    md.rectangle([x1,y1,x2,y2], fill=(255,255,255))
    bd.rectangle([x1,y1,x2,y2], fill=(255,255,255))
    lbl[y1:y2, x1:x2] = 0
    return x1,y1,x2,y2

def draw_window_gap(fd, bd, md, lbl, rect, col, lid):
    """Architectural window symbol:
      - outer and inner wall boundary lines (full span)
      - a frame rectangle with END_PAD gap at both ends of the opening
      - two glass lines inside the frame
    """
    x1, y1, x2, y2 = _erase_opening(fd, bd, md, lbl, rect)
    if x2 <= x1 or y2 <= y1:
        return

    lbl[y1:y2, x1:x2] = lid
    horizontal = (x2 - x1) >= (y2 - y1)
    END_PAD = 3  # gap at each end between wall edge and frame

    if horizontal:
        depth = y2 - y1
        g1 = y1 + max(1, depth // 3)
        g2 = y2 - max(1, depth // 3)
        fx1, fx2 = x1 + END_PAD, x2 - END_PAD
        for draw, color in ((fd, col), (md, col), (bd, (0, 0, 0))):
            draw.line([x1, y1, x2, y1], fill=color, width=1)       # outer boundary
            draw.line([x1, y2, x2, y2], fill=color, width=1)       # inner boundary
            if fx2 > fx1:
                draw.rectangle([fx1, y1, fx2, y2], outline=color)  # frame box
                if g1 > y1 and g1 < y2:
                    draw.line([fx1 + 1, g1, fx2 - 1, g1], fill=color, width=1)
                if g2 > y1 and g2 < y2 and g2 != g1:
                    draw.line([fx1 + 1, g2, fx2 - 1, g2], fill=color, width=1)
    else:
        depth = x2 - x1
        g1 = x1 + max(1, depth // 3)
        g2 = x2 - max(1, depth // 3)
        fy1, fy2 = y1 + END_PAD, y2 - END_PAD
        for draw, color in ((fd, col), (md, col), (bd, (0, 0, 0))):
            draw.line([x1, y1, x1, y2], fill=color, width=1)       # outer boundary
            draw.line([x2, y1, x2, y2], fill=color, width=1)       # inner boundary
            if fy2 > fy1:
                draw.rectangle([x1, fy1, x2, fy2], outline=color)  # frame box
                if g1 > x1 and g1 < x2:
                    draw.line([g1, fy1 + 1, g1, fy2 - 1], fill=color, width=1)
                if g2 > x1 and g2 < x2 and g2 != g1:
                    draw.line([g2, fy1 + 1, g2, fy2 - 1], fill=color, width=1)

def draw_door_with_arc(fd, bd, md, lbl, rect, wall_is_horizontal, col, lid, swing=1):
    x1,y1,x2,y2 = _erase_opening(fd, bd, md, lbl, rect)
    if x2 <= x1 or y2 <= y1:
        return

    # SEG MASK:
    # only label the straight opening rectangle between walls
    md.rectangle([x1,y1,x2,y2], fill=col)
    lbl[y1:y2, x1:x2] = lid

    def _draw_line(xa, ya, xb, yb, width=2):
        # visible floor plan
        fd.line([xa,ya,xb,yb], fill=col, width=width)

        # black-white image visibility
        bd.line([xa,ya,xb,yb], fill=(0,0,0), width=width)

        # IMPORTANT:
        # do not draw this line on md segmentation mask

    def _draw_arc(box, a0, a1, width=2):
        # visible floor plan
        fd.arc(box, a0, a1, fill=col, width=width)

        # black-white image visibility
        bd.arc(box, a0, a1, fill=(0,0,0), width=width)

        # IMPORTANT:
        # do not draw arc on md segmentation mask

    if wall_is_horizontal:
        y_wall = (y1 + y2) // 2
        r = max(10, x2 - x1)
        if swing >= 0:
            hinge_x, hinge_y = x1, y_wall
            leaf_end = (hinge_x, hinge_y + r)
            box = [hinge_x - r, hinge_y - r, hinge_x + r, hinge_y + r]
            angles = (0, 90)
        else:
            hinge_x, hinge_y = x2, y_wall
            leaf_end = (hinge_x, hinge_y - r)
            box = [hinge_x - r, hinge_y - r, hinge_x + r, hinge_y + r]
            angles = (180, 270)

        _draw_line(hinge_x, hinge_y, leaf_end[0], leaf_end[1])
        _draw_arc(box, angles[0], angles[1])

    else:
        x_wall = (x1 + x2) // 2
        r = max(10, y2 - y1)
        if swing >= 0:
            hinge_x, hinge_y = x_wall, y1
            leaf_end = (hinge_x + r, hinge_y)
            box = [hinge_x - r, hinge_y - r, hinge_x + r, hinge_y + r]
            angles = (0, 90)
        else:
            hinge_x, hinge_y = x_wall, y2
            leaf_end = (hinge_x - r, hinge_y)
            box = [hinge_x - r, hinge_y - r, hinge_x + r, hinge_y + r]
            angles = (180, 270)

        _draw_line(hinge_x, hinge_y, leaf_end[0], leaf_end[1])
        _draw_arc(box, angles[0], angles[1])

def draw_plain_opening(fd, bd, md, lbl, rect):
    _erase_opening(fd, bd, md, lbl, rect)

def _opening_rect_between(ra, rb, width):
    tol = 2
    checks = [
        (ra["x"]+ra["w"], rb["x"],       False),
        (rb["x"]+rb["w"], ra["x"],       False),
        (ra["y"]+ra["h"], rb["y"],       True ),
        (rb["y"]+rb["h"], ra["y"],       True ),
    ]
    for ea, eb, horiz in checks:
        if abs(ea-eb) > tol:
            continue
        wc = eb - WALL_T // 2
        if horiz:
            ox0 = max(ra["x"],rb["x"]) + WALL_T
            ox1 = min(ra["x"]+ra["w"],rb["x"]+rb["w"]) - WALL_T
            if ox1 - ox0 < width:
                continue
            mid = (ox0 + ox1) // 2
            swing = 1 if rb["y"] >= ra["y"] else -1
            return [mid-width//2, wc, mid+width//2, wc+WALL_T], True, swing
        else:
            oy0 = max(ra["y"],rb["y"]) + WALL_T
            oy1 = min(ra["y"]+ra["h"],rb["y"]+rb["h"]) - WALL_T
            if oy1 - oy0 < width:
                continue
            mid = (oy0 + oy1) // 2
            swing = 1 if rb["x"] >= ra["x"] else -1
            return [wc, mid-width//2, wc+WALL_T, mid+width//2], False, swing
    return None, None, None

def _contains_room(container, inner, margin=2):
    return (
        container["x"] <= inner["x"] + margin and
        container["y"] <= inner["y"] + margin and
        container["x"] + container["w"] >= inner["x"] + inner["w"] - margin and
        container["y"] + container["h"] >= inner["y"] + inner["h"] - margin and
        container["idx"] != inner["idx"]
    )

def draw_contained_room_door(fd, bd, md, lbl, inner, col, lid, width):
    # Door on top wall of contained room, opening into the containing room.
    w = min(width, max(10, inner["w"] - 2 * WALL_T))
    mid = inner["x"] + inner["w"] // 2
    rect = [mid - w//2, inner["y"] - WALL_T//2, mid + w//2, inner["y"] + WALL_T//2]
    draw_door_with_arc(fd, bd, md, lbl, rect, True, col, lid, -1)
    return rect

def _touch(a, b, tol=2):
    ax1,ay1,ax2,ay2 = a["x"],a["y"],a["x"]+a["w"],a["y"]+a["h"]
    bx1,by1,bx2,by2 = b["x"],b["y"],b["x"]+b["w"],b["y"]+b["h"]
    vt = abs(ax2-bx1)<=tol or abs(bx2-ax1)<=tol
    ht = abs(ay2-by1)<=tol or abs(by2-ay1)<=tol
    return (vt and min(ay2,by2)-max(ay1,by1)>WALL_T) or \
           (ht and min(ax2,bx2)-max(ax1,bx1)>WALL_T)

def _try_door(fd, bd, md, lbl, ra, rb, dw, col, lid):
    rect, horiz, swing = _opening_rect_between(ra, rb, dw)
    if rect is None:
        return False
    draw_door_with_arc(fd, bd, md, lbl, rect, horiz, col, lid, swing)
    return True
def _build_r2f(rooms, flat_bounds_ft):
    if not flat_bounds_ft: return {r["idx"]:0 for r in rooms}
    bpx = [(k,f2p(x)+PAD,f2p(y)+PAD,f2p(x+w)+PAD,f2p(y+h)+PAD)
           for k,(x,y,w,h) in enumerate(flat_bounds_ft)]
    m = {}
    for r in rooms:
        cx,cy = r["x"]+r["w"]//2, r["y"]+r["h"]//2
        for k,x1,y1,x2,y2 in bpx:
            if x1<=cx<=x2 and y1<=cy<=y2: m[r["idx"]]=k; break
    return m

def _filter_exterior_sides(r, sides, all_rooms, tol=6):
    """Remove any side that has another room directly adjacent — that wall is
    interior (shared), so no window should appear there."""
    result = []
    for side in sides:
        blocked = False
        for other in all_rooms:
            if other["idx"] == r["idx"]:
                continue
            rx1, ry1 = r["x"], r["y"]
            rx2, ry2 = r["x"] + r["w"], r["y"] + r["h"]
            ox1, oy1 = other["x"], other["y"]
            ox2, oy2 = other["x"] + other["w"], other["y"] + other["h"]
            if side == "right":
                if abs(rx2 - ox1) <= tol and min(ry2, oy2) - max(ry1, oy1) > 4:
                    blocked = True; break
            elif side == "left":
                if abs(rx1 - ox2) <= tol and min(ry2, oy2) - max(ry1, oy1) > 4:
                    blocked = True; break
            elif side == "bottom":
                if abs(ry2 - oy1) <= tol and min(rx2, ox2) - max(rx1, ox1) > 4:
                    blocked = True; break
            elif side == "top":
                if abs(ry1 - oy2) <= tol and min(rx2, ox2) - max(rx1, ox1) > 4:
                    blocked = True; break
        if not blocked:
            result.append(side)
    return result


def _apt_boundary_sides(r, ax0, ay0, ax1, ay1, tol=3):
    sides = set()
    if abs(r["x"] - ax0) <= tol:          sides.add("left")
    if abs(r["x"]+r["w"] - ax1) <= tol:   sides.add("right")
    if abs(r["y"] - ay0) <= tol:          sides.add("top")
    if abs(r["y"]+r["h"] - ay1) <= tol:   sides.add("bottom")
    return sides


# ═══════════════════════════════════════════════════════════════
#  RENDER PLAN  (door + window logic driven by RULES constants)
# ═══════════════════════════════════════════════════════════════
def _fmt_dim_ft(px_len):
    ft = px_len / SCALE
    if abs(ft - round(ft)) < 1e-6:
        return f"{int(round(ft))}'"
    return f"{ft:.1f}'"


def _draw_dim_h(draw, x1, x2, y, text, col):
    if x2 - x1 < 16:
        return
    draw.line([x1, y, x2, y], fill=col, width=1)
    draw.line([x1, y-3, x1, y+3], fill=col, width=1)
    draw.line([x2, y-3, x2, y+3], fill=col, width=1)
    tx = (x1 + x2) // 2
    try:
        draw.text((tx, y-4), text, fill=col, anchor="ms")
    except:
        draw.text((tx - len(text) * 2, y-10), text, fill=col)


def _draw_dim_v(draw, x, y1, y2, text, col):
    if y2 - y1 < 16:
        return
    draw.line([x, y1, x, y2], fill=col, width=1)
    draw.line([x-3, y1, x+3, y1], fill=col, width=1)
    draw.line([x-3, y2, x+3, y2], fill=col, width=1)
    ty = (y1 + y2) // 2
    try:
        draw.text((x+4, ty), text, fill=col, anchor="ls")
    except:
        draw.text((x+4, ty-5), text, fill=col)


def draw_room_dimensions(fd, rooms, color=(170,170,170)):
    skip = {"background", "apartment", "wall"}
    for r in rooms:
        if r["name"] in skip:
            continue
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        if w < 24 or h < 24:
            continue
        inset_x = max(5, min(10, w // 6))
        inset_y = max(5, min(10, h // 6))
        top_y = y + inset_y
        left_x = x + inset_x
        _draw_dim_h(fd, x + inset_x, x + w - inset_x, top_y, _fmt_dim_ft(w), color)
        _draw_dim_v(fd, left_x, y + inset_y, y + h - inset_y, _fmt_dim_ft(h), color)



def _draw_stair_pattern(draw, room, color=(90,90,90)):
    if room["name"] != "stair":
        return
    x, y, w, h = room["x"], room["y"], room["w"], room["h"]
    if w < 18 or h < 18:
        return
    pad = max(3, min(8, min(w, h) // 8))
    ix0, iy0 = x + pad, y + pad
    ix1, iy1 = x + w - pad, y + h - pad
    iw, ih = ix1 - ix0, iy1 - iy0
    style = ["straight", "l_shaped", "u_shaped"][room["idx"] % 3]

    def _treads_h(xx0, yy0, xx1, yy1, n=8):
        if yy1 <= yy0 or xx1 <= xx0: return
        for i in range(n + 1):
            yy = int(round(yy0 + (yy1 - yy0) * i / max(1, n)))
            draw.line([xx0, yy, xx1, yy], fill=color, width=1)

    def _treads_v(xx0, yy0, xx1, yy1, n=8):
        if yy1 <= yy0 or xx1 <= xx0: return
        for i in range(n + 1):
            xx = int(round(xx0 + (xx1 - xx0) * i / max(1, n)))
            draw.line([xx, yy0, xx, yy1], fill=color, width=1)

    draw.rectangle([ix0, iy0, ix1, iy1], outline=color, width=1)
    if style == "straight":
        if iw >= ih:
            _treads_v(ix0 + 2, iy0 + 2, ix1 - 2, iy1 - 2, max(6, iw // 8))
        else:
            _treads_h(ix0 + 2, iy0 + 2, ix1 - 2, iy1 - 2, max(6, ih // 8))
    elif style == "l_shaped":
        if iw >= ih:
            midx = ix0 + int(iw * 0.58)
            midy = iy0 + int(ih * 0.45)
            draw.line([midx, iy0, midx, midy], fill=color, width=1)
            draw.line([midx, midy, ix1, midy], fill=color, width=1)
            _treads_h(ix0 + 2, iy0 + 2, midx - 2, iy1 - 2, max(4, ih // 7))
            _treads_v(midx + 2, iy0 + 2, ix1 - 2, midy - 2, max(4, (ix1-midx) // 7))
        else:
            midx = ix0 + int(iw * 0.45)
            midy = iy0 + int(ih * 0.58)
            draw.line([ix0, midy, midx, midy], fill=color, width=1)
            draw.line([midx, iy0, midx, midy], fill=color, width=1)
            _treads_v(ix0 + 2, midy + 2, ix1 - 2, iy1 - 2, max(4, iw // 7))
            _treads_h(ix0 + 2, iy0 + 2, midx - 2, midy - 2, max(4, midy // 7))
    else:  # u_shaped
        if iw >= ih:
            lane = max(5, iw // 3)
            draw.line([ix0 + lane, iy0, ix0 + lane, iy1], fill=color, width=1)
            draw.line([ix1 - lane, iy0, ix1 - lane, iy1], fill=color, width=1)
            _treads_h(ix0 + 2, iy0 + 2, ix0 + lane - 2, iy1 - 2, max(4, ih // 7))
            _treads_h(ix1 - lane + 2, iy0 + 2, ix1 - 2, iy1 - 2, max(4, ih // 7))
            draw.line([ix0 + lane, iy1 - 2, ix1 - lane, iy1 - 2], fill=color, width=1)
        else:
            lane = max(5, ih // 3)
            draw.line([ix0, iy0 + lane, ix1, iy0 + lane], fill=color, width=1)
            draw.line([ix0, iy1 - lane, ix1, iy1 - lane], fill=color, width=1)
            _treads_v(ix0 + 2, iy0 + 2, ix1 - 2, iy0 + lane - 2, max(4, iw // 7))
            _treads_v(ix0 + 2, iy1 - lane + 2, ix1 - 2, iy1 - 2, max(4, iw // 7))
            draw.line([ix1 - 2, iy0 + lane, ix1 - 2, iy1 - lane], fill=color, width=1)


def _rect_contains_outer(outer, inner, margin=2):
    return (
        outer["idx"] != inner["idx"] and
        outer["x"] <= inner["x"] + margin and
        outer["y"] <= inner["y"] + margin and
        outer["x"] + outer["w"] >= inner["x"] + inner["w"] - margin and
        outer["y"] + outer["h"] >= inner["y"] + inner["h"] - margin
    )


def _simplify_polygon_points(points):
    if len(points) <= 2:
        return points
    pts = points[:]
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 2:
        changed = False
        out = []
        n = len(pts)
        for i in range(n):
            a = pts[(i-1) % n]
            b = pts[i]
            c = pts[(i+1) % n]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                changed = True
                continue
            out.append(b)
        pts = out
    return [[int(x), int(y)] for x, y in pts]


def _cycle_area(poly):
    if len(poly) < 3:
        return 0
    a = 0
    for i in range(len(poly)):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1) % len(poly)]
        a += x1*y2 - x2*y1
    return abs(a) / 2


def _visible_polygons_for_room(room, rooms):
    # Build the visible polygon of a room after subtracting contained rooms.
    # This keeps generation unchanged; it only changes metadata.
    x0, y0, x1, y1 = room["x"], room["y"], room["x"] + room["w"], room["y"] + room["h"]
    holes = []
    for other in rooms:
        if _rect_contains_outer(room, other):
            hx0, hy0 = max(x0, other["x"]), max(y0, other["y"])
            hx1, hy1 = min(x1, other["x"] + other["w"]), min(y1, other["y"] + other["h"])
            if hx1 > hx0 and hy1 > hy0:
                holes.append((hx0, hy0, hx1, hy1))

    if not holes:
        return [[[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]]

    xs = sorted(set([x0, x1] + [v for h in holes for v in (h[0], h[2])]))
    ys = sorted(set([y0, y1] + [v for h in holes for v in (h[1], h[3])]))

    filled = set()
    for ix in range(len(xs)-1):
        for iy in range(len(ys)-1):
            cx = (xs[ix] + xs[ix+1]) / 2
            cy = (ys[iy] + ys[iy+1]) / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            inside_hole = any(hx0 <= cx <= hx1 and hy0 <= cy <= hy1 for hx0,hy0,hx1,hy1 in holes)
            if not inside_hole:
                filled.add((ix, iy))

    if not filled:
        return []

    # Boundary edges of the visible cell union.
    edges = set()
    def add_edge(a,b):
        if a > b:
            a,b = b,a
        if (a,b) in edges:
            edges.remove((a,b))
        else:
            edges.add((a,b))

    for ix,iy in filled:
        xL,xR = xs[ix], xs[ix+1]
        yT,yB = ys[iy], ys[iy+1]
        if (ix, iy-1) not in filled: add_edge((xL,yT),(xR,yT))
        if (ix, iy+1) not in filled: add_edge((xL,yB),(xR,yB))
        if (ix-1, iy) not in filled: add_edge((xL,yT),(xL,yB))
        if (ix+1, iy) not in filled: add_edge((xR,yT),(xR,yB))

    # Convert boundary graph into cycles.
    adj = {}
    for a,b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    unused = set(edges)
    cycles = []
    while unused:
        a,b = next(iter(unused))
        start = a
        prev = None
        cur = a
        nxt = b
        poly = [cur]
        guard = 0
        while guard < 500:
            guard += 1
            edge = (cur, nxt) if cur < nxt else (nxt, cur)
            if edge in unused:
                unused.remove(edge)
            prev, cur = cur, nxt
            if cur == start:
                break
            poly.append(cur)
            candidates = [v for v in adj.get(cur, []) if v != prev]
            if not candidates:
                break
            # Prefer continuing straight; otherwise take any available neighbor.
            straight = []
            if len(poly) >= 2:
                pprev = prev
                for c in candidates:
                    if (pprev[0] == cur[0] == c[0]) or (pprev[1] == cur[1] == c[1]):
                        straight.append(c)
            nxt = straight[0] if straight else candidates[0]
        if len(poly) >= 4:
            cycles.append(_simplify_polygon_points(poly))

    cycles = [c for c in cycles if _cycle_area(c) > 0]
    cycles.sort(key=_cycle_area, reverse=True)
    return cycles or [[[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]]


def _build_graph_nodes(rooms):
    nodes = []
    for r in rooms:
        polys = _visible_polygons_for_room(r, rooms)
        main_poly = polys[0] if polys else [[r["x"],r["y"]],[r["x"]+r["w"],r["y"]],[r["x"]+r["w"],r["y"]+r["h"]],[r["x"],r["y"]+r["h"]]]
        node = {
            "id": int(r["idx"]),
            "name": r["name"],
            "label_id": int(r["label"]),
            "bbox_px": main_poly,
        }
        if len(polys) > 1:
            node["holes_px"] = polys[1:]
        nodes.append(node)
    return nodes


def _rect_center(rect):
    x1,y1,x2,y2 = [int(v) for v in rect]
    return [(x1+x2)//2, (y1+y2)//2]


def _edge_payload(a, b, conn_type, rect):
    x1,y1,x2,y2 = [int(v) for v in rect]
    return {
        "from": a["name"] if isinstance(a, dict) else str(a),
        "from_id": int(a["idx"]) if isinstance(a, dict) and "idx" in a else None,
        "to": b["name"] if isinstance(b, dict) else str(b),
        "to_id": int(b["idx"]) if isinstance(b, dict) and "idx" in b else None,
        "connection_type": conn_type,
        "coords_px": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
        "center_px": _rect_center(rect),
    }

def render_plan(apt_type):
    dw  = random.randint(*DOOR_W_RANGE)
    sdw = int(dw * 1.7)
    # Window length is computed from the selected exterior wall length.
    wall_mode = random.choice(["filled_black", "filled_gray", "hollow"])
    bw_with_text = random.random() < 0.5
    bw_with_dimensions = random.random() < 0.5

    is_cluster     = apt_type.startswith("CLUSTER_")
    flat_bounds_ft = None; corr_x_ft = None

    if is_cluster:
        rooms_ft, tw_ft, th_ft, flat_bounds_ft, corr_x_ft = make_cluster_from_type(apt_type)
        val = validate(apt_type, rooms_ft, tw_ft, th_ft, strict=False)
    else:
        rooms_ft, tw_ft, th_ft, val = gen_flat(apt_type)

    render_plan.last_val = val
    pw = f2p(tw_ft)+2*PAD; ph = f2p(th_ft)+2*PAD
    fp  = Image.new("RGB",(pw,ph),LCOL[LID["background"]])
    bw  = Image.new("RGB",(pw,ph),(255,255,255))
    mk  = Image.new("RGB",(pw,ph),LCOL[LID["background"]])
    lbl = np.zeros((ph,pw),dtype=np.uint8)
    fd  = ImageDraw.Draw(fp); bd = ImageDraw.Draw(bw); md = ImageDraw.Draw(mk)

    rooms = to_px(rooms_ft)
    r2f   = _build_r2f(rooms, flat_bounds_ft)

    # Per-room flat boundary (for window placement) from RULES: at_flat_boundary=True
    if flat_bounds_ft:
        apt_bounds = {}
        for r in rooms:
            fid = r2f.get(r["idx"])
            if fid is not None and fid < len(flat_bounds_ft):
                xf,yf,wf,hf = flat_bounds_ft[fid]
                apt_bounds[r["idx"]] = (f2p(xf)+PAD, f2p(yf)+PAD,
                                        f2p(xf+wf)+PAD, f2p(yf+hf)+PAD)
            else:
                apt_bounds[r["idx"]] = (PAD, PAD, PAD+f2p(tw_ft), PAD+f2p(th_ft))
    else:
        bnd = (PAD, PAD, PAD+f2p(tw_ft), PAD+f2p(th_ft))
        apt_bounds = {r["idx"]: bnd for r in rooms}

    # if flat_bounds_ft:
    #     ac,al = LCOL[LID["apartment"]],LID["apartment"]; bt=3
    #     for (xf,yf,wf,hf) in flat_bounds_ft:
    #         ax=f2p(xf)+PAD; ay=f2p(yf)+PAD; aw=f2p(wf); ah=f2p(hf)
    #         for rc in [[ax,ay,ax+aw,ay+bt],[ax,ay+ah-bt,ax+aw,ay+ah],
    #                    [ax,ay,ax+bt,ay+ah],[ax+aw-bt,ay,ax+aw,ay+ah]]:
    #             blit(fd,md,lbl,rc,ac,al)

    for r in rooms:
        rc = [r["x"],r["y"],r["x"]+r["w"],r["y"]+r["h"]]
        blit(fd,md,lbl,rc,LCOL[r["label"]],r["label"]); blit_bw(bd,rc,False)

    for r in rooms:
        if r["name"] == "stair":
            _draw_stair_pattern(fd, r, color=(80,80,80))
            _draw_stair_pattern(bd, r, color=(60,60,60))

    wc,wl = LCOL[LID["wall"]],LID["wall"]
    drawn_walls = set()
    half = WALL_T // 2

    def _same_living_neighbor_side(r, side, tol=2):
        # Used after internal flat corridors are renamed to living_space.
        # Skip the shared wall between adjacent living_space rectangles so they
        # read as one merged hall, with no door/opening between them.
        if r["name"] != "living_space":
            return False
        rx1, ry1 = r["x"], r["y"]
        rx2, ry2 = r["x"] + r["w"], r["y"] + r["h"]
        for o in rooms:
            if o["idx"] == r["idx"] or o["name"] != "living_space":
                continue
            # Only merge pieces from the same flat; separate apartments must keep walls.
            if r2f.get(o["idx"]) != r2f.get(r["idx"]):
                continue
            ox1, oy1 = o["x"], o["y"]
            ox2, oy2 = o["x"] + o["w"], o["y"] + o["h"]
            if side == "top" and abs(ry1 - oy2) <= tol and min(rx2, ox2) - max(rx1, ox1) > WALL_T:
                return True
            if side == "bottom" and abs(ry2 - oy1) <= tol and min(rx2, ox2) - max(rx1, ox1) > WALL_T:
                return True
            if side == "left" and abs(rx1 - ox2) <= tol and min(ry2, oy2) - max(ry1, oy1) > WALL_T:
                return True
            if side == "right" and abs(rx2 - ox1) <= tol and min(ry2, oy2) - max(ry1, oy1) > WALL_T:
                return True
        return False

    for r in rooms:
        x,y,w,h = r["x"],r["y"],r["w"],r["h"]
        wall_rects = [
            ("top",    [x,          y-half,     x+w,          y+WALL_T-half]),
            ("bottom", [x,          y+h-half,   x+w,          y+h+WALL_T-half]),
            ("left",   [x-half,     y,          x+WALL_T-half,y+h]),
            ("right",  [x+w-half,   y,          x+w+WALL_T-half,y+h]),
        ]
        for side_name, rc in wall_rects:
            if _same_living_neighbor_side(r, side_name):
                continue
            key = tuple(int(v) for v in rc)
            if key in drawn_walls:
                continue
            drawn_walls.add(key)
            draw_wall_outline(fd, bd, md, lbl, rc, wc, wl, wall_mode=wall_mode)

    dc,dl   = LCOL[LID["door"]],LID["door"]
    fdc,fdl = LCOL[LID["front_door"]],LID["front_door"]
    has_door      = set()
    door_count    = Counter()
    door_partners = {}
    passage_graph = {}
    graph_edges = []
    max_door_given = set()   # tracks rooms that hit their DOOR_LIMITS max_doors
    used_openings = set()
    occupied_openings = []   # door/front-door/opening rectangles reserved before drawing windows

    def _norm_rect(rect):
        x1,y1,x2,y2 = [int(v) for v in rect]
        return [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]

    def _rects_overlap(a, b, pad=0):
        ax1,ay1,ax2,ay2 = _norm_rect(a)
        bx1,by1,bx2,by2 = _norm_rect(b)
        ax1 -= pad; ay1 -= pad; ax2 += pad; ay2 += pad
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def _open_key(rect, tag="open"):
        x1,y1,x2,y2 = _norm_rect(rect)
        return (tag, x1, y1, x2, y2)

    def _claim_opening(rect, tag="open"):
        key = _open_key(rect, tag)
        if key in used_openings:
            return False
        used_openings.add(key)
        # Windows are drawn after doors. Store all door/opening zones so
        # windows can avoid door, front_door, lift_door, stair_door and gaps.
        if tag in ("door", "front_door", "lift_door", "stair_door", "gap"):
            occupied_openings.append(_norm_rect(rect))
        return True

    def _window_is_clear(rect, clearance=12):
        return all(not _rects_overlap(rect, occ, pad=clearance) for occ in occupied_openings)

    def _add_passage(ra, rb):
        passage_graph.setdefault(ra["idx"], set()).add(rb["idx"])
        passage_graph.setdefault(rb["idx"], set()).add(ra["idx"])

    def _record_edge(ra, rb, conn_type, rect):
        if rect is not None:
            graph_edges.append(_edge_payload(ra, rb, conn_type, rect))

    def _record_door(ra, rb):
        has_door.add(ra["idx"]); has_door.add(rb["idx"])
        door_count[ra["idx"]] += 1
        door_count[rb["idx"]] += 1
        door_partners.setdefault(ra["idx"], []).append(rb["name"])
        door_partners.setdefault(rb["idx"], []).append(ra["name"])
        _add_passage(ra, rb)

    def _door_limit_reached(room):
        if room["name"] == "bathroom":
            return door_count[room["idx"]] >= 1
        if room["name"] == "bedroom":
            return door_count[room["idx"]] >= 2
        return False

    def do_door(ra, rb, w=None, col=None, lid=None, force=False):
        # Bathroom: exactly one door. Bedroom: normally up to two; force allows balcony attachment.
        if ra["name"] == "bathroom" and door_count[ra["idx"]] >= 1:
            return False
        if rb["name"] == "bathroom" and door_count[rb["idx"]] >= 1:
            return False
        if not force:
            if _door_limit_reached(ra):
                return False
            if _door_limit_reached(rb):
                return False

        # Enforce RULE-based one-door rooms, but bedrooms are allowed up to two.
        ra_max = DOOR_LIMITS.get(ra["name"], {}).get("max_doors")
        rb_max = DOOR_LIMITS.get(rb["name"], {}).get("max_doors")
        if ra["name"] == "bedroom": ra_max = None
        if rb["name"] == "bedroom": rb_max = None
        if ra_max == 1 and ra["idx"] in max_door_given: return False
        if rb_max == 1 and rb["idx"] in max_door_given: return False

        rect, horiz, swing = _opening_rect_between(ra, rb, w or dw)
        if rect is None or not _claim_opening(rect, "door"):
            return False
        draw_door_with_arc(fd, bd, md, lbl, rect, horiz, col or dc, lid or dl, swing)
        _record_door(ra, rb)
        _record_edge(ra, rb, "door", rect)
        if ra_max == 1: max_door_given.add(ra["idx"])
        if rb_max == 1: max_door_given.add(rb["idx"])
        return True


    # ── Explicit ensuite door (from RULES: master_bedroom.ensuite.door_explicit=True)
    #    Identified by horizontal touch (below/above master), not side touch
    for r in rooms:
        if r["name"] != "master_bedroom": continue
        for other in rooms:
            if other["name"] != "bathroom": continue
            if r2f.get(r["idx"]) != r2f.get(other["idx"]): continue
            if not _touch(r, other): continue
            horiz = (abs((r["y"]+r["h"]) - other["y"]) <= 2 or
                     abs((other["y"]+other["h"]) - r["y"]) <= 2)
            if not horiz: continue
            if other["idx"] in max_door_given: continue
            rect, horiz, swing = _opening_rect_between(r, other, dw)
            if rect is not None and _claim_opening(rect, "door"):
                draw_door_with_arc(fd, bd, md, lbl, rect, horiz, dc, dl, swing)
                _record_door(r, other)
                _record_edge(r, other, "door", rect)
                max_door_given.add(other["idx"])
            break

    # ── Only these inside-flat pairs get a plain empty opening: living↔kitchen and living↔corridor.
    plain_opening_pairs = {
        frozenset(["living_space", "kitchen"]),
        frozenset(["living_space", "corridor"]),
    }
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i+1:]:
            pair = frozenset([r1["name"], r2["name"]])
            if pair not in plain_opening_pairs:
                continue
            if r2f.get(r1["idx"]) != r2f.get(r2["idx"]):
                continue
            if not _touch(r1, r2):
                continue
            rect, _, _ = _opening_rect_between(r1, r2, dw)
            if rect is not None and _claim_opening(rect, "gap"):
                draw_plain_opening(fd, bd, md, lbl, rect)
                has_door.add(r1["idx"]); has_door.add(r2["idx"])
                _add_passage(r1, r2)
                _record_edge(r1, r2, "opening", rect)

    # ── General interior doors (from INTERIOR_DOOR + NO_DOOR derived from RULES)
    for i,r1 in enumerate(rooms):
        if r1["name"] in ("corridor","lift","stair","balcony"): continue
        for r2 in rooms[i+1:]:
            if r2["name"] in ("corridor","lift","stair"): continue
            if r2f.get(r1["idx"]) != r2f.get(r2["idx"]): continue
            if not _touch(r1,r2): continue
            pair = frozenset([r1["name"],r2["name"]])
            if pair in plain_opening_pairs: continue
            if pair in NO_DOOR: continue
            if pair in BALCONY_CONNECTS:
                do_door(r1, r2, sdw)
            elif pair in INTERIOR_DOOR:
                do_door(r1, r2)

    # ── Front door / front gate
    # Cluster: one front door for every flat whose hall/living touches the building corridor.
    # Single flat: place the front door on a living/hall piece that touches the outside boundary.
    def _draw_front_door_between(corr, liv):
        rect, horiz, swing = _opening_rect_between(corr, liv, dw)
        if rect is None or not _claim_opening(rect, "front_door"):
            return False
        draw_door_with_arc(fd, bd, md, lbl, rect, horiz, fdc, fdl, swing)
        has_door.add(liv["idx"])
        _add_passage(corr, liv)
        _record_edge(corr, liv, "front_door", rect)
        return True

    def _draw_front_door_to_outside(liv):
        bounds = (PAD, PAD, PAD + f2p(tw_ft), PAD + f2p(th_ft))
        sides = list(_apt_boundary_sides(liv, *bounds))
        sides = _filter_exterior_sides(liv, sides, rooms)
        if not sides:
            return False
        # Prefer a side with enough length; use left/right first for the usual plan style.
        for side in ("left", "right", "bottom", "top"):
            if side not in sides:
                continue
            x,y,w,h = liv["x"],liv["y"],liv["w"],liv["h"]
            if side in ("left", "right"):
                length = h
                if length < dw + 2 * WALL_T:
                    continue
                my = y + h // 2
                xx = x if side == "left" else x + w
                rect = [xx - WALL_T//2, my-dw//2, xx + WALL_T//2, my+dw//2]
                horiz = False
                swing = 1 if side == "left" else -1
            else:
                length = w
                if length < dw + 2 * WALL_T:
                    continue
                mx = x + w // 2
                yy = y if side == "top" else y + h
                rect = [mx-dw//2, yy-WALL_T//2, mx+dw//2, yy+WALL_T//2]
                horiz = True
                swing = 1 if side == "top" else -1
            if not _claim_opening(rect, "front_door"):
                continue
            draw_door_with_arc(fd, bd, md, lbl, rect, horiz, fdc, fdl, swing)
            has_door.add(liv["idx"])
            _record_edge(liv, "outside", "front_door", rect)
            return True
        return False

    if is_cluster:
        corridor_rooms = [r for r in rooms if r["name"] == "corridor"]
        flats = sorted(set(r2f.values()))
        for fid in flats:
            living_rooms = [r for r in rooms if r["name"] == "living_space" and r2f.get(r["idx"]) == fid]
            if not living_rooms:
                continue
            made = False
            # Prefer the living/hall piece that touches the building corridor.
            for corr in corridor_rooms:
                for liv in living_rooms:
                    if _touch(corr, liv) and _draw_front_door_between(corr, liv):
                        made = True
                        break
                if made:
                    break
            # Safety fallback: if no hall piece touches the building corridor,
            # still add a visible front gate to the flat at the room that touches
            # the corridor. This prevents missing front doors in horizontal/random
            # cluster layouts where the generated living room is not on the corridor edge.
            if not made:
                fallback_rooms = sorted(
                    [r for r in rooms
                     if r["name"] not in ("corridor", "lift", "stair", "balcony")
                     and r2f.get(r["idx"]) == fid],
                    key=lambda r: {"living_space": 0, "kitchen": 1, "bedroom": 2,
                                   "master_bedroom": 3, "bathroom": 4}.get(r["name"], 9)
                )
                for corr in corridor_rooms:
                    for room in fallback_rooms:
                        if _touch(corr, room) and _draw_front_door_between(corr, room):
                            made = True
                            break
                    if made:
                        break
    else:
        living_rooms = sorted([r for r in rooms if r["name"] == "living_space"],
                              key=lambda r: r["w"] * r["h"], reverse=True)
        for liv in living_rooms:
            if _draw_front_door_to_outside(liv):
                break

    # ── Lift doors: lifts are attached to the OUTER corridor wall.
    # Draw each lift door on the opposite/inner side so apartment front doors stay clear.
    for lift_r in [r for r in rooms if r["name"] == "lift"]:
        corr_for_lift = next((rr for rr in rooms if rr["name"] == "corridor" and _touch(rr, lift_r)), None)
        if corr_for_lift is not None:
            rect, horiz, swing = _opening_rect_between(lift_r, corr_for_lift, dw)
        else:
            corr_for_lift = next((rr for rr in rooms if rr["name"] == "corridor" and _contains_room(rr, lift_r, margin=2)), None)
            if corr_for_lift is None:
                continue

            wdoor = min(dw, max(10, lift_r["w"] - 2 * WALL_T))
            mx = lift_r["x"] + lift_r["w"] // 2

            # Horizontal corridor: if the lift touches the corridor top, door faces down;
            # if it touches the corridor bottom, door faces up.
            # Vertical corridor uses the same top/bottom end rule.
            if abs(lift_r["y"] - corr_for_lift["y"]) <= WALL_T:
                yy = lift_r["y"] + lift_r["h"]
                swing = 1
            elif abs((lift_r["y"] + lift_r["h"]) - (corr_for_lift["y"] + corr_for_lift["h"])) <= WALL_T:
                yy = lift_r["y"]
                swing = -1
            else:
                # Fallback: choose the side that points toward the larger open corridor area.
                corr_mid_y = corr_for_lift["y"] + corr_for_lift["h"] / 2
                if lift_r["y"] + lift_r["h"] / 2 <= corr_mid_y:
                    yy = lift_r["y"] + lift_r["h"]
                    swing = 1
                else:
                    yy = lift_r["y"]
                    swing = -1
            rect = [mx - wdoor//2, yy - WALL_T//2, mx + wdoor//2, yy + WALL_T//2]
            horiz = True

        if rect is None or not _claim_opening(rect, "lift_door"):
            continue
        draw_door_with_arc(fd, bd, md, lbl, rect, horiz, dc, dl, swing)
        has_door.add(lift_r["idx"])
        _record_edge(lift_r, corr_for_lift, "door", rect)

    # ── Guarantee every bathroom has exactly one door and every bedroom has an interior door.
    def _bedroom_has_interior_door(room):
        return any(p != "balcony" for p in door_partners.get(room["idx"], []))

    def ensure_single_room_door(target):
        if target["name"] == "bathroom" and door_count[target["idx"]] >= 1:
            return
        if target["name"] == "bedroom" and _bedroom_has_interior_door(target):
            return
        if target["name"] not in ("bathroom", "bedroom"):
            return

        preferred = {
            "bathroom": ["master_bedroom", "bedroom", "living_space"],
            "bedroom":  ["corridor", "living_space", "bedroom", "bathroom"],
        }[target["name"]]

        candidates = []
        for other in rooms:
            if other["idx"] == target["idx"]:
                continue
            if r2f.get(target["idx"]) != r2f.get(other["idx"]):
                continue
            if not _touch(target, other):
                continue
            pair = frozenset([target["name"], other["name"]])
            if pair in plain_opening_pairs:
                continue
            if pair in NO_DOOR:
                continue
            try:
                rank = preferred.index(other["name"])
            except ValueError:
                continue
            candidates.append((rank, other))

        for _, other in sorted(candidates, key=lambda x: x[0]):
            if do_door(target, other):
                return

        # Handles bathrooms drawn inside master/bedroom/living rectangles.
        if target["name"] == "bathroom" and door_count[target["idx"]] < 1:
            containers = [
                other for other in rooms
                if other["name"] in ("master_bedroom", "bedroom", "living_space")
                and r2f.get(target["idx"]) == r2f.get(other["idx"])
                and _contains_room(other, target)
            ]
            if containers:
                container = sorted(containers, key=lambda r: {"master_bedroom": 0, "bedroom": 1, "living_space": 2}.get(r["name"], 9))[0]
                w = min(dw, max(10, target["w"] - 2 * WALL_T))
                mid = target["x"] + target["w"] // 2
                rect = [mid - w//2, target["y"] - WALL_T//2, mid + w//2, target["y"] + WALL_T//2]
                if _claim_opening(rect, "door"):
                    draw_door_with_arc(fd, bd, md, lbl, rect, True, dc, dl, -1)
                    _record_door(target, container)
                    _record_edge(target, container, "door", rect)
                    max_door_given.add(target["idx"])

    for target in rooms:
        if target["name"] in ("bathroom", "bedroom"):
            ensure_single_room_door(target)

    # ── Guarantee balcony is visually attached, preferring bedroom/master/corridor.
    for bal in [r for r in rooms if r["name"] == "balcony"]:
        attached = False
        candidates = []
        for other in rooms:
            if other["idx"] == bal["idx"]:
                continue
            if r2f.get(bal["idx"]) != r2f.get(other["idx"]):
                continue
            if not _touch(bal, other):
                continue
            pref = {"master_bedroom": 0, "bedroom": 1, "corridor": 2, "living_space": 3}.get(other["name"])
            if pref is not None:
                candidates.append((pref, other))

        for _, other in sorted(candidates, key=lambda x: x[0]):
            rect, horiz, swing = _opening_rect_between(bal, other, sdw if other["name"] != "corridor" else dw)
            if rect is None:
                continue
            if do_door(bal, other, sdw if other["name"] != "corridor" else dw, force=True):
                attached = True
            if attached:
                has_door.add(bal["idx"]); has_door.add(other["idx"])
                break

    # ── Fallback: ensure every non-structural room has at least one door
    for r in rooms:
        if r["name"] in ("corridor","background","apartment","lift","stair"): continue
        if r["name"] in ("bathroom", "bedroom") and door_count[r["idx"]] >= 1: continue
        if r["idx"] in has_door: continue
        for other in rooms:
            if other["idx"] == r["idx"]: continue
            if not _touch(r, other): continue
            pair = frozenset([r["name"],other["name"]])
            if pair in plain_opening_pairs: continue
            if pair in NO_DOOR: continue
            if do_door(r, other): break

    # ── Final pass after balcony/fallback: no bathroom or bedroom should be left without a real door.
    for target in rooms:
        if target["name"] in ("bathroom", "bedroom"):
            ensure_single_room_door(target)

    # ── Bedrooms must be able to reach living space or corridor, directly or through connected rooms.
    idx_to_room = {r["idx"]: r for r in rooms}

    def _can_reach_public(start_idx):
        seen = set()
        stack = [start_idx]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            cur_room = idx_to_room.get(cur)
            if cur_room and cur_room["name"] in ("living_space", "corridor"):
                return True
            for nxt in passage_graph.get(cur, set()):
                if nxt not in seen:
                    stack.append(nxt)
        return False

    for room in rooms:
        if room["name"] not in ("bedroom", "master_bedroom"):
            continue
        if _can_reach_public(room["idx"]):
            continue
        candidates = []
        for other in rooms:
            if other["idx"] == room["idx"]:
                continue
            if r2f.get(room["idx"]) != r2f.get(other["idx"]):
                continue
            if not _touch(room, other):
                continue
            pair = frozenset([room["name"], other["name"]])
            if pair in NO_DOOR or pair in plain_opening_pairs:
                continue
            reachable_other = _can_reach_public(other["idx"]) or other["name"] in ("living_space", "corridor", "kitchen")
            if not reachable_other:
                continue
            pref = {"corridor": 0, "living_space": 1, "kitchen": 2, "bedroom": 3, "master_bedroom": 4, "bathroom": 5}.get(other["name"], 9)
            candidates.append((pref, other))
        for _, other in sorted(candidates, key=lambda x: x[0]):
            if do_door(room, other, force=True):
                break

    # ── Windows (from RULES: global_window_rules.at_flat_boundary=True)
    #    Width is proportional to the wall (45-70%), position randomised along wall.
    #    Large rooms with 2+ exterior sides may get a second window.
    wic, wil = LCOL[LID["window"]], LID["window"]
    _WIN_GAP = 8   # minimum clear gap between window edge and room corner (px)

    for r in rooms:
        if r["name"] not in WIN_ROOMS: continue
        if random.random() > WIN_PROB.get(r["name"], 1.0): continue
        bounds = apt_bounds.get(r["idx"])
        if not bounds: continue
        sides = list(_apt_boundary_sides(r, *bounds))
        if not sides: continue
        sides = _filter_exterior_sides(r, sides, rooms)
        if not sides: continue

        x, y, w, h = r["x"], r["y"], r["w"], r["h"]

        # Large rooms (>~100 sq ft at scale) may get windows on 2 exterior sides
        room_area_px2 = w * h
        large_room = room_area_px2 >= (100 * SCALE * SCALE)
        n_win_sides = 2 if (large_room and len(sides) >= 2 and random.random() < 0.40) else 1
        win_sides = random.sample(sides, min(n_win_sides, len(sides)))

        for side in win_sides:
            wall_len = w if side in ("top", "bottom") else h
            usable = max(1, wall_len - 2 * _WIN_GAP)

            # Window spans 45-72% of the usable wall, minimum MIN_WINDOW_LEN_PX
            win_frac = random.uniform(0.45, 0.72)
            win_len  = max(MIN_WINDOW_LEN_PX, int(usable * win_frac))
            win_len  = min(win_len, usable)

            # Randomise position along wall (not always centred)
            slack = max(0, usable - win_len)
            offset = random.randint(0, slack) if slack > 0 else 0

            half = WALL_T // 2
            rect = None
            if side == "top":
                s = x + _WIN_GAP + offset
                rect = [s, y - half, s + win_len, y + WALL_T - half]
            elif side == "bottom":
                s = x + _WIN_GAP + offset
                rect = [s, y + h - half, s + win_len, y + h + WALL_T - half]
            elif side == "left":
                s = y + _WIN_GAP + offset
                rect = [x - half, s, x + WALL_T - half, s + win_len]
            elif side == "right":
                s = y + _WIN_GAP + offset
                rect = [x + w - half, s, x + w + WALL_T - half, s + win_len]

            if rect is not None:
                # Avoid collisions with any existing door/front_door/opening on the same wall.
                # Try alternative positions first; skip the window if no safe slot exists.
                if not _window_is_clear(rect):
                    safe_rect = None
                    max_tries = 12
                    for _try in range(max_tries):
                        # progressively shrink a little if needed, then retry random offsets
                        trial_len = max(MIN_WINDOW_LEN_PX, int(win_len * (0.92 ** (_try // 4))))
                        trial_len = min(trial_len, usable)
                        trial_slack = max(0, usable - trial_len)
                        trial_offset = random.randint(0, trial_slack) if trial_slack > 0 else 0
                        if side == "top":
                            s = x + _WIN_GAP + trial_offset
                            trial = [s, y - half, s + trial_len, y + WALL_T - half]
                        elif side == "bottom":
                            s = x + _WIN_GAP + trial_offset
                            trial = [s, y + h - half, s + trial_len, y + h + WALL_T - half]
                        elif side == "left":
                            s = y + _WIN_GAP + trial_offset
                            trial = [x - half, s, x + WALL_T - half, s + trial_len]
                        else:  # right
                            s = y + _WIN_GAP + trial_offset
                            trial = [x + w - half, s, x + w + WALL_T - half, s + trial_len]
                        if _window_is_clear(trial):
                            safe_rect = trial
                            break
                    rect = safe_rect

                if rect is not None and _window_is_clear(rect):
                    draw_window_gap(fd, bd, md, lbl, rect, wic, wil)
                    _record_edge(r, "outside", "window", rect)

    if bw_with_dimensions:
        draw_room_dimensions(bd, rooms, color=(170,170,170))

    counts, seen = {}, {}
    for r in rooms: counts[r["name"]] = counts.get(r["name"],0)+1
    for r in rooms:
        seen[r["name"]] = seen.get(r["name"],0)+1
        cx,cy = r["x"]+r["w"]//2, r["y"]+r["h"]//2
        txt = r["name"].replace("_"," ").upper()
        # Keep hall labels generic: show "LIVING SPACE" without numeric suffixes.
        if counts[r["name"]]>1 and r["name"] != "living_space":
            txt += f" {seen[r['name']]}"
        try:
            fd.text((cx,cy),txt,fill=(20,20,20),anchor="mm")
            if bw_with_text:
                bd.text((cx,cy),txt,fill=(0,0,0),anchor="mm")
        except:
            fd.text((cx-len(txt)*3,cy-5),txt,fill=(20,20,20))
            if bw_with_text:
                bd.text((cx-len(txt)*3,cy-5),txt,fill=(0,0,0))

    render_plan.last_graph = {
        "nodes": _build_graph_nodes(rooms),
        "edges": graph_edges,
    }
    return fp, bw, mk, lbl, rooms


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    fpd  = os.path.join(BASE_DIR,"floor_plans")
    segd = os.path.join(BASE_DIR,"seg_masks")
    metd = os.path.join(BASE_DIR,"metadata")
    for d in (fpd,segd,metd): os.makedirs(d,exist_ok=True)

    lmap = {str(k):{"name":v[0],"color":list(v[1])} for k,v in LABELS.items()}
    with open(os.path.join(metd,"label_map.json"),"w") as f: json.dump(lmap,f,indent=2)
    print(f"Label map → {metd}/label_map.json\n")

    for i in range(NUM_SAMPLES):
        apt = random.choice(APARTMENT_TYPES); idx = f"{i+1:04d}"
        fp_img,bw_img,mk_img,lbl_arr,rooms = render_plan(apt)
        v = getattr(render_plan,"last_val",{"valid":True,"errors":[],"warnings":[]})
        fp_img.save(os.path.join(fpd, f"image_{idx}.png"))
        bw_img.save(os.path.join(fpd, f"image_{idx}_bw.png"))
        mk_img.save(os.path.join(segd,f"mask_{idx}.png"))
        np.save(    os.path.join(segd,f"mask_{idx}.npy"), lbl_arr)
        unique = sorted(int(x) for x in np.unique(lbl_arr))
        graph = getattr(render_plan, "last_graph", {"nodes": [], "edges": []})
        meta = {
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "sample_id":idx, "apartment_type":apt,
            "image_size_wh":list(fp_img.size), "scale_px_per_ft":SCALE,
            "files":{
                "floor_plan":    f"floor_plans/image_{idx}.png",
                "floor_plan_bw": f"floor_plans/image_{idx}_bw.png",
                "seg_mask_png":  f"seg_masks/mask_{idx}.png",
                "seg_mask_npy":  f"seg_masks/mask_{idx}.npy",
            },
            "unique_label_ids":unique,
            "unique_classes":[LABELS[l][0] for l in unique],
            "rule_validation":v, "label_map":lmap,
        }
        with open(os.path.join(metd,f"meta_{idx}.json"),"w") as f: json.dump(meta,f,indent=2)
        print(f"  [{i+1:>4}/{NUM_SAMPLES}]  {'✓' if v['valid'] else '✗'}  {idx}  {apt:<22}  ({fp_img.size[0]}×{fp_img.size[1]}px)")

    print(f"\n✓  {NUM_SAMPLES} samples → {BASE_DIR}/")

if __name__ == "__main__":
    main()