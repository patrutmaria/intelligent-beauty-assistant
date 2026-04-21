"""Color family definitions and recommendation engine for makeup products."""

from __future__ import annotations

FAMILIES: dict[str, dict] = {
    "soft_pink":  {"label": "Soft Pink",   "emoji": "🌸", "hex": "#f4b8c8", "hue": (335, 360),  "sat_min": 0.20, "sat_max": 0.50, "v_min": 0.65},
    "baby_pink":  {"label": "Baby Pink",   "emoji": "🩷", "hex": "#fcc6d4", "hue": (340, 360),  "sat_min": 0.10, "sat_max": 0.30, "v_min": 0.85},
    "hot_pink":   {"label": "Hot Pink",    "emoji": "💖", "hex": "#e84a90", "hue": (320, 350),  "sat_min": 0.55, "sat_max": 1.00, "v_min": 0.55},
    "rose":       {"label": "Rose",        "emoji": "🌹", "hex": "#cf5e6e", "hue": (345, 360),  "sat_min": 0.40, "sat_max": 0.80, "v_min": 0.45},
    "mauve":      {"label": "Mauve",       "emoji": "💜", "hex": "#a16b86", "hue": (300, 340),  "sat_min": 0.20, "sat_max": 0.55, "v_min": 0.35},
    "berry":      {"label": "Berry",       "emoji": "🍒", "hex": "#7d2842", "hue": (340, 360),  "sat_min": 0.50, "sat_max": 1.00, "v_min": 0.20},
    "plum":       {"label": "Plum",        "emoji": "🍇", "hex": "#5e2545", "hue": (290, 340),  "sat_min": 0.40, "sat_max": 1.00, "v_min": 0.15},
    "wine":       {"label": "Wine",        "emoji": "🍷", "hex": "#5c1a2a", "hue": (340, 360),  "sat_min": 0.55, "sat_max": 1.00, "v_min": 0.15},
    "peach":      {"label": "Peach",       "emoji": "🍑", "hex": "#f4a682", "hue": (15, 35),    "sat_min": 0.30, "sat_max": 0.65, "v_min": 0.65},
    "apricot":    {"label": "Apricot",     "emoji": "🍊", "hex": "#f3a368", "hue": (20, 40),    "sat_min": 0.45, "sat_max": 0.75, "v_min": 0.65},
    "coral":      {"label": "Coral",       "emoji": "🪸", "hex": "#ec6a55", "hue": (5, 20),     "sat_min": 0.55, "sat_max": 0.90, "v_min": 0.55},
    "warm_rose":  {"label": "Warm Rose",   "emoji": "🌷", "hex": "#cc6080", "hue": (340, 360),  "sat_min": 0.40, "sat_max": 0.65, "v_min": 0.40},
    "true_red":   {"label": "True Red",    "emoji": "❤️", "hex": "#c8222e", "hue": (350, 360),  "sat_min": 0.65, "sat_max": 1.00, "v_min": 0.35},
    "warm_red":   {"label": "Warm Red",    "emoji": "🔥", "hex": "#c8312c", "hue": (0, 12),     "sat_min": 0.65, "sat_max": 1.00, "v_min": 0.35},
    "cool_red":   {"label": "Cool Red",    "emoji": "❄️", "hex": "#b21942", "hue": (340, 355),  "sat_min": 0.60, "sat_max": 1.00, "v_min": 0.30},
    "nude":       {"label": "Nude",        "emoji": "🤎", "hex": "#c19078", "hue": (15, 30),    "sat_min": 0.20, "sat_max": 0.45, "v_min": 0.55},
    "warm_brown": {"label": "Warm Brown",  "emoji": "🌰", "hex": "#8b5a3c", "hue": (15, 35),    "sat_min": 0.30, "sat_max": 0.70, "v_min": 0.25},
    "cool_brown": {"label": "Cool Brown",  "emoji": "🪵", "hex": "#6b4838", "hue": (15, 30),    "sat_min": 0.20, "sat_max": 0.50, "v_min": 0.20},
    "terracotta": {"label": "Terracotta",  "emoji": "🏺", "hex": "#b25638", "hue": (10, 25),    "sat_min": 0.50, "sat_max": 0.85, "v_min": 0.40},
    "brick":      {"label": "Brick",       "emoji": "🧱", "hex": "#923524", "hue": (5, 20),     "sat_min": 0.60, "sat_max": 0.95, "v_min": 0.30},
    "bronze":     {"label": "Bronze",      "emoji": "🥉", "hex": "#a86836", "hue": (20, 40),    "sat_min": 0.55, "sat_max": 0.90, "v_min": 0.35},
    "copper":     {"label": "Copper",      "emoji": "🟠", "hex": "#b6612b", "hue": (15, 30),    "sat_min": 0.55, "sat_max": 0.95, "v_min": 0.40},
    "champagne":  {"label": "Champagne",   "emoji": "🥂", "hex": "#f0d9a4", "hue": (35, 50),    "sat_min": 0.20, "sat_max": 0.50, "v_min": 0.85},
    "gold":       {"label": "Gold",        "emoji": "✨", "hex": "#e6c150", "hue": (40, 55),    "sat_min": 0.45, "sat_max": 0.85, "v_min": 0.75},
    "rose_gold":  {"label": "Rose Gold",   "emoji": "💗", "hex": "#dfa48a", "hue": (10, 25),    "sat_min": 0.30, "sat_max": 0.55, "v_min": 0.75},
    "pearl":      {"label": "Pearl",       "emoji": "🤍", "hex": "#f2e8da", "hue": (30, 50),    "sat_min": 0.05, "sat_max": 0.20, "v_min": 0.90},
    "bronze_glow":{"label": "Bronze Glow", "emoji": "🌅", "hex": "#c5895a", "hue": (20, 40),    "sat_min": 0.45, "sat_max": 0.75, "v_min": 0.60},
    "taupe":      {"label": "Taupe",       "emoji": "🍂", "hex": "#8c7565", "hue": (20, 40),    "sat_min": 0.10, "sat_max": 0.35, "v_min": 0.40},
    "smoky":      {"label": "Smoky",       "emoji": "🌑", "hex": "#3c3438", "hue": (0, 360),    "sat_min": 0.00, "sat_max": 0.30, "v_min": 0.10},
    "purple":     {"label": "Purple",      "emoji": "🟣", "hex": "#6b3982", "hue": (260, 295),  "sat_min": 0.30, "sat_max": 1.00, "v_min": 0.20},
    "burgundy":   {"label": "Burgundy",    "emoji": "🍷", "hex": "#5c1a2a", "hue": (340, 360),  "sat_min": 0.55, "sat_max": 1.00, "v_min": 0.20},
}

# Best families per category x skin profile (lightness, undertone)
BEST: dict[str, dict[tuple[str, str], list[str]]] = {
    "blush": {
        ("light",       "cool"):    ["baby_pink", "soft_pink", "mauve", "rose"],
        ("light",       "warm"):    ["peach", "apricot", "soft_pink", "coral"],
        ("light",       "neutral"): ["soft_pink", "peach", "rose", "mauve"],
        ("light-medium","cool"):    ["soft_pink", "rose", "mauve", "berry"],
        ("light-medium","warm"):    ["peach", "coral", "warm_rose", "apricot"],
        ("light-medium","neutral"): ["rose", "peach", "soft_pink", "warm_rose"],
        ("medium",      "cool"):    ["rose", "mauve", "berry", "plum"],
        ("medium",      "warm"):    ["coral", "warm_rose", "terracotta", "apricot"],
        ("medium",      "neutral"): ["rose", "coral", "warm_rose", "berry"],
        ("medium-deep", "cool"):    ["berry", "plum", "wine", "mauve"],
        ("medium-deep", "warm"):    ["terracotta", "brick", "bronze", "warm_rose"],
        ("medium-deep", "neutral"): ["berry", "terracotta", "rose", "plum"],
        ("deep",        "cool"):    ["plum", "wine", "berry"],
        ("deep",        "warm"):    ["brick", "bronze", "terracotta", "copper"],
        ("deep",        "neutral"): ["plum", "berry", "brick", "bronze"],
    },
    "lipstick": {
        ("light",       "cool"):    ["baby_pink", "soft_pink", "mauve", "rose", "cool_red"],
        ("light",       "warm"):    ["peach", "coral", "nude", "warm_red", "apricot"],
        ("light",       "neutral"): ["nude", "soft_pink", "rose", "peach", "warm_rose"],
        ("light-medium","cool"):    ["mauve", "rose", "berry", "cool_red", "soft_pink"],
        ("light-medium","warm"):    ["coral", "peach", "warm_red", "nude", "warm_brown"],
        ("light-medium","neutral"): ["nude", "rose", "warm_rose", "warm_brown", "true_red"],
        ("medium",      "cool"):    ["berry", "mauve", "plum", "cool_red", "rose"],
        ("medium",      "warm"):    ["warm_brown", "terracotta", "warm_red", "coral", "brick"],
        ("medium",      "neutral"): ["nude", "warm_rose", "berry", "true_red", "rose"],
        ("medium-deep", "cool"):    ["plum", "wine", "berry", "cool_red"],
        ("medium-deep", "warm"):    ["brick", "terracotta", "bronze", "warm_brown", "warm_red"],
        ("medium-deep", "neutral"): ["berry", "plum", "warm_brown", "wine"],
        ("deep",        "cool"):    ["plum", "wine", "berry", "cool_brown"],
        ("deep",        "warm"):    ["brick", "bronze", "copper", "warm_brown", "terracotta"],
        ("deep",        "neutral"): ["plum", "wine", "warm_brown", "cool_brown"],
    },
    "highlighter": {
        ("light",       "cool"):    ["pearl", "champagne", "rose_gold"],
        ("light",       "warm"):    ["champagne", "gold", "rose_gold", "pearl"],
        ("light",       "neutral"): ["pearl", "champagne", "rose_gold"],
        ("light-medium","cool"):    ["pearl", "rose_gold", "champagne"],
        ("light-medium","warm"):    ["champagne", "gold", "rose_gold", "bronze_glow"],
        ("light-medium","neutral"): ["champagne", "rose_gold", "gold", "pearl"],
        ("medium",      "cool"):    ["rose_gold", "champagne", "pearl"],
        ("medium",      "warm"):    ["gold", "bronze_glow", "champagne", "rose_gold"],
        ("medium",      "neutral"): ["champagne", "gold", "rose_gold", "bronze_glow"],
        ("medium-deep", "cool"):    ["rose_gold", "champagne"],
        ("medium-deep", "warm"):    ["bronze_glow", "gold", "copper"],
        ("medium-deep", "neutral"): ["bronze_glow", "gold", "rose_gold"],
        ("deep",        "cool"):    ["rose_gold", "bronze_glow"],
        ("deep",        "warm"):    ["bronze_glow", "copper", "gold", "bronze"],
        ("deep",        "neutral"): ["bronze_glow", "copper", "rose_gold", "gold"],
    },
    "contour": {
        ("light",       "cool"):    ["taupe", "cool_brown"],
        ("light",       "warm"):    ["taupe", "warm_brown"],
        ("light",       "neutral"): ["taupe", "warm_brown"],
        ("light-medium","cool"):    ["cool_brown", "taupe"],
        ("light-medium","warm"):    ["warm_brown", "terracotta"],
        ("light-medium","neutral"): ["warm_brown", "taupe", "cool_brown"],
        ("medium",      "cool"):    ["cool_brown", "taupe"],
        ("medium",      "warm"):    ["warm_brown", "terracotta", "bronze"],
        ("medium",      "neutral"): ["warm_brown", "cool_brown", "bronze"],
        ("medium-deep", "cool"):    ["cool_brown"],
        ("medium-deep", "warm"):    ["warm_brown", "bronze", "brick"],
        ("medium-deep", "neutral"): ["warm_brown", "bronze"],
        ("deep",        "cool"):    ["cool_brown"],
        ("deep",        "warm"):    ["warm_brown", "bronze", "brick", "copper"],
        ("deep",        "neutral"): ["cool_brown", "warm_brown", "bronze"],
    },
    "eyeshadow": {
        ("light",       "cool"):    ["taupe", "mauve", "soft_pink", "smoky"],
        ("light",       "warm"):    ["peach", "champagne", "gold", "warm_brown"],
        ("light",       "neutral"): ["taupe", "champagne", "rose_gold", "warm_brown"],
        ("light-medium","cool"):    ["mauve", "taupe", "smoky", "purple"],
        ("light-medium","warm"):    ["bronze", "gold", "warm_brown", "copper"],
        ("light-medium","neutral"): ["taupe", "warm_brown", "rose_gold", "champagne"],
        ("medium",      "cool"):    ["smoky", "mauve", "purple", "burgundy"],
        ("medium",      "warm"):    ["bronze", "copper", "warm_brown", "terracotta"],
        ("medium",      "neutral"): ["taupe", "warm_brown", "smoky", "bronze"],
        ("medium-deep", "cool"):    ["smoky", "burgundy", "purple"],
        ("medium-deep", "warm"):    ["bronze", "copper", "brick", "gold"],
        ("medium-deep", "neutral"): ["smoky", "warm_brown", "bronze"],
        ("deep",        "cool"):    ["smoky", "purple", "burgundy"],
        ("deep",        "warm"):    ["copper", "bronze", "gold", "brick"],
        ("deep",        "neutral"): ["smoky", "bronze", "copper"],
    },
}


def recommend_families(category: str,
                       lightness: str | None,
                       undertone: str | None) -> list[dict]:
    """Return recommended family dicts for the user's skin profile."""
    cat = (category or "").lower().strip()
    if cat not in BEST:
        return []
    lt = (lightness or "medium").lower().strip()
    ut = (undertone or "neutral").lower().strip()
    if lt not in {"light","light-medium","medium","medium-deep","deep"}:
        lt = "medium"
    if ut not in {"warm","cool","neutral"}:
        ut = "neutral"
    keys = BEST[cat].get((lt, ut), [])
    return [{"key": k, **FAMILIES[k]} for k in keys if k in FAMILIES]


def classify_color(hex_str: str, category: str = "") -> str:
    """
    Map a product hex to one of the families. Returns the family key or "".
    Tries each family's hue/sat/v ranges; for ties, picks closest to family's representative hex.
    """
    if not hex_str or not hex_str.startswith("#") or len(hex_str) != 7:
        return ""
    try:
        r = int(hex_str[1:3], 16)
        g = int(hex_str[3:5], 16)
        b = int(hex_str[5:7], 16)
    except ValueError:
        return ""

    h, s, v = _rgb_to_hsv(r, g, b)

    matches = []
    for key, fam in FAMILIES.items():
        h_lo, h_hi = fam["hue"]
        if h_lo <= h_hi:
            in_hue = h_lo <= h <= h_hi
        else:
            in_hue = h >= h_lo or h <= h_hi
        in_sat = fam["sat_min"] <= s <= fam.get("sat_max", 1.0)
        in_v   = v >= fam.get("v_min", 0.0)
        if in_hue and in_sat and in_v:
            dist = _hex_distance(hex_str, fam["hex"])
            matches.append((dist, key))

    if matches:
        matches.sort()
        return matches[0][1]

    nearest = min(FAMILIES.items(),
                  key=lambda kv: _hex_distance(hex_str, kv[1]["hex"]))
    return nearest[0]


def family_label(key: str) -> str:
    return FAMILIES.get(key, {}).get("label", key)


def family_score(category: str, hex_str: str,
                 user_undertone: str | None,
                 user_lightness: str | None) -> tuple[float, str]:
    """
    Returns (score 0..1, family_key) for how well a product's color suits
    the user. 1.0 = product family is the top recommendation.
    """
    fam = classify_color(hex_str, category)
    if not fam:
        return 0.0, ""
    rec = recommend_families(category, user_lightness, user_undertone)
    rec_keys = [r["key"] for r in rec]
    if fam in rec_keys:
        idx = rec_keys.index(fam)
        return 1.0 - idx * 0.1, fam
    return 0.3, fam


def _rgb_to_hsv(r, g, b):
    rn, gn, bn = r/255, g/255, b/255
    cmax = max(rn, gn, bn); cmin = min(rn, gn, bn)
    delta = cmax - cmin
    if delta == 0: h = 0
    elif cmax == rn: h = 60 * (((gn - bn)/delta) % 6)
    elif cmax == gn: h = 60 * (((bn - rn)/delta) + 2)
    else:            h = 60 * (((rn - gn)/delta) + 4)
    s = 0 if cmax == 0 else delta/cmax
    return h, s, cmax


def _hex_distance(a: str, b: str) -> float:
    a = a.lstrip("#"); b = b.lstrip("#")
    ra, ga, ba = int(a[0:2],16), int(a[2:4],16), int(a[4:6],16)
    rb, gb, bb = int(b[0:2],16), int(b[2:4],16), int(b[4:6],16)
    return ((ra-rb)**2 + (ga-gb)**2 + (ba-bb)**2) ** 0.5
