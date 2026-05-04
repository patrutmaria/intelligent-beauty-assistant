"""Matches detected skin colour to foundation shades using CIEDE2000 perceptual distance."""

import os
import json
import re
import math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHADES_PATH = os.path.join(_ROOT, "data", "shades.json")

_SHADES: list[dict] = []
_SHADES_BY_BRAND: dict[str, list[dict]] = {}
_MIN_BRAND_SHADES = 15

_BRAND_ALIASES = {
    "fenty beauty":            "fenty beauty by rihanna",
    "fenty beauty by rihanna": "fenty beauty",
    "loreal":                  "l'oreal",
    "l'oreal":                 "l'oreal",
    "estee lauder":            "estee lauder",
    "make up for ever":        "make up for ever",
    "mufe":                    "make up for ever",
    "tarte cosmetics":         "tarte",
    "anastasia":               "anastasia beverly hills",
    "anastasia beverly hills": "anastasia",
}


#  CIEDE2000 implementation 

def _rgb_to_lab(r, g, b):
    """Convert sRGB (0-255) to CIELAB (L*, a*, b*)."""
    # Linearize sRGB
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = lin(r), lin(g), lin(b)

    # Linear RGB -> XYZ (D65)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # D65 reference white
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return L, a, b_val


def _ciede2000(lab1, lab2):
    """
    CIEDE2000 color difference. Returns a perceptual distance where:
    - < 1.0: imperceptible
    - 1-2: barely perceptible
    - 2-5: noticeable at close inspection
    - 5-10: clearly different
    - > 10: very different colors
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    avg_C7 = avg_C**7
    G = 0.5 * (1 - math.sqrt(avg_C7 / (avg_C7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp += 360 if dhp < 0 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    if abs(h1p - h2p) <= 180:
        avg_Hp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        avg_Hp = (h1p + h2p + 360) / 2.0
    else:
        avg_Hp = (h1p + h2p - 360) / 2.0

    T = (1
         - 0.17 * math.cos(math.radians(avg_Hp - 30))
         + 0.24 * math.cos(math.radians(2 * avg_Hp))
         + 0.32 * math.cos(math.radians(3 * avg_Hp + 6))
         - 0.20 * math.cos(math.radians(4 * avg_Hp - 63)))

    SL = 1 + 0.015 * (avg_Lp - 50)**2 / math.sqrt(20 + (avg_Lp - 50)**2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T

    avg_Cp7 = avg_Cp**7
    RT = (-math.sin(math.radians(60 * math.exp(-((avg_Hp - 275) / 25)**2)))
          * 2 * math.sqrt(avg_Cp7 / (avg_Cp7 + 25**7)))

    dE = math.sqrt(
        (dLp / SL)**2 +
        (dCp / SC)**2 +
        (dHp / SH)**2 +
        RT * (dCp / SC) * (dHp / SH)
    )
    return dE


def _shade_distance(target_lab, target_rgb, shade, user_undertone):
    """Combined CIEDE2000 + undertone + lightness scoring."""
    shade_lab = _rgb_to_lab(*shade["rgb"])
    de = _ciede2000(target_lab, shade_lab)

    # Undertone adjustment
    ut = (user_undertone or "").lower()
    shade_ut = (shade.get("undertone") or "").lower()
    if ut and shade_ut:
        if shade_ut == ut:
            de *= 0.82  # strong bonus for matching undertone
        elif ut == "neutral" or shade_ut == "neutral":
            de *= 0.95  # mild bonus, neutral is versatile
        elif (ut == "warm" and shade_ut == "cool") or (ut == "cool" and shade_ut == "warm"):
            de *= 1.15  # penalty for opposite undertone

    # Extra lightness penalty — being too dark/light is worse than hue shift
    target_L = target_lab[0]
    shade_L = float(shade.get("L") or shade_lab[0])
    L_diff = abs(target_L - shade_L)
    if L_diff > 10:
        de += (L_diff - 10) * 0.3  # progressive penalty beyond 10 L* units

    return de


def _distance_to_confidence(distance):
    """Convert CIEDE2000 distance to a 0-100 confidence percentage."""
    # dE < 2 = perfect match, dE > 20 = poor match
    if distance <= 2:
        return 98
    if distance >= 25:
        return 15
    return max(15, min(98, int(100 - (distance - 2) * 3.6)))


def _match_explanation(distance, shade, user_undertone):
    """Generate a human-readable explanation for the shade match."""
    conf = _distance_to_confidence(distance)
    shade_ut = (shade.get("undertone") or "").lower()
    ut = (user_undertone or "").lower()

    if conf >= 90:
        quality = "Excellent match"
    elif conf >= 75:
        quality = "Very good match"
    elif conf >= 60:
        quality = "Good match"
    elif conf >= 45:
        quality = "Approximate match"
    else:
        quality = "Rough estimate"

    parts = [quality]

    if shade_ut and ut:
        if shade_ut == ut:
            parts.append(f"undertone aligns ({shade_ut})")
        elif shade_ut == "neutral" or ut == "neutral":
            parts.append("neutral undertone is versatile")
        else:
            parts.append(f"undertone differs ({shade_ut} vs your {ut})")

    desc = shade.get("description", "")
    if desc and len(desc) > 10:
        parts.append(desc.split(".")[0].strip())

    return " — ".join(parts)


#  Data loading 

def _load_shades():
    global _SHADES, _SHADES_BY_BRAND
    if _SHADES:
        return
    with open(_SHADES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    for s in raw:
        h = (s.get("hex") or "").lstrip("#").lower()
        if len(h) != 6:
            continue
        s["hex"] = "#" + h
        s["rgb"] = _hex_to_rgb(h)
        s["lab"] = _rgb_to_lab(*s["rgb"])
        _SHADES.append(s)
        _SHADES_BY_BRAND.setdefault(s["brand"].lower(), []).append(s)


def match_shades(detected_hex: str | None,
                 brand: str = "",
                 product_name: str = "",
                 user_undertone: str | None = None,
                 top_n: int = 5) -> dict:
    """
    Find closest shades using CIEDE2000 perceptual distance.
    Resolution: brand+product -> brand only -> global pool.
    """
    candidates, source = _resolve_candidates(brand, product_name)

    if not candidates:
        return {
            "recommended": None,
            "alternatives": [],
            "shades": [],
            "source": "fallback",
        }

    is_reference_only = source in ("global", "fallback")

    if detected_hex:
        target_rgb = _hex_to_rgb(detected_hex.lstrip("#"))
        target_lab = _rgb_to_lab(*target_rgb)

        scored = []
        for c in candidates:
            d = _shade_distance(target_lab, target_rgb, c, user_undertone)
            scored.append((d, c))
        scored.sort(key=lambda t: t[0])

        best_d, best_s = scored[0]
        recommended = _shade_to_dict(best_s, distance=best_d)
        recommended["is_reference_only"] = is_reference_only
        recommended["confidence"] = _distance_to_confidence(best_d)
        recommended["explanation"] = _match_explanation(best_d, best_s, user_undertone)

        alternatives = []
        for d, c in scored[1:top_n]:
            alt = _shade_to_dict(c, distance=d)
            alt["confidence"] = _distance_to_confidence(d)
            alternatives.append(alt)
    else:
        recommended = None
        alternatives = []

    return {
        "recommended":       recommended,
        "alternatives":      alternatives,
        "shades":            [_shade_to_dict(c) for c in candidates],
        "source":            source,
        "is_reference_only": is_reference_only,
    }


def _resolve_candidates(brand, product_name):
    bn = _normalise_brand(brand)
    pn = (product_name or "").lower()

    aliases = {bn}
    if bn in _BRAND_ALIASES:
        aliases.add(_normalise_brand(_BRAND_ALIASES[bn]))

    brand_pool = []
    seen = set()
    for known_brand, items in _SHADES_BY_BRAND.items():
        kb = _normalise_brand(known_brand)
        if any(a == kb or a in kb or kb in a for a in aliases if a):
            if known_brand not in seen:
                brand_pool.extend(items)
                seen.add(known_brand)

    if brand_pool and pn:
        narrowed = [
            s for s in brand_pool
            if s["product"].lower() in pn
            or _normalise(s["product"]) in _normalise(pn)
            or pn in s["product"].lower()
        ]
        if narrowed and len(narrowed) >= 5:
            return narrowed, "brand+product"

    if brand_pool and len(brand_pool) >= _MIN_BRAND_SHADES:
        return brand_pool, "brand"

    return list(_SHADES), "global"


def _normalise_brand(b):
    import unicodedata
    s = unicodedata.normalize("NFKD", (b or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower().strip())


def _shade_to_dict(s, distance=None):
    out = {
        "name":        s["shade"],
        "hex":         s["hex"],
        "brand":       s["brand"],
        "product":     s["product"],
        "group":       s.get("group", ""),
        "lightness":   float(s.get("L", 0) or 0),
        "url":         s.get("url", ""),
        "undertone":   s.get("undertone", ""),
        "description": s.get("description", ""),
    }
    if distance is not None:
        out["distance"] = round(float(distance), 2)
    return out


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _normalise(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_load_shades()
