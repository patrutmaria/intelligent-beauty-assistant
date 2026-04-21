"""Matches detected skin colour to the nearest foundation shades from a dataset of 6,700+ real shades."""

import os
import json
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHADES_PATH = os.path.join(_ROOT, "data", "shades.json")

_SHADES: list[dict] = []
_SHADES_BY_BRAND: dict[str, list[dict]] = {}

# Below this threshold we use the global pool instead of brand-only matching
_MIN_BRAND_SHADES = 15

_BRAND_ALIASES = {
    "fenty beauty":           "fenty beauty by rihanna",
    "fenty beauty by rihanna": "fenty beauty",
    "loreal":                 "l'oreal",
    "l'oreal":                "l'oreal",
    "l'oreal":                "l'oreal",
    "estee lauder":           "estee lauder",
    "estee lauder":           "estee lauder",
    "make up for ever":       "make up for ever",
    "mufe":                   "make up for ever",
    "tarte cosmetics":        "tarte",
    "anastasia":              "anastasia beverly hills",
    "anastasia beverly hills": "anastasia",
}


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
        _SHADES.append(s)
        _SHADES_BY_BRAND.setdefault(s["brand"].lower(), []).append(s)


def match_shades(detected_hex: str | None,
                 brand: str = "",
                 product_name: str = "",
                 user_undertone: str | None = None,
                 top_n: int = 5) -> dict:
    """
    Find the closest shades to detected_hex from a given brand+product.
    Resolution: brand+product -> brand only -> global pool.
    Returns recommended shade, alternatives, full palette, and source label.
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
        target = _hex_to_rgb(detected_hex.lstrip("#"))
        ut = (user_undertone or "").lower().strip()
        scored = []
        for c in candidates:
            d = _rgb_distance(target, c["rgb"])
            # Undertone bonus: 15% closer if matching, 10% further if opposite
            cand_ut = (c.get("undertone") or "").lower()
            if ut and cand_ut:
                if cand_ut == ut:
                    d *= 0.85
                elif (ut == "warm" and cand_ut == "cool") or (ut == "cool" and cand_ut == "warm"):
                    d *= 1.10
            scored.append((d, c))
        scored.sort(key=lambda t: t[0])
        recommended = _shade_to_dict(scored[0][1], distance=scored[0][0])
        recommended["is_reference_only"] = is_reference_only
        alternatives = [_shade_to_dict(c, distance=d) for d, c in scored[1:top_n]]
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


def _resolve_candidates(brand: str, product_name: str) -> tuple[list[dict], str]:
    """Walk the resolution ladder: brand+product -> brand -> global."""
    bn = _normalise_brand(brand)
    pn = (product_name or "").lower()

    aliases = {bn}
    if bn in _BRAND_ALIASES:
        aliases.add(_normalise_brand(_BRAND_ALIASES[bn]))

    brand_pool: list[dict] = []
    seen_keys: set[str] = set()
    for known_brand, items in _SHADES_BY_BRAND.items():
        kb = _normalise_brand(known_brand)
        if any(a == kb or a in kb or kb in a for a in aliases if a):
            if known_brand not in seen_keys:
                brand_pool.extend(items)
                seen_keys.add(known_brand)

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


def _normalise_brand(b: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace."""
    import unicodedata
    s = unicodedata.normalize("NFKD", (b or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower().strip())


def _shade_to_dict(s: dict, distance: float | None = None) -> dict:
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


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_distance(a, b) -> float:
    """
    Luminance-weighted RGB distance. Being "too dark/light" matters more
    than hue shifts for foundation matching, so luminance (Y) gets a 5x penalty.
    """
    dR = a[0] - b[0]
    dG = a[1] - b[1]
    dB = a[2] - b[2]
    rgb_sq = 3 * dR * dR + 4 * dG * dG + 2 * dB * dB

    Ya = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
    Yb = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]
    dY = Ya - Yb

    return (rgb_sq + 5.0 * dY * dY) ** 0.5


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_load_shades()
