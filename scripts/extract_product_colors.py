"""Extract dominant product color from photos using K-means clustering."""

import os
import sys
import io
import argparse
import urllib.request
import urllib.error
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_CSV = os.path.join(_ROOT, "data", "products.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
}

COLOR_CATEGORIES = {
    "blush", "lipstick", "highlighter", "contour", "eyeshadow"
}


def rgb_to_hsv(r, g, b):
    """RGB 0-255 -> HSV (h:0-360, s:0-1, v:0-1)."""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rn, gn, bn)
    cmin = min(rn, gn, bn)
    delta = cmax - cmin
    if delta == 0:
        h = 0
    elif cmax == rn:
        h = 60 * (((gn - bn) / delta) % 6)
    elif cmax == gn:
        h = 60 * (((bn - rn) / delta) + 2)
    else:
        h = 60 * (((rn - gn) / delta) + 4)
    s = 0 if cmax == 0 else delta / cmax
    return h, s, cmax


def is_packaging(r, g, b) -> bool:
    """Skip near-white, near-black, neutral grey, and packaging beige."""
    h, s, v = rgb_to_hsv(r, g, b)
    if v > 0.95: return True
    if v < 0.10: return True
    if s < 0.15: return True
    return False


def is_skin(r, g, b) -> bool:
    """Rough skin-tone detector to skip arms/face in model photos."""
    if r < g or r < b: return False
    if r - b < 15:    return False
    if r > 250 and g > 220: return False
    h, s, v = rgb_to_hsv(r, g, b)
    if 5 <= h <= 50 and 0.15 <= s <= 0.55 and 0.40 <= v <= 0.95:
        return True
    return False


def vividness_score(r, g, b, count) -> float:
    """Score cluster interestingness: saturation x log(count)."""
    h, s, v = rgb_to_hsv(r, g, b)
    return s * (count ** 0.4) * (1.0 - abs(v - 0.6) * 0.5)


def extract_dominant_color(image_bytes: bytes, k: int = 6) -> tuple[str, str] | None:
    """
    Returns (primary_hex, alternate_hex) or None on failure.
    Filters out packaging/skin clusters and picks by vividness.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((200, 200))
        arr = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    except Exception:
        return None

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=4, random_state=42)
        km.fit(arr)
        centers = km.cluster_centers_
        labels  = km.labels_
        counts  = np.bincount(labels, minlength=k)
    except Exception:
        return None

    scored = []
    for i, (r, g, b) in enumerate(centers):
        ri, gi, bi = int(round(r)), int(round(g)), int(round(b))
        if is_packaging(ri, gi, bi):
            continue
        if is_skin(ri, gi, bi):
            continue
        score = vividness_score(ri, gi, bi, counts[i])
        scored.append((score, ri, gi, bi))

    if not scored:
        for i, (r, g, b) in enumerate(centers):
            ri, gi, bi = int(round(r)), int(round(g)), int(round(b))
            if is_packaging(ri, gi, bi):
                continue
            score = vividness_score(ri, gi, bi, counts[i])
            scored.append((score, ri, gi, bi))

    if not scored:
        return None

    scored.sort(reverse=True)
    primary = scored[0]
    alt = scored[1] if len(scored) > 1 else primary
    return (
        f"#{primary[1]:02x}{primary[2]:02x}{primary[3]:02x}",
        f"#{alt[1]:02x}{alt[2]:02x}{alt[3]:02x}",
    )


def fetch(url: str, timeout: int = 10) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"      fetch fail: {e}", flush=True)
        return None


def main(only: set[str] | None, force: bool):
    df = pd.read_csv(PRODUCTS_CSV)

    if "product_hex" not in df.columns:
        df["product_hex"] = ""
    if "product_hex_alt" not in df.columns:
        df["product_hex_alt"] = ""

    target_cats = only if only else COLOR_CATEGORIES
    print(f"Targeting categories: {sorted(target_cats)}")

    todo = df[df["category"].isin(target_cats)]
    print(f"{len(todo)} candidate products\n")

    extracted = 0
    skipped = 0
    for idx, row in todo.iterrows():
        pid    = int(row["product_id"])
        name   = str(row["name"])[:42]
        cat    = str(row["category"])
        url    = str(row.get("image_url", "") or "")
        cur    = str(row.get("product_hex", "") or "")

        if not url.startswith("http"):
            print(f"[{pid:03d}] {cat:<11} {name:<42} no url")
            continue

        if cur and cur.startswith("#") and not force:
            print(f"[{pid:03d}] {cat:<11} {name:<42} already {cur}")
            skipped += 1
            continue

        print(f"[{pid:03d}] {cat:<11} {name:<42} fetching... ", end="", flush=True)
        img_bytes = fetch(url)
        if not img_bytes:
            print("failed")
            continue

        result = extract_dominant_color(img_bytes)
        if not result:
            print("extraction failed")
            continue

        primary, alt = result
        df.at[idx, "product_hex"] = primary
        df.at[idx, "product_hex_alt"] = alt
        extracted += 1
        print(f"{primary} / {alt}")

        if extracted % 20 == 0:
            df.to_csv(PRODUCTS_CSV, index=False)

        time.sleep(0.15)

    df.to_csv(PRODUCTS_CSV, index=False)
    print(f"\n[Done] Extracted: {extracted}  | Skipped (already had hex): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated category list (default: all color categories)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if product already has hex")
    args = parser.parse_args()
    only = set(s.strip().lower() for s in args.only.split(",") if s.strip()) or None
    main(only, args.force)
