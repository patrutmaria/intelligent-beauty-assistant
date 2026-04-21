"""Hardcode product_hex for products where K-means extraction failed and classify eyeshadow palettes."""

import os
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_CSV = os.path.join(_ROOT, "data", "products.csv")

# Manual hex for non-palette products that failed extraction.
# Format: (product_id, hex, alt_hex_or_None)
MANUAL_HEX = [
    (3,   "#a83649", None),
    (4,   "#d56c8a", None),
    (48,  "#a02c3a", None),
    (49,  "#9b3a44", None),
    (54,  "#9b1f2c", None),
    (80,  "#a8262e", None),
    (101, "#d97a8d", None),
    (108, "#a55460", None),
    (122, "#7d2230", None),
    (150, "#a02340", None),
    (157, "#cd6772", None),
    (244, "#c46e72", None),
    (69,  "#f5d4a2", "#e8b8d0"),
    (70,  "#e8c89c", None),
    (203, "#f0c598", None),
    (228, "#f0d8b0", None),
    (45,  "#7c4828", None),
    (64,  "#6e3b22", None),
    (117, "#5e3a25", None),
    (169, "#a86840", None),
    (73,  "#e87a85", None),
    (85,  "#d8635c", None),
    (148, "#dd6f78", None),
]

# Eyeshadow palette classification by keywords
PALETTE_KEYWORDS = [
    (["naked heat", "heat", "burning", "fire", "ember", "warm"],
     "warm",     "#a04025", "Warm Tones"),
    (["smoky", "smokey", "noir", "black", "darker", "midnight"],
     "smoky",    "#3c2a2e", "Smoky"),
    (["nude", "neutral", "natural", "everyday", "soft glam", "tartelette toasted"],
     "nude",     "#b59175", "Nude / Neutral"),
    (["bloom", "rose", "pink", "berry", "burgundy", "wine"],
     "rose",     "#c25577", "Rose / Berry"),
    (["bronze", "gold", "shimmer", "metallic", "luxury"],
     "shimmer",  "#c08540", "Shimmer / Bronze"),
    (["colorful", "color", "rainbow", "playful", "festival", "snap shadows"],
     "colorful", "#7a4090", "Colorful"),
    (["amazonian clay", "matte", "tartelette amazonian"],
     "matte_neutral", "#9c7456", "Matte Neutral"),
    (["naked3", "naked 3", "rose gold"],
     "rose",     "#c87a6a", "Rose"),
    (["naked"],
     "nude",     "#b88e6f", "Nude / Neutral"),
    (["24/7"],
     "colorful", "#5a4070", "Colorful"),
    (["glitter", "diamond", "luminous"],
     "shimmer",  "#d4ad6f", "Shimmer"),
]


def classify_palette(name: str) -> tuple[str, str, str] | None:
    """Returns (palette_type, swatch_hex, label) or None."""
    n = (name or "").lower()
    for keywords, ptype, hex_, label in PALETTE_KEYWORDS:
        if any(k in n for k in keywords):
            return (ptype, hex_, label)
    return None


def main():
    df = pd.read_csv(PRODUCTS_CSV)
    if "product_hex" not in df.columns:
        df["product_hex"] = ""
    if "product_hex_alt" not in df.columns:
        df["product_hex_alt"] = ""
    if "palette_type" not in df.columns:
        df["palette_type"] = ""
    if "palette_label" not in df.columns:
        df["palette_label"] = ""

    print("[1/2] Filling hardcoded hex for non-palette products...")
    filled = 0
    for pid, hex_val, alt in MANUAL_HEX:
        idx = df.index[df["product_id"] == pid]
        if len(idx) == 0:
            print(f"  pid={pid} not found")
            continue
        i = idx[0]
        cur = str(df.at[i, "product_hex"] or "")
        if cur.startswith("#"):
            print(f"  pid={pid} already has {cur}, skipping")
            continue
        df.at[i, "product_hex"] = hex_val
        if alt:
            df.at[i, "product_hex_alt"] = alt
        filled += 1
        print(f"  pid={pid:3d} {df.at[i,'name'][:38]:<38} -> {hex_val}")
    print(f"  Filled: {filled}/{len(MANUAL_HEX)}\n")

    print("[2/2] Classifying eyeshadow palettes...")
    palettes = df[
        (df["category"] == "eyeshadow")
        & (df["product_hex"].fillna("") == "")
    ]
    classified = 0
    for i, row in palettes.iterrows():
        result = classify_palette(row["name"])
        if not result:
            result = ("nude", "#b59175", "Nude / Neutral")
        ptype, hex_, label = result
        df.at[i, "product_hex"]   = hex_
        df.at[i, "palette_type"]  = ptype
        df.at[i, "palette_label"] = label
        classified += 1
        print(f"  {row['name'][:42]:<42} -> {label} ({hex_})")
    print(f"  Classified: {classified} palettes\n")

    other_eyeshadow = df[
        (df["category"] == "eyeshadow")
        & (df["palette_type"].fillna("") == "")
    ]
    for i, row in other_eyeshadow.iterrows():
        result = classify_palette(row["name"])
        if result:
            df.at[i, "palette_type"]  = result[0]
            df.at[i, "palette_label"] = result[2]

    df.to_csv(PRODUCTS_CSV, index=False)
    print(f"[Done] Saved {PRODUCTS_CSV}")

    color_cats = ["blush","lipstick","highlighter","contour","eyeshadow"]
    relevant = df[df["category"].isin(color_cats)]
    has_hex  = relevant["product_hex"].fillna("").str.startswith("#").sum()
    print(f"\nFinal coverage: {has_hex}/{len(relevant)} ({100*has_hex/len(relevant):.0f}%) color products have hex")


if __name__ == "__main__":
    main()
