"""Import Sephora HuggingFace dataset -> products.csv."""

import os
import re
import sys
import argparse
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(_ROOT, "data", "sephora_raw.csv")
OUT   = os.path.join(_ROOT, "data", "products.csv")

CATEGORY_MAP = [
    ("foundation",  ["foundation", "tinted moisturizer", "bb & cc", "bb cream", "cc cream"]),
    ("concealer",   ["concealer", "color correct", "corrector"]),
    ("blush",       ["blush"]),
    ("eyeshadow",   ["eyeshadow", "eye shadow", "eye palette"]),
    ("mascara",     ["mascara", "lash primer"]),
    ("lipstick",    ["lipstick", "lip gloss", "lip stain", "liquid lip", "lip plumper", "lip balm"]),
    ("powder",      ["setting powder", "setting spray", "loose powder", "pressed powder"]),
    ("highlighter", ["highlighter", "luminizer"]),
    ("contour",     ["contour", "bronzer", "sculpting"]),
    ("primer",      ["primer"]),
]

LOOK_TYPES = {
    "foundation":  "natural,office,wedding,glam",
    "concealer":   "natural,office,wedding,glam",
    "blush":       "natural,glam,wedding,festival",
    "eyeshadow":   "glam,evening,editorial,festival",
    "mascara":     "natural,office,glam,wedding",
    "lipstick":    "glam,evening,wedding",
    "powder":      "office,wedding,glam,evening",
    "highlighter": "glam,evening,wedding,festival",
    "contour":     "glam,evening,editorial",
    "primer":      "natural,office,wedding",
    "perfume":     "wedding,glam,evening",
}

FINISHES = ["matte", "dewy", "satin", "luminous", "natural"]

SCENT_FAMILIES = [
    ("floral",        ["floral scent", "rose", "jasmine", "peony", "tuberose",
                       "gardenia", "lily", "iris", "violet"]),
    ("fresh_citrus",  ["fresh scent", "citrus", "bergamot", "lemon", "lime",
                       "grapefruit", "neroli", "aquatic", "marine"]),
    ("woody",         ["woody & earthy scent", "woody", "sandalwood", "cedar",
                       "vetiver", "patchouli", "earthy"]),
    ("warm_spicy",    ["warm &spicy scent", "warm & spicy", "amber", "oud",
                       "cinnamon", "clove", "pepper", "saffron", "smoky"]),
    ("sweet_gourmand",["sweet scent", "gourmand", "vanilla", "caramel",
                       "praline", "honey", "chocolate", "almond"]),
    ("aromatic",      ["aromatic", "lavender", "mint", "sage", "rosemary",
                       "thyme", "herbal", "herbaceous"]),
]

NAME_HINTS = [
    ("warm_spicy",   ["opium", "noir", "intense", "spice", "tobacco", "leather",
                      "amber", "oud", "fireplace", "santal", "tabac", "myrrh"]),
    ("woody",        ["coco", "wood", "cedar", "santal", "vetiver", "oak",
                      "forest", "bois", "earth"]),
    ("sweet_gourmand",["vanilla", "caramel", "praline", "honey", "chocolate",
                       "almond", "sugar", "candy", "dessert", "gourmand"]),
    ("floral",       ["rose", "jasmine", "peony", "tuberose", "gardenia", "lily",
                      "iris", "violet", "daisy", "flower", "bloom", "bouquet",
                      "blossom", "petal", "miss dior", "chloe", "chloe", "her"]),
    ("fresh_citrus", ["light blue", "bright", "fresh", "citrus", "lemon", "lime",
                      "bergamot", "aqua", "marine", "ocean", "sea", "crystal",
                      "verde", "blanc", "white"]),
    ("aromatic",     ["lavender", "mint", "sage", "rosemary", "basil", "thyme",
                      "herbal", "tea", "matcha"]),
]


def infer_scent_family(highlights_list: list[str], description: str = "",
                        product_name: str = "") -> str:
    """Pick the most prominent olfactory family from highlights, description, or name."""
    hl_blob = " ".join(highlights_list).lower()
    counts = {}
    for fam, kws in SCENT_FAMILIES:
        n = sum(1 for k in kws if k in hl_blob)
        if n > 0:
            counts[fam] = n

    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]

    blob = (product_name or "").lower() + " " + (description or "").lower()
    name_counts = {}
    for fam, kws in NAME_HINTS:
        n = sum(1 for k in kws if k in blob)
        if n > 0:
            name_counts[fam] = n
    if name_counts:
        return max(name_counts.items(), key=lambda kv: kv[1])[0]

    return "floral"


def is_unisex(highlights_list: list[str], product_name: str = "") -> bool:
    blob = " ".join(highlights_list).lower() + " " + (product_name or "").lower()
    return any(k in blob for k in ["unisex", "genderless", "for him & her"])


def perfume_look_types(scent_family: str) -> str:
    """Map scent family -> event types for graph features."""
    return {
        "floral":         "wedding,natural,glam",
        "fresh_citrus":   "office,natural,festival",
        "woody":          "evening,glam,office",
        "warm_spicy":     "evening,glam,wedding",
        "sweet_gourmand": "evening,glam,natural",
        "aromatic":       "office,natural,festival",
    }.get(scent_family, "natural,evening")


def map_category(row) -> str | None:
    for col in ("tertiary_category", "secondary_category", "primary_category"):
        v = row.get(col)
        if pd.isna(v):
            continue
        v_low = str(v).lower()
        for our_cat, keywords in CATEGORY_MAP:
            if any(k in v_low for k in keywords):
                return our_cat
    return None


def infer_finish(name: str, highlights: str) -> str:
    text = f"{name} {highlights}".lower()
    if "matte" in text:    return "matte"
    if "dewy" in text:     return "dewy"
    if "luminous" in text or "radiant" in text or "glow" in text: return "luminous"
    if "satin" in text:    return "satin"
    return "natural"


def parse_highlights(s) -> list[str]:
    if pd.isna(s):
        return []
    try:
        items = eval(s, {"__builtins__": {}}, {})
        if isinstance(items, list):
            return [str(x).lower() for x in items]
    except Exception:
        pass
    return []


def infer_skin_type(highlights_list: list[str]) -> str:
    """Pull explicit 'Best for X skin' tags out of highlights."""
    blob = " ".join(highlights_list)
    is_dry  = "dry" in blob
    is_oily = "oily" in blob
    is_combo = "combo" in blob or "combination" in blob
    is_norm = "normal" in blob
    if is_oily and not is_dry and not is_combo: return "oily"
    if is_dry  and not is_oily and not is_combo: return "dry"
    if is_combo and not is_oily and not is_dry:  return "combination"
    if is_norm and not is_oily and not is_dry and not is_combo: return "normal"
    if is_oily or is_dry or is_combo or is_norm: return "all"
    return "all"


def infer_look_types(highlights_list: list[str], category: str) -> str:
    blob = " ".join(highlights_list)
    looks = set()
    if "long-wear" in blob or "longwear" in blob or "24" in blob or "long lasting" in blob or "wedding" in blob:
        looks.add("wedding")
    if "natural finish" in blob or "light coverage" in blob or "tinted" in blob or "everyday" in blob:
        looks.add("natural")
    if "full coverage" in blob or "high coverage" in blob or "matte" in blob:
        looks.add("office")
    if "luminous" in blob or "radiant" in blob or "dewy" in blob or "glow" in blob:
        looks.add("glam")
    if "transfer" in blob or "waterproof" in blob:
        looks.add("evening")
    if not looks:
        looks = set(LOOK_TYPES.get(category, "natural").split(","))
    return ",".join(sorted(looks))


def infer_concerns(highlights_list: list[str], product_name: str = "",
                    ingredients_text: str = "") -> list[str]:
    """Pull skin concerns from highlights, name, and ingredients."""
    blob = " ".join(highlights_list).lower() + " " + (product_name or "").lower()
    ing_blob = (ingredients_text or "").lower()

    concerns = []
    rules = [
        ("acne",         ["acne", "blemish", "anti-blemish", "pimple", "breakout"],
                          ["salicylic", "niacinamide", "tea tree", "zinc"]),
        ("dryness",      ["dryness", "hydrating", "hydration", "dry skin", "moistur"],
                          ["hyaluronic", "glycerin", "squalane", "shea butter", "ceramide"]),
        ("dehydrated",   ["dehydrat", "plumping"],
                          ["hyaluronic", "glycerin"]),
        ("dull",         ["dullness", "radiance", "brightening", "luminous", "glow", "illuminating"],
                          ["vitamin c", "niacinamide", "alpha arbutin", "lactic"]),
        ("fine lines",   ["fine lines", "anti-aging", "anti aging", "wrinkle", "youth", "youthful", "firming"],
                          ["retinol", "peptide", "bakuchiol", "resveratrol"]),
        ("wrinkles",     ["wrinkle", "anti-aging", "anti aging", "youthful"],
                          ["retinol", "peptide", "bakuchiol"]),
        ("redness",      ["redness", "calming", "soothing", "anti-redness"],
                          ["centella", "allantoin", "panthenol", "aloe", "bisabolol"]),
        ("pores",        ["pore", "minimizing", "minimising", "pore-refining", "blurring"],
                          ["niacinamide", "salicylic", "kaolin"]),
        ("sensitive",    ["sensitive", "gentle", "fragrance-free"],
                          ["centella", "panthenol", "aloe", "ceramide"]),
        ("dark spots",   ["dark spot", "pigmentation", "even tone", "brightening", "discoloration"],
                          ["vitamin c", "niacinamide", "alpha arbutin", "kojic", "tranexamic"]),
        ("dark circles", ["dark circle", "under-eye", "under eye", "eye area"],
                          ["caffeine", "vitamin c", "peptide"]),
    ]

    for tag, kw_text, kw_ingr in rules:
        if any(k in blob for k in kw_text):
            concerns.append(tag)
        elif any(k in ing_blob for k in kw_ingr):
            concerns.append(tag)

    return concerns


NOTABLE_INGREDIENTS = [
    "Niacinamide", "Hyaluronic Acid", "Retinol", "Vitamin C", "Vitamin E",
    "Salicylic Acid", "Glycolic Acid", "Lactic Acid", "Glycerin", "Squalane",
    "Ceramide", "Peptide", "Caffeine", "Zinc Oxide", "Titanium Dioxide",
    "Mica", "Iron Oxides", "Talc", "Kaolin", "Bentonite", "Silica",
    "Dimethicone", "Cyclopentasiloxane", "Aloe", "Centella", "Allantoin",
    "Panthenol", "Castor Oil", "Coconut Oil", "Jojoba Oil", "Shea Butter",
    "Argan Oil", "Camellia", "Green Tea", "SPF", "Tocopherol", "Bisabolol",
    "Resveratrol", "Alpha Arbutin", "Kojic Acid", "Tranexamic", "Bakuchiol",
]


def parse_ingredients(s) -> str:
    """Extract notable active ingredients from the raw ingredients field."""
    if pd.isna(s):
        return ""
    try:
        items = eval(s, {"__builtins__": {}}, {})
        if isinstance(items, list):
            text = " ".join(str(x) for x in items)
        else:
            text = str(s)
    except Exception:
        text = str(s)

    text_lower = text.lower()
    found = []
    for ing in NOTABLE_INGREDIENTS:
        if ing.lower() in text_lower and ing not in found:
            found.append(ing)
        if len(found) >= 6:
            break
    return ",".join(found)


def _build_makeup(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    mk = df[df["primary_category"] == "Makeup"].copy()
    mk["our_category"] = mk.apply(map_category, axis=1)
    mk = mk[mk["our_category"].notna()]
    mk = mk[mk["price_usd"].notna() & (mk["price_usd"] > 0)]
    mk = mk[mk["out_of_stock"] != 1]
    print(f"      -> {len(mk):,} mappable makeup products")

    mk["loves_count"] = pd.to_numeric(mk["loves_count"], errors="coerce").fillna(0)
    mk = mk.sort_values("loves_count", ascending=False).head(top_n).reset_index(drop=True)
    mk["_highlights_list"] = mk["highlights"].apply(parse_highlights)
    mk["_ingredients"] = mk["ingredients"].apply(parse_ingredients)

    return pd.DataFrame({
        "name":            mk["product_name"],
        "brand":           mk["brand_name"],
        "category":        mk["our_category"],
        "price":           mk["price_usd"].round(2),
        "skin_type":       mk["_highlights_list"].apply(infer_skin_type),
        "finish":          [infer_finish(n, " ".join(h)) for n, h in zip(mk["product_name"].fillna(""), mk["_highlights_list"])],
        "rating":          mk["rating"].fillna(4.0).round(1),
        "look_types":      [infer_look_types(h, c) for h, c in zip(mk["_highlights_list"], mk["our_category"])],
        "key_ingredients": mk["_ingredients"],
        "concerns":        [
            ",".join(infer_concerns(h, n, ing))
            for h, n, ing in zip(mk["_highlights_list"], mk["product_name"].fillna(""), mk["_ingredients"])
        ],
        "scent_family":    "",
        "is_unisex":       False,
    })


def _build_perfumes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    fr = df[df["primary_category"] == "Fragrance"].copy()
    fr = fr[fr["tertiary_category"].fillna("").isin(["Perfume", "Cologne"])]
    fr = fr[fr["price_usd"].notna() & (fr["price_usd"] > 0)]
    fr = fr[fr["out_of_stock"] != 1]
    print(f"      -> {len(fr):,} mappable perfumes")

    fr["loves_count"] = pd.to_numeric(fr["loves_count"], errors="coerce").fillna(0)
    fr = fr.sort_values("loves_count", ascending=False).head(top_n).reset_index(drop=True)
    fr["_highlights_list"] = fr["highlights"].apply(parse_highlights)
    fr["_scent_family"]    = [
        infer_scent_family(h, "", n)
        for h, n in zip(fr["_highlights_list"], fr["product_name"].fillna(""))
    ]

    return pd.DataFrame({
        "name":            fr["product_name"],
        "brand":           fr["brand_name"],
        "category":        "perfume",
        "price":           fr["price_usd"].round(2),
        "skin_type":       "all",
        "finish":          fr["_scent_family"],
        "rating":          fr["rating"].fillna(4.0).round(1),
        "look_types":      [perfume_look_types(s) for s in fr["_scent_family"]],
        "key_ingredients": "",
        "concerns":        "",
        "scent_family":    fr["_scent_family"],
        "is_unisex":       [is_unisex(h, n) for h, n in zip(fr["_highlights_list"], fr["product_name"].fillna(""))],
    })


def main(top_n_makeup: int, top_n_perfume: int):
    if not os.path.exists(RAW):
        print(f"ERROR: {RAW} not found.")
        print("Download with:")
        print('  curl -sL "https://huggingface.co/datasets/MayaKitzis/sephora_products/resolve/main/product_info.csv" -o data/sephora_raw.csv')
        sys.exit(1)

    df = pd.read_csv(RAW)
    print(f"[1/4] Loaded {len(df):,} raw products")

    print(f"[2/4] Building makeup catalog (top {top_n_makeup})...")
    makeup = _build_makeup(df, top_n_makeup)

    print(f"[3/4] Building perfume catalog (top {top_n_perfume})...")
    perfumes = _build_perfumes(df, top_n_perfume)

    out = pd.concat([makeup, perfumes], ignore_index=True)
    out.insert(0, "product_id", range(1, len(out) + 1))
    out["image_url"] = ""

    if os.path.exists(OUT):
        try:
            old = pd.read_csv(OUT)
            if "image_url" in old.columns:
                lookup = {(str(r["name"]).strip(), str(r["brand"]).strip()): str(r.get("image_url",""))
                          for _, r in old.iterrows() if str(r.get("image_url","")).startswith("http")}
                preserved = 0
                for i, r in out.iterrows():
                    key = (str(r["name"]).strip(), str(r["brand"]).strip())
                    if key in lookup:
                        out.at[i, "image_url"] = lookup[key]
                        preserved += 1
                print(f"      Preserved {preserved} existing image URLs")
        except Exception as e:
            print(f"      Could not preserve images: {e}")

    out.to_csv(OUT, index=False)
    print(f"[4/4] Wrote {len(out)} products to {OUT}")
    print(f"      Categories: {out['category'].value_counts().to_dict()}")
    if "scent_family" in out.columns:
        sf = out[out["category"]=="perfume"]["scent_family"].value_counts().to_dict()
        print(f"      Scent families: {sf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=250,
                        help="Top makeup products to keep (default: 250)")
    parser.add_argument("--top-n-perfume", type=int, default=600,
                        help="Top perfumes to keep (default: 600 = essentially all)")
    args = parser.parse_args()
    main(args.top_n, args.top_n_perfume)
