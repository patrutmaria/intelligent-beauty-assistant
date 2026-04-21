"""Merge The Pudding shades dataset into shades.json (FoundationFinder base)."""

import os
import re
import json
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUDDING_CSV = os.path.join(_ROOT, "data", "pudding_allShades.csv")
SHADES_JSON = os.path.join(_ROOT, "data", "shades.json")

UNDERTONE_KEYWORDS = {
    "cool":    ["cool", " c ", "pink"],
    "warm":    ["warm", " w ", "golden", "yellow", "olive", "peach"],
    "neutral": ["neutral", " n "],
}

LIGHTNESS_GROUPS = [
    (95, "0"),
    (85, "1"),
    (75, "2"),
    (65, "3"),
    (55, "4"),
    (45, "5"),
    (35, "6"),
    (0,  "7"),
]


def detect_undertone(description: str, name: str) -> str:
    """Pull undertone from description or shade name."""
    blob = f"{description or ''} {name or ''}".lower()
    m = re.search(r"\b\d+([cwn])\b", blob)
    if m:
        return {"c": "cool", "w": "warm", "n": "neutral"}[m.group(1)]
    for ut, kws in UNDERTONE_KEYWORDS.items():
        if any(k in blob for k in kws):
            return ut
    return ""


def lightness_to_group(L01: float) -> str:
    """L from CSV is 0..1; map to a 0-7 darkness group."""
    L100 = L01 * 100
    for thresh, grp in LIGHTNESS_GROUPS:
        if L100 >= thresh:
            return grp
    return "7"


def main():
    print("[1/3] Loading existing shades.json...")
    with open(SHADES_JSON, encoding="utf-8") as f:
        existing = json.load(f)
    print(f"      {len(existing)} entries from FoundationFinder")

    print("[2/3] Loading Pudding allShades.csv...")
    df = pd.read_csv(PUDDING_CSV)
    df = df[df["hex"].notna() & (df["hex"].str.len() == 7)]
    print(f"      {len(df)} valid entries from Pudding")

    new_entries = []
    for _, row in df.iterrows():
        hex_val = str(row["hex"]).lower().lstrip("#")
        if len(hex_val) != 6:
            continue
        try:
            int(hex_val, 16)
        except ValueError:
            continue

        try:
            r = int(hex_val[0:2], 16); g = int(hex_val[2:4], 16); b = int(hex_val[4:6], 16)
            if r + g + b < 60 or r + g + b > 720:
                continue
            if r < b - 10:
                continue
        except Exception:
            continue

        L = float(row.get("lightness", 0) or 0)
        name = ""
        for col in ("name", "specific"):
            v = row.get(col, "")
            if not pd.isna(v) and str(v).strip().lower() not in ("nan", "none", ""):
                name = str(v).strip()
                break
        if not name:
            continue
        undertone = detect_undertone(row.get("description", ""), name)

        entry = {
            "brand":         str(row["brand"]),
            "brand_short":   str(row["brand"])[:3].lower(),
            "product":       str(row["product"]),
            "product_short": str(row["product"])[:5].lower().replace(" ", ""),
            "hex":           "#" + hex_val,
            "H":             str(row.get("hue", "")),
            "S":             str(row.get("sat", "")),
            "V":             "",
            "L":             str(int(round(L * 100))),
            "group":         lightness_to_group(L),
            "shade":         str(name).strip(),
            "url":           str(row.get("url", "")),
            "undertone":     undertone,
            "source":        "pudding",
        }
        new_entries.append(entry)

    for e in existing:
        e.setdefault("source", "foundationfinder")
        e.setdefault("undertone", "")

    print(f"      Cleaned to {len(new_entries)} valid Pudding entries")

    print("[3/3] Merging...")
    seen = set()
    merged = []
    for e in existing + new_entries:
        key = (e["brand"].lower().strip(), e["hex"].lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(e)

    with open(SHADES_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    print(f"\n[Done] Wrote {len(merged)} merged shades to {SHADES_JSON}")
    by_brand = {}
    for e in merged:
        b = e["brand"]
        by_brand[b] = by_brand.get(b, 0) + 1
    print(f"       Brands: {len(by_brand)}")
    print(f"       Top 15 brands by shade count:")
    for b, n in sorted(by_brand.items(), key=lambda kv: -kv[1])[:15]:
        print(f"         {b:<35} {n}")


if __name__ == "__main__":
    main()
