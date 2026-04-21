"""
Combines three shade datasets into a single high-coverage shades.json:
allShades.csv (Pudding), sephora.csv + ulta.csv (Pudding descriptions), and shades.json (FoundationFinder).
"""

import os
import re
import json
import pandas as pd
import unicodedata

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLSHADES_CSV = os.path.join(_ROOT, "data", "pudding_allShades.csv")
SEPHORA_CSV   = os.path.join(_ROOT, "data", "pudding_sephora.csv")
ULTA_CSV      = os.path.join(_ROOT, "data", "pudding_ulta.csv")
TENSORSHADE   = os.path.join(_ROOT, "data", "tensorshade_shades.json")
OUT           = os.path.join(_ROOT, "data", "shades.json")


def normalize(s) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def detect_undertone_from_description(desc: str) -> str:
    """Pull undertone from a Sephora-style description."""
    if not desc:
        return ""
    d = desc.lower()
    if "neutral" in d:
        return "neutral"
    if "warm" in d:    return "warm"
    if "cool" in d:    return "cool"
    if "olive" in d:   return "neutral"
    if "pink" in d:    return "cool"
    if "yellow" in d or "golden" in d: return "warm"
    return ""


def detect_undertone_from_name(name: str) -> str:
    """Letter suffix in shade name (e.g. '355N', '210W', '120C')."""
    if not name:
        return ""
    m = re.search(r"\b\d+([cwn])\b", str(name).lower())
    if m:
        return {"c": "cool", "w": "warm", "n": "neutral"}[m.group(1)]
    return ""


def main():
    print("[1/4] Loading allShades.csv (canonical hex source)...")
    df = pd.read_csv(ALLSHADES_CSV)
    df = df[df["hex"].notna() & (df["hex"].str.len() == 7)].copy()
    print(f"      {len(df):,} rows with valid hex")

    print("[2/4] Loading sephora.csv + ulta.csv (description sources)...")
    sephora = pd.read_csv(SEPHORA_CSV)
    ulta    = pd.read_csv(ULTA_CSV)
    desc_df = pd.concat([sephora, ulta], ignore_index=True)
    desc_df = desc_df[desc_df["description"].notna()]
    print(f"      {len(desc_df):,} rows with descriptions")

    desc_lookup = {}
    for _, r in desc_df.iterrows():
        brand_n   = normalize(r["brand"])
        product_n = normalize(r["product"])
        for name_col in ("name", "specific"):
            name_n = normalize(r.get(name_col))
            if name_n:
                desc_lookup[(brand_n, product_n, name_n)] = str(r["description"])
    print(f"      {len(desc_lookup):,} unique (brand,product,name) keys")

    print("[3/4] Enriching allShades with descriptions + undertones...")
    out_entries = []
    enriched = 0
    skipped_garbage = 0

    for _, row in df.iterrows():
        hex_val = str(row["hex"]).lower().lstrip("#")
        if len(hex_val) != 6:
            continue
        try:
            r = int(hex_val[0:2], 16); g = int(hex_val[2:4], 16); b = int(hex_val[4:6], 16)
            if r + g + b < 60 or r + g + b > 720:
                skipped_garbage += 1
                continue
            if r < b - 10:
                skipped_garbage += 1
                continue
        except Exception:
            continue

        shade_name = ""
        for col in ("name", "specific"):
            v = row.get(col)
            if not pd.isna(v) and str(v).strip().lower() not in ("nan", "none", ""):
                shade_name = str(v).strip()
                break
        if not shade_name:
            continue

        brand   = str(row["brand"])
        product = str(row["product"])
        b_n = normalize(brand)
        p_n = normalize(product)
        n_n = normalize(shade_name)

        description = desc_lookup.get((b_n, p_n, n_n)) or ""
        if not description:
            for (bb, pp, nn), d in desc_lookup.items():
                if bb == b_n and nn == n_n:
                    description = d
                    break

        ut = (detect_undertone_from_description(description)
              or detect_undertone_from_name(shade_name))

        L = float(row.get("lightness", 0) or 0)
        entry = {
            "brand":         brand,
            "brand_short":   brand[:3].lower(),
            "product":       product,
            "product_short": product[:5].lower().replace(" ", ""),
            "hex":           "#" + hex_val,
            "H":             str(row.get("hue", "")),
            "S":             str(row.get("sat", "")),
            "V":             "",
            "L":             str(int(round(L * 100))),
            "group":         _lightness_group(L),
            "shade":         shade_name,
            "url":           str(row.get("url", "")),
            "description":   description,
            "undertone":     ut,
            "source":        "pudding+desc" if description else "pudding",
        }
        out_entries.append(entry)
        if description:
            enriched += 1

    print(f"      {len(out_entries):,} entries kept ({skipped_garbage} skipped)")
    print(f"      {enriched:,} enriched with description ({100*enriched/len(out_entries):.0f}%)")

    # Append tensorshade entries for brands not already covered
    print("[4/4] Adding tensorshade brands not in allShades...")
    if os.path.exists(TENSORSHADE):
        ts = json.load(open(TENSORSHADE, encoding="utf-8-sig"))
        existing_brand_norms = {normalize(e["brand"]) for e in out_entries}
        existing_brand_words = set()
        for bn in existing_brand_norms:
            existing_brand_words.add(bn)
            for w in re.split(r"(?=[A-Z])", bn):
                if len(w) >= 4:
                    existing_brand_words.add(w.lower())

        added_ts = 0
        skipped_overlap = 0
        for s in ts:
            b_raw = s.get("brand", "")
            b_n = normalize(b_raw)
            overlap = any(
                (bn in b_n or b_n in bn) and len(bn) >= 4 and len(b_n) >= 4
                for bn in existing_brand_norms
            )
            if overlap:
                skipped_overlap += 1
                continue
            hex_val = str(s.get("hex", "")).lstrip("#").lower()
            if len(hex_val) != 6:
                continue
            try:
                int(hex_val, 16)
            except ValueError:
                continue
            s["hex"] = "#" + hex_val
            s["undertone"] = detect_undertone_from_name(s.get("shade", ""))
            s["source"] = "tensorshade"
            s.setdefault("description", "")
            out_entries.append(s)
            added_ts += 1
        print(f"      Added {added_ts} entries (skipped {skipped_overlap} overlapping)")

    # Dedup by hex, preferring entries with descriptions
    print(f"      Deduping {len(out_entries):,} entries by hex...")
    by_hex: dict[str, dict] = {}
    for e in out_entries:
        h = e["hex"].lower()
        existing = by_hex.get(h)
        if existing is None:
            by_hex[h] = e
            continue
        if e.get("description") and not existing.get("description"):
            by_hex[h] = e
        elif e.get("description") == existing.get("description") and len(e["brand"]) > len(existing["brand"]):
            by_hex[h] = e
    out_entries = list(by_hex.values())
    print(f"      After dedup: {len(out_entries):,}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out_entries, f, ensure_ascii=False, indent=1)
    print(f"\n[Done] Wrote {len(out_entries):,} shades to {OUT}")

    by_brand = {}
    for e in out_entries:
        by_brand[e["brand"]] = by_brand.get(e["brand"], 0) + 1
    print(f"       Brands: {len(by_brand)}")
    ut_count = sum(1 for e in out_entries if e.get("undertone"))
    print(f"       With undertone tag: {ut_count} ({100*ut_count/len(out_entries):.0f}%)")
    desc_count = sum(1 for e in out_entries if e.get("description"))
    print(f"       With description:    {desc_count} ({100*desc_count/len(out_entries):.0f}%)")
    print(f"\n       Top 15 brands:")
    for b, n in sorted(by_brand.items(), key=lambda kv: -kv[1])[:15]:
        print(f"         {b:<35} {n}")


def _lightness_group(L: float) -> str:
    L100 = L * 100
    if L100 >= 95: return "0"
    if L100 >= 85: return "1"
    if L100 >= 75: return "2"
    if L100 >= 65: return "3"
    if L100 >= 55: return "4"
    if L100 >= 45: return "5"
    if L100 >= 35: return "6"
    return "7"


if __name__ == "__main__":
    main()
