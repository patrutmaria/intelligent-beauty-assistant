"""Fetch real product images via DuckDuckGo Image Search and update products.csv."""

import os
import re
import json
import time
import random
import urllib.parse
import urllib.request

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_CSV = os.path.join(_ROOT, "data", "products.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def ddg_image_search(query: str) -> str | None:
    """Search DuckDuckGo Images, return the first result URL."""
    try:
        q = urllib.parse.quote(query)
        url1 = f"https://duckduckgo.com/?q={q}&iax=images&ia=images"
        req = urllib.request.Request(url1, headers=HEADERS)
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "replace")
        m = re.search(r'vqd=([\d-]+)', html) or re.search(r'vqd="([\d-]+)"', html)
        if not m:
            return None
        vqd = m.group(1)
        url2 = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={q}&vqd={vqd}&p=1&f=type:photo"
        req2 = urllib.request.Request(
            url2,
            headers={**HEADERS, "Referer": "https://duckduckgo.com/", "Accept": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req2, timeout=10).read())
        results = data.get("results", [])
        for r in results[:5]:
            img = r.get("image", "")
            if not img.startswith("http"):
                continue
            w = r.get("width", 0)
            if w and w < 200:
                continue
            return img
        if results:
            return results[0].get("image") or None
    except Exception as e:
        print(f"   ERROR: {e}")
    return None


def main():
    df = pd.read_csv(PRODUCTS_CSV)
    if "image_url" not in df.columns:
        df["image_url"] = ""

    updated = 0
    for idx, row in df.iterrows():
        pid    = int(row["product_id"])
        name   = str(row["name"])
        brand  = str(row["brand"])
        cur    = str(row.get("image_url", "") or "")

        if cur and "loremflickr" not in cur and "duckduckgo" not in cur and cur.startswith("http"):
            print(f"[{pid:03d}] {name[:45]:<45} already real", flush=True)
            continue

        query = f"{brand} {name} product"
        print(f"[{pid:03d}] {name[:45]:<45} searching... ", end="", flush=True)

        img = ddg_image_search(query)
        if img:
            df.at[idx, "image_url"] = img
            updated += 1
            print(f"{img[:65]}", flush=True)
        else:
            print("no result", flush=True)

        if (idx + 1) % 10 == 0:
            df.to_csv(PRODUCTS_CSV, index=False)

        time.sleep(random.uniform(1.2, 2.4))

    df.to_csv(PRODUCTS_CSV, index=False)
    print(f"\n[Done] Updated {updated}/{len(df)} image URLs", flush=True)
    print(f"       Saved to {PRODUCTS_CSV}", flush=True)


if __name__ == "__main__":
    random.seed(42)
    main()
