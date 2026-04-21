"""Fetch product images from Sephora/Open Beauty Facts and generate synthetic reviews."""

import os
import sys
import time
import json
import random
import urllib.parse
import urllib.request

import pandas as pd

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    print("[WARN] beautifulsoup4 not installed — HTML parsing disabled.")
    print("       Run: conda run -n base python -m pip install beautifulsoup4 lxml")
    _BS4 = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_CSV = os.path.join(_ROOT, "data", "products.csv")


HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
}


def _get(url: str, timeout: int = 8) -> str | None:
    """HTTP GET with browser headers; returns HTML text or None."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            charset = r.headers.get_content_charset("utf-8")
            return r.read().decode(charset, errors="replace")
    except Exception as e:
        print(f"    GET failed: {e}")
        return None


def _get_json(url: str, timeout: int = 8) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={**HEADERS, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"    JSON GET failed: {e}")
        return None


def _try_sephora_ro(name: str, brand: str) -> str | None:
    if not _BS4:
        return None
    q = urllib.parse.quote(f"{brand} {name}")
    html = _get(f"https://www.sephora.ro/search?q={q}")
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    for sel in ["img.product-image", "img.product-tile__image",
                "img[class*='product'][src*='cdnl']",
                "img[src*='sephora']", ".product-tile img"]:
        tag = soup.select_one(sel)
        if tag and tag.get("src", "").startswith("http"):
            return tag["src"]
        if tag and tag.get("data-src", "").startswith("http"):
            return tag["data-src"]
    return None


def _try_sephora_com(name: str, brand: str) -> str | None:
    """Use Sephora's internal catalog API."""
    q = urllib.parse.quote(f"{name} {brand}")
    url = (
        f"https://www.sephora.com/api/catalog/product/findProduct"
        f"?keyword={q}&currentPage=0&pageSize=3&content=true"
        f"&country=US&locale=en-US"
    )
    data = _get_json(url)
    if not data:
        return None
    try:
        products = (data.get("data", {}).get("products") or
                    data.get("products") or
                    data.get("searchResults", {}).get("products") or [])
        if not products:
            return None
        p = products[0]
        img = (p.get("primaryImage") or p.get("heroImage") or
               p.get("imageUrl") or "")
        if img.startswith("//"):
            img = "https:" + img
        return img if img.startswith("http") else None
    except Exception:
        return None


def _try_open_beauty_facts(name: str, brand: str) -> tuple[str | None, list[dict]]:
    """Returns (image_url, []) — OBF has no reviews."""
    q = urllib.parse.quote(name)
    b = urllib.parse.quote(brand)
    url = (
        f"https://world.openbeautyfacts.org/cgi/search.pl"
        f"?action=process&json=1&search_terms={q}&brands={b}"
        f"&fields=product_name,brands,image_url,image_front_url&page_size=3"
    )
    data = _get_json(url)
    if not data:
        return None, []
    try:
        products = data.get("products", [])
        for p in products:
            img = p.get("image_front_url") or p.get("image_url") or ""
            if img.startswith("http"):
                return img, []
    except Exception:
        pass
    return None, []


def main():
    df = pd.read_csv(PRODUCTS_CSV)

    if "image_url" not in df.columns:
        df["image_url"] = ""

    updated = 0

    for idx, row in df.iterrows():
        pid   = int(row["product_id"])
        name  = str(row["name"])
        brand = str(row["brand"])
        cur   = str(row.get("image_url", "") or "")

        print(f"[{pid:02d}] {name[:40]:<40} ", end="", flush=True)

        if cur and "loremflickr" not in cur and cur.startswith("http"):
            print("already has real image — skipping")
            continue

        img = None
        time.sleep(random.uniform(0.4, 0.9))

        print("sephora.ro... ", end="", flush=True)
        img = _try_sephora_ro(name, brand)

        if not img:
            print("sephora.com... ", end="", flush=True)
            img = _try_sephora_com(name, brand)

        if not img:
            print("openfacts... ", end="", flush=True)
            img, _ = _try_open_beauty_facts(name, brand)

        if img:
            df.at[idx, "image_url"] = img
            updated += 1
            print(f"{img[:60]}")
        else:
            print("keeping LoremFlickr fallback")

    df.to_csv(PRODUCTS_CSV, index=False)
    print(f"\n[Done] Updated {updated}/{len(df)} image URLs in {PRODUCTS_CSV}")


if __name__ == "__main__":
    random.seed(42)
    main()
