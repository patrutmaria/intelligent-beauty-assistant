import os
import math
import torch
import numpy as np
import pandas as pd

from .graph_builder import BeautyGraphBuilder, FEATURE_DIM, SKIN_TYPES, EVENTS
from .vgae_model import VGAEBeauty
from skin.shade_matcher import match_shades
from skin.color_advisor import recommend_families, classify_color, family_label, family_score, FAMILIES as COLOR_FAMILIES


class BeautyRecommender:

    def __init__(self, model_path: str = "models/vgae_beauty.pt", data_dir: str = "data/"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.builder = BeautyGraphBuilder(data_dir)
        self.graph_data, self.node_id_map, self.node_types = self.builder.build()

        self.products_df = pd.read_csv(os.path.join(data_dir, "products.csv"))
        self.looks_df = pd.read_csv(os.path.join(data_dir, "looks.csv"))

        reviews_path = os.path.join(data_dir, "reviews.csv")
        self.reviews_df = (
            pd.read_csv(reviews_path) if os.path.exists(reviews_path)
            else pd.DataFrame(columns=["product_id", "reviewer", "rating", "title", "body", "date"])
        )

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = VGAEBeauty(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["out_channels"]).to(self.device)
        self.model.load_state_dict({k: v.to(self.device) for k, v in ckpt["model_state_dict"].items()})
        self.model.eval()

        with torch.no_grad():
            x = self.graph_data.x.to(self.device)
            ei = self.graph_data.edge_index.to(self.device)
            mu, _ = self.model.encoder(x, ei)
            self.embeddings = mu

        print(f"[BeautyRecommender] ready | nodes={self.graph_data.num_nodes} | edges={self.graph_data.num_edges} | latent_dim={ckpt['out_channels']}")

    #Helpers

    @staticmethod
    def _safe_scent(r):
        for key in ("scent_family", "finish"):
            v = r.get(key, "")
            if v is None:
                continue
            try:
                if isinstance(v, float) and math.isnan(v):
                    continue
            except (TypeError, ValueError):
                pass
            s = str(v).strip()
            if s and s.lower() != "nan":
                return s
        return ""

    #Public API

    def recommend_for_new_user(self, skin_type: str, event_type: str,
                               budget: float | None = None, undertone: str | None = None,
                               lightness: str | None = None, concerns: list[str] | None = None,
                               acne: str | None = None, skin_hex: str | None = None,
                               top_k_products: int = 5, top_k_looks: int = 3) -> dict:
        user_emb = self._cold_start_embedding(skin_type, event_type)
        result = self._predict_links(user_emb, skin_type, event_type, budget, undertone,
                                     top_k_products, top_k_looks,
                                     concerns=concerns, acne=acne, lightness=lightness)
        for p in result.get("recommended_products", []):
            self._attach_shades(p, skin_hex, undertone)
            self._attach_color_match(p, undertone, lightness)
        return result

    #Routine builder

    _ROUTINE_STEPS = [
        ("primer",      "Primer",         "Smooth canvas — minimises pores and helps everything last longer."),
        ("foundation",  "Foundation",     "Even out your skin tone with the right coverage for your event."),
        ("concealer",   "Concealer",      "Brighten under-eyes and conceal any spots."),
        ("powder",      "Setting Powder", "Lock everything in place, especially the T-zone."),
        ("contour",     "Contour",        "Add structure and definition to your face."),
        ("blush",       "Blush",          "Bring back a healthy flush of colour."),
        ("highlighter", "Highlighter",    "Catch the light on cheekbones, brow bones and cupid's bow."),
        ("eyeshadow",   "Eyeshadow",      "Build dimension on the lid for the look."),
        ("mascara",     "Mascara",        "Open up the eyes with length and volume."),
        ("lipstick",    "Lipstick",       "Finish with the perfect lip colour."),
        ("perfume",     "Perfume",        "The final spritz — your signature scent."),
    ]

    _EVENT_SCENT_PREFS = {
        "wedding":  ["floral", "warm_spicy", "woody"],
        "office":   ["fresh_citrus", "aromatic", "floral"],
        "evening":  ["warm_spicy", "woody", "sweet_gourmand"],
        "natural":  ["fresh_citrus", "floral", "aromatic"],
        "glam":     ["warm_spicy", "woody", "floral"],
        "festival": ["fresh_citrus", "sweet_gourmand", "aromatic"],
    }

    def build_routine(self, skin_type: str, event_type: str,
                      budget: float | None = None, undertone: str | None = None,
                      lightness: str | None = None, concerns: list[str] | None = None,
                      acne: str | None = None, skin_hex: str | None = None,
                      categories: list[str] | None = None, top_k_per_category: int = 1,
                      preferred_scents: list[str] | None = None) -> dict:
        cat_filter = {c.lower().strip() for c in categories if c} if categories else None
        user_emb = self._cold_start_embedding(skin_type, event_type)
        prod_gids = self.builder.get_product_node_ids()

        prod_scores = torch.sigmoid(
            (self.embeddings[prod_gids] * user_emb.unsqueeze(0)).sum(dim=1)
        ).cpu().detach().numpy()
        prod_scores = self._boost_product_scores(
            prod_scores, prod_gids, skin_type, event_type, budget, undertone,
            concerns=concerns, acne=acne, lightness=lightness,
            preferred_scents=preferred_scents,
        )

        # Bucket products by category
        cat_buckets: dict[str, list] = {}
        for i, gid in enumerate(prod_gids):
            pid = self.builder.global_id_to_product_id(gid)
            if pid is None:
                continue
            row = self.products_df[self.products_df["product_id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            if budget is not None and float(r["price"]) > budget:
                continue
            cat_buckets.setdefault(str(r["category"]).lower(), []).append((pid, float(prod_scores[i]), r))

        steps = []
        total_price = 0.0
        step_idx = 0
        for cat, label, hint in self._ROUTINE_STEPS:
            if cat_filter is not None and cat not in cat_filter:
                continue
            bucket = cat_buckets.get(cat, [])
            if not bucket:
                continue
            bucket.sort(key=lambda t: t[1], reverse=True)
            for pid, score, r in bucket[:max(1, top_k_per_category)]:
                step_idx += 1
                total_price += float(r["price"])
                step_dict = {
                    "step": step_idx, "category": cat, "category_label": label,
                    "instruction": hint, "product_id": int(pid),
                    "name": r["name"], "brand": r["brand"],
                    "price": float(r["price"]), "finish": r["finish"],
                    "rating": float(r["rating"]), "score": float(score),
                    "image_url": str(r.get("image_url", "") or ""),
                    "key_ingredients": str(r.get("key_ingredients", "") or ""),
                    "matched_ingredients": self._get_matched_ingredients(
                        str(r.get("key_ingredients", "") or ""), concerns, acne),
                    "reason": self._explain(r, skin_type, event_type, budget, undertone),
                    "reviews": self._get_reviews(int(pid)),
                    "scent_family": self._safe_scent(r),
                    "is_unisex": bool(r.get("is_unisex", False)),
                }
                self._attach_shades(step_dict, skin_hex, undertone)
                self._attach_color_match(step_dict, undertone, lightness)
                steps.append(step_dict)

        return {
            "user_profile": {"skin_type": skin_type, "event_type": event_type, "budget": budget, "undertone": undertone},
            "routine_steps": steps, "total_price": round(total_price, 2), "step_count": len(steps),
        }

    #Cold-start embedding

    def _cold_start_embedding(self, skin_type: str, event_type: str) -> torch.Tensor:
        user_gids = self.builder.get_user_node_ids()
        new_feat = self._build_user_feature(skin_type, event_type)

        if not user_gids:
            prod_gids = self.builder.get_product_node_ids()
            return self.embeddings[prod_gids].mean(dim=0)

        user_feats = self.graph_data.x[user_gids].to(self.device)
        user_embs = self.embeddings[user_gids]

        nf_norm = new_feat / (new_feat.norm() + 1e-8)
        uf_norm = user_feats / (user_feats.norm(dim=1, keepdim=True) + 1e-8)
        sims = (uf_norm @ nf_norm).clamp(min=0)

        if sims.sum() < 1e-8:
            return user_embs.mean(dim=0)

        weights = sims / sims.sum()
        return (user_embs * weights.unsqueeze(1)).sum(dim=0)

    def _build_user_feature(self, skin_type: str, event_type: str) -> torch.Tensor:
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[0] = 1.0  # node type: user

        sk = skin_type.lower().strip()
        sk6 = SKIN_TYPES[:6]
        if sk in sk6:
            f[4 + sk6.index(sk)] = 1.0

        ev = event_type.lower().strip()
        if ev in EVENTS:
            f[10 + EVENTS.index(ev)] = 1.0

        return torch.tensor(f, dtype=torch.float, device=self.device)

    #Link prediction

    def _predict_links(self, user_emb, skin_type, event_type, budget, undertone,
                       top_k_products, top_k_looks,
                       concerns: list[str] | None = None, acne: str | None = None,
                       lightness: str | None = None) -> dict:
        prod_gids = self.builder.get_product_node_ids()
        look_gids = self.builder.get_look_node_ids()

        prod_scores = torch.sigmoid(
            (self.embeddings[prod_gids] * user_emb.unsqueeze(0)).sum(dim=1)
        ).cpu().detach().numpy()
        look_scores = torch.sigmoid(
            (self.embeddings[look_gids] * user_emb.unsqueeze(0)).sum(dim=1)
        ).cpu().detach().numpy()

        prod_scores = self._boost_product_scores(
            prod_scores, prod_gids, skin_type, event_type, budget, undertone,
            concerns=concerns, acne=acne, lightness=lightness)

        top_prod_idx = np.argsort(prod_scores)[::-1][:top_k_products]
        recommended_products = []
        for i in top_prod_idx:
            pid = self.builder.global_id_to_product_id(prod_gids[i])
            if pid is None:
                continue
            row = self.products_df[self.products_df["product_id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            recommended_products.append({
                "product_id": int(pid), "name": r["name"], "brand": r["brand"],
                "category": r["category"], "price": float(r["price"]),
                "finish": r["finish"], "rating": float(r["rating"]),
                "score": float(prod_scores[i]),
                "reason": self._explain(r, skin_type, event_type, budget, undertone),
                "image_url": str(r.get("image_url", "") or ""),
                "key_ingredients": str(r.get("key_ingredients", "") or ""),
                "matched_ingredients": self._get_matched_ingredients(
                    str(r.get("key_ingredients", "") or ""), concerns, acne),
                "reviews": self._get_reviews(pid),
                "scent_family": self._safe_scent(r),
                "is_unisex": bool(r.get("is_unisex", False)),
            })

        top_look_idx = np.argsort(look_scores)[::-1][:top_k_looks]
        recommended_looks = []
        for i in top_look_idx:
            lid = self.builder.global_id_to_look_id(look_gids[i])
            if lid is None:
                continue
            row = self.looks_df[self.looks_df["look_id"] == lid]
            if row.empty:
                continue
            r = row.iloc[0]
            recommended_looks.append({
                "look_id": int(lid), "name": r["name"],
                "style": r["style"], "occasion": r["occasion"],
                "score": float(look_scores[i]),
            })

        return {
            "user_profile": {"skin_type": skin_type, "event_type": event_type, "budget": budget, "undertone": undertone},
            "recommended_look": recommended_looks[0] if recommended_looks else None,
            "alternative_looks": recommended_looks[1:],
            "recommended_products": recommended_products,
        }

    #Concern-ingredient matching

    _CONCERN_INGREDIENTS = {
        "acne":         ["niacinamide", "salicylic acid", "tea tree", "zinc", "kaolin", "bentonite"],
        "blackheads":   ["salicylic acid", "kaolin", "bentonite", "charcoal"],
        "whiteheads":   ["salicylic acid", "niacinamide", "kaolin"],
        "blemishes":    ["niacinamide", "salicylic acid", "zinc"],
        "redness":      ["niacinamide", "centella", "allantoin", "panthenol", "aloe", "green tea"],
        "sensitive":    ["centella", "allantoin", "panthenol", "aloe", "ceramide", "squalane"],
        "fine lines":   ["retinol", "peptide", "hyaluronic acid", "vitamin c", "ceramide"],
        "wrinkles":     ["retinol", "peptide", "vitamin c", "hyaluronic acid", "ceramide"],
        "dull":         ["vitamin c", "glycolic acid", "lactic acid", "niacinamide", "alpha arbutin"],
        "pores":        ["niacinamide", "salicylic acid", "kaolin", "bentonite"],
        "pigmentation": ["vitamin c", "niacinamide", "alpha arbutin", "kojic acid", "tranexamic"],
        "blackspots":   ["vitamin c", "niacinamide", "alpha arbutin", "kojic acid"],
        "dark spots":   ["vitamin c", "niacinamide", "alpha arbutin", "kojic acid"],
        "dark circles": ["caffeine", "vitamin c", "niacinamide", "peptide", "hyaluronic acid"],
        "eye bags":     ["caffeine", "peptide", "hyaluronic acid"],
        "dehydrated":   ["hyaluronic acid", "glycerin", "squalane", "ceramide", "panthenol"],
        "dryness":      ["hyaluronic acid", "glycerin", "squalane", "ceramide", "shea butter"],
    }

    def _get_matched_ingredients(self, product_ingredients: str,
                                 concerns: list[str] | None, acne: str | None) -> list[dict]:
        if not product_ingredients:
            return []

        target_to_concern: dict[str, list[str]] = {}
        if concerns:
            for c in concerns:
                key = (c or "").lower().strip()
                for ing in self._CONCERN_INGREDIENTS.get(key, []):
                    target_to_concern.setdefault(ing, []).append(key)
        if acne and acne.lower() in {"moderate", "severe"}:
            for ing in self._CONCERN_INGREDIENTS["acne"]:
                if "acne" not in target_to_concern.get(ing, []):
                    target_to_concern.setdefault(ing, []).append("acne")

        if not target_to_concern:
            return []

        matched = []
        seen = set()
        for raw_ing in product_ingredients.split(","):
            ing = raw_ing.strip()
            if not ing or ing.lower() in seen:
                continue
            il = ing.lower()
            for target, concern_list in target_to_concern.items():
                if target in il:
                    matched.append({"ingredient": ing, "concerns": sorted(set(concern_list))})
                    seen.add(il)
                    break
        return matched

    #Score boosting

    _WARM_FINISHES = {"dewy", "luminous", "satin"}
    _COOL_FINISHES = {"matte"}
    _WARM_BRANDS = {"charlotte tilbury", "benefit", "nars", "too faced"}
    _COOL_BRANDS = {"mac", "urban decay", "smashbox"}
    _COLOR_CATEGORIES = {"blush", "lipstick", "highlighter", "contour", "eyeshadow"}

    def _boost_product_scores(self, scores, prod_gids, skin_type, event_type, budget, undertone,
                               concerns: list[str] | None = None, acne: str | None = None,
                               lightness: str | None = None,
                               preferred_scents: list[str] | None = None):
        scores = scores.copy()

        target_ingredients: set[str] = set()
        if concerns:
            for c in concerns:
                target_ingredients.update(self._CONCERN_INGREDIENTS.get((c or "").lower().strip(), []))
        if acne and acne.lower() in {"moderate", "severe"}:
            target_ingredients.update(self._CONCERN_INGREDIENTS["acne"])

        for i, gid in enumerate(prod_gids):
            pid = self.builder.global_id_to_product_id(gid)
            if pid is None:
                continue
            row = self.products_df[self.products_df["product_id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]

            # Skin-type match
            prod_skin = r["skin_type"].lower()
            if prod_skin == skin_type.lower():
                scores[i] *= 1.20
            elif prod_skin == "all":
                scores[i] *= 1.05

            # Finish-skin compatibility
            finish_lower = str(r.get("finish", "")).lower()
            if skin_type.lower() == "dry":
                if finish_lower in {"dewy", "luminous", "satin"}:
                    scores[i] *= 1.30
                elif finish_lower == "matte":
                    scores[i] *= 0.75
            elif skin_type.lower() == "oily":
                if finish_lower == "matte":
                    scores[i] *= 1.30
                elif finish_lower in {"dewy", "luminous"}:
                    scores[i] *= 0.80
            elif skin_type.lower() == "combination":
                if finish_lower in {"satin", "natural"}:
                    scores[i] *= 1.15

            # Event match
            look_types = str(r.get("look_types", "")).lower()
            if event_type.lower() in look_types:
                scores[i] *= 1.10

            # Undertone boost
            if undertone:
                finish = r["finish"].lower()
                brand = r["brand"].lower()
                if undertone == "warm":
                    if finish in self._WARM_FINISHES: scores[i] *= 1.08
                    if brand in self._WARM_BRANDS: scores[i] *= 1.05
                elif undertone == "cool":
                    if finish in self._COOL_FINISHES: scores[i] *= 1.08
                    if brand in self._COOL_BRANDS: scores[i] *= 1.05

            # Concern ingredient boost
            if target_ingredients:
                ingr = str(r.get("key_ingredients", "")).lower()
                hits = sum(1 for ti in target_ingredients if ti in ingr)
                if hits > 0:
                    scores[i] *= 1.0 + min(0.60, 0.12 * hits)
            if concerns:
                prod_concerns = str(r.get("concerns", "")).lower()
                if prod_concerns:
                    user_set = {c.lower().strip() for c in concerns}
                    tag_hits = sum(1 for c in user_set if c in prod_concerns)
                    if tag_hits > 0:
                        scores[i] *= 1.0 + min(0.75, 0.15 * tag_hits)

            if acne == "severe" and r["finish"].lower() in {"dewy", "luminous"}:
                scores[i] *= 0.85

            # Perfume scent boost : scent family + event applied independently
            cat_lower = str(r.get("category", "")).lower()
            if cat_lower == "perfume":
                scent = self._safe_scent(r)
                if preferred_scents:
                    pref_set = {s.lower() for s in preferred_scents}
                    scores[i] *= 3.0 if scent.lower() in pref_set else 0.15
                if scent and event_type:
                    prefs = self._EVENT_SCENT_PREFS.get(event_type.lower(), [])
                    if scent in prefs:
                        idx = prefs.index(scent)
                        scores[i] *= [1.50, 1.30, 1.15][idx] if idx < 3 else 1.0
                    elif not preferred_scents:
                        scores[i] *= 0.55
                look_types = str(r.get("look_types", "") or "").lower()
                if event_type and event_type.lower() in look_types:
                    scores[i] *= 1.40

            # Color family boost
            if cat_lower in self._COLOR_CATEGORIES and undertone:
                prod_hex = str(r.get("product_hex", "") or "")
                if prod_hex.startswith("#"):
                    fscore, _ = family_score(cat_lower, prod_hex, undertone, lightness)
                    if fscore >= 0.8:   scores[i] *= 1.20
                    elif fscore >= 0.6: scores[i] *= 1.10
                    elif 0 < fscore < 0.4: scores[i] *= 0.92

            # Budget penalty
            if budget is not None and float(r["price"]) > budget:
                scores[i] *= 0.30

        return scores

    #Shade matching

    def _attach_shades(self, product: dict, skin_hex: str | None, user_undertone: str | None = None) -> None:
        if product.get("category") not in {"foundation", "concealer", "contour"}:
            return
        result = match_shades(
            detected_hex=skin_hex,
            brand=product.get("brand", ""),
            product_name=product.get("name", ""),
            user_undertone=user_undertone,
        )
        product["shades"] = result["shades"]
        product["recommended_shade"] = result["recommended"]
        product["alternative_shades"] = result["alternatives"]
        product["shade_match_source"] = result["source"]

    #Color advisor

    def _attach_color_match(self, product: dict, user_undertone: str | None, user_lightness: str | None) -> None:
        cat = product.get("category", "")
        if cat not in self._COLOR_CATEGORIES:
            return

        pid = product.get("product_id")
        row = self.products_df[self.products_df["product_id"] == pid]
        if row.empty:
            return
        r = row.iloc[0]
        prod_hex = str(r.get("product_hex", "") or "")
        prod_hex_alt = str(r.get("product_hex_alt", "") or "")
        if not prod_hex.startswith("#"):
            return

        score, fam_key = family_score(cat, prod_hex, user_undertone, user_lightness)
        if not fam_key and prod_hex_alt.startswith("#"):
            score, fam_key = family_score(cat, prod_hex_alt, user_undertone, user_lightness)

        verdict = "great" if score >= 0.8 else "decent" if score >= 0.5 else "skip"

        product["product_hex"] = prod_hex
        product["product_hex_alt"] = prod_hex_alt or prod_hex
        product["color_family_key"] = fam_key
        product["color_family"] = family_label(fam_key) if fam_key else ""
        product["color_match"] = verdict
        product["color_score"] = round(score, 2)
        product["recommended_families"] = recommend_families(cat, user_lightness, user_undertone)

        # Palette overrides for eyeshadow
        palette_type = str(r.get("palette_type", "") or "")
        palette_label = str(r.get("palette_label", "") or "")
        if palette_type and cat == "eyeshadow":
            product["palette_type"] = palette_type
            product["palette_label"] = palette_label
            if user_undertone:
                if palette_type == "warm" and user_undertone == "warm":
                    product["color_match"] = "great"
                elif palette_type in ("rose", "smoky") and user_undertone == "cool":
                    product["color_match"] = "great"
                elif palette_type in ("nude", "matte_neutral", "shimmer", "colorful"):
                    product["color_match"] = "decent"
                elif palette_type == "warm" and user_undertone == "cool":
                    product["color_match"] = "decent"
                elif palette_type in ("rose", "smoky") and user_undertone == "warm":
                    product["color_match"] = "decent"

    #Reviews

    def _get_reviews(self, product_id: int, top_n: int = 3) -> list:
        if self.reviews_df.empty:
            return []
        rows = self.reviews_df[self.reviews_df["product_id"] == product_id]
        rows = rows.sort_values("rating", ascending=False).head(top_n)
        return [
            {"reviewer": str(r["reviewer"]), "rating": int(r["rating"]),
             "title": str(r["title"]), "body": str(r["body"]), "date": str(r["date"])}
            for _, r in rows.iterrows()
        ]

    #Explanation generation

    def _explain(self, r, skin_type: str, event_type: str,
                 budget: float | None, undertone: str | None = None) -> str:
        reasons = []
        if r["skin_type"].lower() in (skin_type.lower(), "all"):
            reasons.append(f"formulated for {skin_type} skin")
        finish = r["finish"].lower()
        if finish == "matte" and skin_type.lower() == "oily":
            reasons.append("matte finish controls shine")
        elif finish in ("dewy", "luminous") and skin_type.lower() == "dry":
            reasons.append(f"{finish} finish adds radiance")
        look_types = str(r.get("look_types", "")).lower()
        if event_type.lower() in look_types:
            reasons.append(f"ideal for {event_type} looks")
        if float(r["rating"]) >= 4.7:
            reasons.append(f"top-rated ({r['rating']}/5.0)")
        if budget is not None and float(r["price"]) <= budget * 0.5:
            reasons.append("budget-friendly")
        if undertone == "warm" and finish in self._WARM_FINISHES:
            reasons.append(f"{finish} finish enhances warm undertones")
        elif undertone == "cool" and finish in self._COOL_FINISHES:
            reasons.append("cool-neutral base suits your undertone")
        return "; ".join(reasons) if reasons else "matches your overall profile"
