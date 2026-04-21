import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

FEATURE_DIM = 32   # shared node feature width

# One-hot vocabulary maps
SKIN_TYPES  = ["oily", "dry", "combination", "normal", "all", "unknown"]
EVENTS      = ["wedding", "office", "day", "night", "party", "casual", "natural", "evening", "glam", "festival"]

# TODO: add "skincare" category, import skincare products from sephora_raw.csv
#       to enable real Sephora reviews (eyachawechi/my-sephora-data on HuggingFace)

CATEGORIES  = ["foundation", "concealer", "blush", "eyeshadow", "mascara",
               "lipstick", "powder", "highlighter", "contour", "primer", "perfume"]
FINISHES    = ["matte", "dewy", "natural", "satin", "luminous",
               "floral", "fresh_citrus", "woody", "warm_spicy", "sweet_gourmand", "aromatic"]
STYLES      = ["natural", "glam", "evening", "office", "wedding", "festival", "editorial", "minimal"]
OCCASIONS   = ["day", "night", "work", "party", "wedding", "casual"]
INGR_TYPES  = ["humectant", "emollient", "preservative", "pigment",
               "antioxidant", "surfactant", "film_former", "SPF"]
INGR_EFFECTS = ["moisturizing", "anti_aging", "soothing", "brightening",
                "oil_control", "coverage", "smoothing", "protecting"]


def _onehot(val, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    key = str(val).lower().strip()
    if key in vocab:
        vec[vocab.index(key)] = 1.0
    return vec


class BeautyGraphBuilder:
    """Builds a PyG Data object from CSV files for VGAE training and inference."""

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.node_id_map: dict = {}   # (type_str, original_id) -> global_node_id
        self.node_types:  list = []   # global_node_id -> int type code
        self._product_global_ids: list = []
        self._look_global_ids:    list = []
        self._user_global_ids:    list = []

    def build(self):
        products_df     = pd.read_csv(os.path.join(self.data_dir, "products.csv"))
        looks_df        = pd.read_csv(os.path.join(self.data_dir, "looks.csv"))
        ingredients_df  = pd.read_csv(os.path.join(self.data_dir, "ingredients.csv"))
        interactions_df = pd.read_csv(os.path.join(self.data_dir, "interactions.csv"))

        all_features = []
        edges = []
        idx = 0

        # Products
        prod_feats = self._encode_products(products_df)
        for i, pid in enumerate(products_df["product_id"]):
            self.node_id_map[("product", int(pid))] = idx + i
            self.node_types.append(1)
            self._product_global_ids.append(idx + i)
        all_features.append(prod_feats)
        idx += len(products_df)

        # Looks
        look_feats = self._encode_looks(looks_df)
        for i, lid in enumerate(looks_df["look_id"]):
            self.node_id_map[("look", int(lid))] = idx + i
            self.node_types.append(2)
            self._look_global_ids.append(idx + i)
        all_features.append(look_feats)
        idx += len(looks_df)

        # Ingredients
        ingr_feats = self._encode_ingredients(ingredients_df)
        for i, iid in enumerate(ingredients_df["ingredient_id"]):
            self.node_id_map[("ingredient", int(iid))] = idx + i
            self.node_types.append(3)
        all_features.append(ingr_feats)
        idx += len(ingredients_df)

        # Users
        users = interactions_df["user_id"].unique()
        user_feats = self._encode_users(users, interactions_df)
        for i, uid in enumerate(users):
            self.node_id_map[("user", uid)] = idx + i
            self.node_types.append(0)
            self._user_global_ids.append(idx + i)
        all_features.append(user_feats)
        idx += len(users)

        # Edges: Product <-> Look
        ingr_name_to_id = dict(zip(
            ingredients_df["name"].str.strip(),
            ingredients_df["ingredient_id"]
        ))

        for _, row in looks_df.iterrows():
            lid_g = self.node_id_map[("look", int(row["look_id"]))]
            for pid_str in str(row["product_ids"]).split(","):
                pid = int(pid_str.strip())
                if ("product", pid) in self.node_id_map:
                    pg = self.node_id_map[("product", pid)]
                    edges += [[pg, lid_g], [lid_g, pg]]

        # Edges: Product <-> Ingredient
        for _, row in products_df.iterrows():
            pg = self.node_id_map[("product", int(row["product_id"]))]
            for ingr_name in str(row["key_ingredients"]).split(","):
                ingr_name = ingr_name.strip()
                if ingr_name in ingr_name_to_id:
                    ig = self.node_id_map[("ingredient", int(ingr_name_to_id[ingr_name]))]
                    edges += [[pg, ig], [ig, pg]]

        # Edges: User <-> Product
        for _, row in interactions_df.iterrows():
            uid = row["user_id"]
            pid = int(row["product_id"])
            if ("user", uid) in self.node_id_map and ("product", pid) in self.node_id_map:
                ug = self.node_id_map[("user", uid)]
                pg = self.node_id_map[("product", pid)]
                edges += [[ug, pg], [pg, ug]]

        # Edges: Product <-> Product (KNN by price within category)
        K_NEIGHBOURS = 6
        cat_groups = products_df.groupby("category")["product_id"].apply(list)
        for pids in cat_groups:
            if len(pids) <= K_NEIGHBOURS + 1:
                for i in range(len(pids)):
                    for j in range(i + 1, len(pids)):
                        pa = self.node_id_map.get(("product", int(pids[i])))
                        pb = self.node_id_map.get(("product", int(pids[j])))
                        if pa is not None and pb is not None:
                            edges += [[pa, pb], [pb, pa]]
            else:
                cat_df = products_df[products_df["product_id"].isin(pids)].sort_values("price")
                ordered = cat_df["product_id"].tolist()
                for i, pid in enumerate(ordered):
                    pa = self.node_id_map.get(("product", int(pid)))
                    if pa is None:
                        continue
                    for j in range(i + 1, min(i + 1 + K_NEIGHBOURS, len(ordered))):
                        pb = self.node_id_map.get(("product", int(ordered[j])))
                        if pb is not None:
                            edges += [[pa, pb], [pb, pa]]

        x = torch.tensor(np.vstack(all_features), dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index), self.node_id_map, self.node_types

    def get_product_node_ids(self) -> list[int]:
        return list(self._product_global_ids)

    def get_look_node_ids(self) -> list[int]:
        return list(self._look_global_ids)

    def get_user_node_ids(self) -> list[int]:
        return list(self._user_global_ids)

    def global_id_to_product_id(self, gid: int) -> int | None:
        for (t, oid), g in self.node_id_map.items():
            if g == gid and t == "product":
                return oid
        return None

    def global_id_to_look_id(self, gid: int) -> int | None:
        for (t, oid), g in self.node_id_map.items():
            if g == gid and t == "look":
                return oid
        return None

    # Feature encoding (32 dims per node)

    def _encode_products(self, df):
        rows = []
        for _, r in df.iterrows():
            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            f[1] = 1.0
            f[4:15] = _onehot(r["category"], CATEGORIES)
            f[15:20] = _onehot(r["skin_type"], SKIN_TYPES[:5])
            f[20] = min(float(r["price"]) / 100.0, 1.0)
            f[21] = float(r["rating"]) / 5.0
            f[22:32] = _onehot(r["finish"], FINISHES[:10])
            rows.append(f)
        return np.array(rows)

    def _encode_looks(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, r in df.iterrows():
            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            f[2] = 1.0
            f[4:12] = _onehot(r["style"], STYLES)
            f[12:18] = _onehot(r["occasion"], OCCASIONS)
            rows.append(f)
        return np.array(rows)

    def _encode_ingredients(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, r in df.iterrows():
            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            f[3] = 1.0
            f[4:12] = _onehot(r["type"], INGR_TYPES)
            f[12:20] = _onehot(r["effect"], INGR_EFFECTS)
            rows.append(f)
        return np.array(rows)

    def _encode_users(self, users, interactions_df: pd.DataFrame) -> np.ndarray:
        rows = []
        for uid in users:
            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            f[0] = 1.0
            ui = interactions_df[interactions_df["user_id"] == uid]
            if not ui.empty:
                f[4:10] = _onehot(ui.iloc[0]["skin_type"], SKIN_TYPES[:6])
                f[10:20] = _onehot(ui.iloc[0]["event_type"], EVENTS)
                f[20] = min(len(ui) / 10.0, 1.0)
            rows.append(f)
        return np.array(rows)
