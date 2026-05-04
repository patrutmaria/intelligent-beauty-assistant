"""
    /opt/anaconda3/bin/python evaluation/eval_ranking.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph.graph_builder import BeautyGraphBuilder, SKIN_TYPES, EVENTS, FEATURE_DIM
from graph.vgae_model import VGAEBeauty


#  Metric functions 

def dcg_at_k(relevances, k):
    rel = np.array(relevances[:k], dtype=np.float32)
    positions = np.arange(1, len(rel) + 1)
    return np.sum(rel / np.log2(positions + 1))


def ndcg_at_k(ranked_relevances, k, n_relevant):
    dcg = dcg_at_k(ranked_relevances, k)
    ideal_rel = [1.0] * min(n_relevant, k)
    idcg = dcg_at_k(ideal_rel, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_relevances, k, n_relevant):
    hits = sum(ranked_relevances[:k])
    return hits / n_relevant if n_relevant > 0 else 0.0


def hit_rate_at_k(ranked_relevances, k):
    return 1.0 if sum(ranked_relevances[:k]) > 0 else 0.0


#  User history builder 

def build_user_histories(interactions_df, min_interactions=3):
    """
    Leave-last-out: last positive interaction (like/purchase) per user is test.
    """
    positive = interactions_df[interactions_df["interaction_type"].isin(["like", "purchase"])]

    user_histories = {}
    for uid, group in positive.groupby("user_id"):
        pids = group["product_id"].tolist()
        if len(pids) < min_interactions:
            continue
        train_pids = set(pids[:-1])
        test_pids = set(pids[-1:])
        user_histories[uid] = {"train": train_pids, "test": test_pids}

    return user_histories


#  Domain boost (mirrors BeautyRecommender._boost_product_scores) 

WARM_FINISHES = {"dewy", "luminous", "satin"}
COOL_FINISHES = {"matte"}

def apply_domain_boost(scores, prod_gids, builder, products_df, skin_type, event_type):
    """Lightweight domain boosting based on skin type and event compatibility."""
    scores = scores.copy()
    for i, gid in enumerate(prod_gids):
        pid = builder.global_id_to_product_id(gid)
        if pid is None:
            continue
        row = products_df[products_df["product_id"] == pid]
        if row.empty:
            continue
        r = row.iloc[0]

        # Skin-type match
        prod_skin = str(r["skin_type"]).lower()
        if prod_skin == skin_type.lower():
            scores[i] *= 1.20
        elif prod_skin == "all":
            scores[i] *= 1.05

        # Finish-skin compatibility
        finish = str(r.get("finish", "")).lower()
        if skin_type.lower() == "dry":
            if finish in {"dewy", "luminous", "satin"}:
                scores[i] *= 1.30
            elif finish == "matte":
                scores[i] *= 0.75
        elif skin_type.lower() == "oily":
            if finish == "matte":
                scores[i] *= 1.30
            elif finish in {"dewy", "luminous"}:
                scores[i] *= 0.80

        # Event match
        look_types = str(r.get("look_types", "")).lower()
        if event_type.lower() in look_types:
            scores[i] *= 1.15

    return scores


#  Baseline: Random 

def eval_random_baseline(user_histories, all_product_ids, ks, seed=42):
    rng = np.random.RandomState(seed)
    metrics = {f"Recall@{k}": [] for k in ks}
    metrics.update({f"NDCG@{k}": [] for k in ks})
    metrics.update({f"HitRate@{k}": [] for k in ks})

    all_pids = sorted(all_product_ids)

    for uid, history in user_histories.items():
        train_pids = history["train"]
        test_pids = history["test"]

        candidates = [p for p in all_pids if p not in train_pids]
        rng.shuffle(candidates)

        ranked_rel = [1.0 if pid in test_pids else 0.0 for pid in candidates]
        n_relevant = len(test_pids)

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(ranked_rel, k, n_relevant))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_rel, k, n_relevant))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(ranked_rel, k))

    return {m: np.mean(v) for m, v in metrics.items()}


#  Baseline: Popularity 

def eval_popularity_baseline(user_histories, interactions_df, all_product_ids, ks):
    """Rank products by global interaction count (most popular first)."""
    # Count all positive interactions per product
    positive = interactions_df[interactions_df["interaction_type"].isin(["like", "purchase"])]
    pop_counts = Counter(positive["product_id"].tolist())

    metrics = {f"Recall@{k}": [] for k in ks}
    metrics.update({f"NDCG@{k}": [] for k in ks})
    metrics.update({f"HitRate@{k}": [] for k in ks})

    for uid, history in user_histories.items():
        train_pids = history["train"]
        test_pids = history["test"]

        candidates = [(pid, pop_counts.get(pid, 0)) for pid in all_product_ids if pid not in train_pids]
        candidates.sort(key=lambda x: x[1], reverse=True)

        ranked_rel = [1.0 if pid in test_pids else 0.0 for pid, _ in candidates]
        n_relevant = len(test_pids)

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(ranked_rel, k, n_relevant))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_rel, k, n_relevant))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(ranked_rel, k))

    return {m: np.mean(v) for m, v in metrics.items()}


#  VGAE evaluation (pure + boosted) 

def eval_vgae(user_histories, embeddings, prod_gids, gid_to_pid, node_id_map,
              builder, products_df, interactions_df, ks, use_boost=False):
    """Evaluate VGAE with optional domain boosting."""
    prod_embs = embeddings[prod_gids]

    # Per-user skin/event info (needed for boosting)
    user_profiles = {}
    if use_boost:
        for _, row in interactions_df.drop_duplicates("user_id").iterrows():
            user_profiles[row["user_id"]] = {
                "skin_type": str(row["skin_type"]),
                "event_type": str(row["event_type"]),
            }

    metrics = {f"Recall@{k}": [] for k in ks}
    metrics.update({f"NDCG@{k}": [] for k in ks})
    metrics.update({f"HitRate@{k}": [] for k in ks})

    for uid, history in user_histories.items():
        train_pids = history["train"]
        test_pids = history["test"]

        user_key = ("user", uid)
        if user_key not in node_id_map:
            continue
        user_gid = node_id_map[user_key]
        user_emb = embeddings[user_gid]

        # Inner-product scores
        scores = torch.sigmoid((prod_embs * user_emb.unsqueeze(0)).sum(dim=1))
        scores = scores.cpu().numpy()

        # Apply domain boost if requested
        if use_boost and uid in user_profiles:
            prof = user_profiles[uid]
            scores = apply_domain_boost(
                scores, prod_gids, builder, products_df,
                prof["skin_type"], prof["event_type"]
            )

        # Build candidate list excluding training items
        candidates = []
        for i, gid in enumerate(prod_gids):
            pid = gid_to_pid.get(gid)
            if pid is None:
                continue
            if pid in train_pids:
                continue
            candidates.append((pid, scores[i]))

        candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_rel = [1.0 if pid in test_pids else 0.0 for pid, _ in candidates]
        n_relevant = len(test_pids)

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(ranked_rel, k, n_relevant))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_rel, k, n_relevant))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(ranked_rel, k))

    return {m: np.mean(v) for m, v in metrics.items()}


#  Main 

def main(model_path: str, data_dir: str, ks: list[int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph and model
    builder = BeautyGraphBuilder(data_dir)
    graph_data, node_id_map, node_types = builder.build()

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = VGAEBeauty(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["out_channels"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state_dict"].items()})
    model.eval()

    with torch.no_grad():
        x = graph_data.x.to(device)
        ei = graph_data.edge_index.to(device)
        mu, _ = model.encoder(x, ei)
        embeddings = mu

    # Load data
    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    products_df = pd.read_csv(os.path.join(data_dir, "products.csv"))
    user_histories = build_user_histories(interactions_df)

    # Product mappings
    prod_gids = builder.get_product_node_ids()
    all_product_ids = set()
    gid_to_pid = {}
    for gid in prod_gids:
        pid = builder.global_id_to_product_id(gid)
        if pid is not None:
            all_product_ids.add(pid)
            gid_to_pid[gid] = pid

    #  Run all methods 

    print(f"\n{'='*70}")
    print(f"  VGAE Beauty Recommender — Ranking Evaluation")
    print(f"  Protocol: Leave-Last-Out | Users: {len(user_histories)} | Products: {len(all_product_ids)}")
    print(f"  Model: {model_path} (Val AUC: {ckpt.get('val_auc', 0):.4f})")
    print(f"{'='*70}")

    print("\n  [1/4] Random baseline...")
    res_random = eval_random_baseline(user_histories, all_product_ids, ks)

    print("  [2/4] Popularity baseline...")
    res_pop = eval_popularity_baseline(user_histories, interactions_df, all_product_ids, ks)

    print("  [3/4] VGAE (pure link prediction)...")
    res_vgae = eval_vgae(user_histories, embeddings, prod_gids, gid_to_pid,
                         node_id_map, builder, products_df, interactions_df, ks, use_boost=False)

    print("  [4/4] VGAE + Domain Boost...")
    res_boosted = eval_vgae(user_histories, embeddings, prod_gids, gid_to_pid,
                            node_id_map, builder, products_df, interactions_df, ks, use_boost=True)

    #  Print comparison table 

    methods = [
        ("Random", res_random),
        ("Popularity", res_pop),
        ("VGAE", res_vgae),
        ("VGAE+Boost", res_boosted),
    ]

    print(f"\n{''*70}")
    print(f"  {'Method':<14}", end="")
    for k in ks:
        print(f" {'Rec@'+str(k):>7} {'NDCG@'+str(k):>8} {'HR@'+str(k):>7}", end="")
    print()
    print(f"  {''*14}", end="")
    for k in ks:
        print(f" {''*7} {''*8} {''*7}", end="")
    print()

    for name, res in methods:
        print(f"  {name:<14}", end="")
        for k in ks:
            r = res.get(f"Recall@{k}", 0)
            n = res.get(f"NDCG@{k}", 0)
            h = res.get(f"HitRate@{k}", 0)
            print(f" {r:>7.4f} {n:>8.4f} {h:>7.4f}", end="")
        print()

    print(f"{''*70}")

    #  Improvement summary 

    print(f"\n  Improvement over Random (Recall@{ks[-1]}):")
    rand_val = res_random.get(f"Recall@{ks[-1]}", 1e-8)
    for name, res in methods[1:]:
        val = res.get(f"Recall@{ks[-1]}", 0)
        pct = ((val - rand_val) / rand_val * 100) if rand_val > 0 else 0
        print(f"    {name:<14} {pct:>+7.1f}%")

    print(f"\n{'='*70}\n")

    return {name: res for name, res in methods}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking evaluation for VGAE Beauty Recommender")
    parser.add_argument("--model", default="models/vgae_beauty.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="data/", help="Path to data directory")
    parser.add_argument("--Ks", default="5,10,20", help="Comma-separated K values")
    args = parser.parse_args()

    ks = [int(k) for k in args.Ks.split(",")]
    main(args.model, args.data_dir, ks)
