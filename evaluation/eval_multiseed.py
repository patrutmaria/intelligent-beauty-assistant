"""
    /opt/anaconda3/bin/python evaluation/eval_multiseed.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LGConv
from torch_geometric.utils import negative_sampling
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph.graph_builder import BeautyGraphBuilder
from graph.vgae_model import VGAEBeauty
from graph.trainer import _split_edges
from evaluation.eval_ranking import (
    build_user_histories, recall_at_k, ndcg_at_k, hit_rate_at_k,
    eval_random_baseline, eval_popularity_baseline,
)


#  LightGCN 

class LightGCN(torch.nn.Module):
    
    def __init__(self, num_nodes: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.num_layers = num_layers
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def encode(self, edge_index):
        x = self.embedding.weight
        layer_embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embs.append(x)
        # Layer combination (mean)
        return torch.stack(layer_embs, dim=0).mean(dim=0)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def bpr_loss(self, z, pos_edge_index, neg_edge_index):
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        n = min(pos_scores.size(0), neg_scores.size(0))
        return -F.logsigmoid(pos_scores[:n] - neg_scores[:n]).mean()


def train_lightgcn(graph_data, epochs=500, lr=0.005, emb_dim=64, num_layers=3,
                   seed=42, verbose=False):
    """Train LightGCN and return embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_edges, val_edges, test_edges = _split_edges(graph_data.edge_index, seed=seed)

    model = LightGCN(graph_data.num_nodes, emb_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_edges = train_edges.to(device)

    best_val_score = 0.0
    best_emb = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(t_edges)
        neg_edges = negative_sampling(t_edges, num_nodes=graph_data.num_nodes,
                                      num_neg_samples=t_edges.size(1))
        loss = model.bpr_loss(z, t_edges, neg_edges)
        # L2 regularization on embeddings
        reg_loss = 1e-5 * model.embedding.weight.norm(2).pow(2)
        (loss + reg_loss).backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                z_val = model.encode(t_edges)
                v_edges = val_edges.to(device)
                v_neg = negative_sampling(v_edges, graph_data.num_nodes, v_edges.size(1))
                pos_pred = torch.sigmoid(model.decode(z_val, v_edges)).cpu().numpy()
                neg_pred = torch.sigmoid(model.decode(z_val, v_neg)).cpu().numpy()
                from sklearn.metrics import roc_auc_score
                preds = np.concatenate([pos_pred, neg_pred])
                labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
                val_auc = roc_auc_score(labels, preds)

            if verbose:
                print(f"    Epoch {epoch:4d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

            if val_auc > best_val_score:
                best_val_score = val_auc
                best_emb = z_val.clone()

    if verbose:
        print(f"    Best Val AUC: {best_val_score:.4f}")

    return best_emb, best_val_score


#  Generic ranking evaluator 

def eval_model_ranking(embeddings, user_histories, prod_gids, gid_to_pid, node_id_map, ks):
    """Evaluate embeddings with leave-last-out ranking."""
    prod_embs = embeddings[prod_gids]
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

        scores = torch.sigmoid((prod_embs * user_emb.unsqueeze(0)).sum(dim=1))
        scores = scores.cpu().numpy()

        candidates = []
        for i, gid in enumerate(prod_gids):
            pid = gid_to_pid.get(gid)
            if pid is None or pid in train_pids:
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


#  Multi-seed runner 

def run_multiseed(data_dir, n_seeds, epochs, ks, verbose=False):
    seeds = [42, 123, 256, 512, 1024, 2048, 4096, 7777, 9999, 31415][:n_seeds]

    # Build graph once
    builder = BeautyGraphBuilder(data_dir)
    graph_data, node_id_map, node_types = builder.build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    user_histories = build_user_histories(interactions_df)

    prod_gids = builder.get_product_node_ids()
    all_product_ids = set()
    gid_to_pid = {}
    for gid in prod_gids:
        pid = builder.global_id_to_product_id(gid)
        if pid is not None:
            all_product_ids.add(pid)
            gid_to_pid[gid] = pid

    #  Baselines (deterministic, run once) 
    print("\n  Baselines (no variance):")
    res_random = eval_random_baseline(user_histories, all_product_ids, ks)
    res_pop = eval_popularity_baseline(user_histories, interactions_df, all_product_ids, ks)
    print(f"    Random      — Recall@10: {res_random['Recall@10']:.4f}")
    print(f"    Popularity  — Recall@10: {res_pop['Recall@10']:.4f}")

    #  VGAE multi-seed 
    print(f"\n  VGAE (BPR, beta=0.1) — {n_seeds} seeds:")
    vgae_results = []
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_edges, val_edges, test_edges = _split_edges(graph_data.edge_index, seed=seed)
        in_ch = graph_data.x.size(1)
        model = VGAEBeauty(in_ch, 64, 32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        x = graph_data.x.to(device)
        t_edges = train_edges.to(device)

        best_auc = 0.0
        best_emb = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            z = model.encode(x, t_edges)
            neg = negative_sampling(t_edges, graph_data.num_nodes, t_edges.size(1))
            loss = model.total_loss(z, t_edges, neg, beta=0.1)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    z_v = model.encode(x, t_edges)
                    v_e = val_edges.to(device)
                    v_neg = negative_sampling(v_e, graph_data.num_nodes, v_e.size(1))
                    from sklearn.metrics import roc_auc_score
                    pp = torch.sigmoid(model.decode(z_v, v_e)).cpu().numpy()
                    np_ = torch.sigmoid(model.decode(z_v, v_neg)).cpu().numpy()
                    preds = np.concatenate([pp, np_])
                    labels = np.concatenate([np.ones(len(pp)), np.zeros(len(np_))])
                    auc = roc_auc_score(labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_emb = z_v.clone()

        res = eval_model_ranking(best_emb, user_histories, prod_gids, gid_to_pid, node_id_map, ks)
        res["Val_AUC"] = best_auc
        vgae_results.append(res)
        print(f"    Seed {seed:>5} — AUC: {best_auc:.4f} | Rec@10: {res['Recall@10']:.4f} | NDCG@10: {res['NDCG@10']:.4f}")

    #  LightGCN multi-seed 
    print(f"\n  LightGCN (BPR, 3 layers) — {n_seeds} seeds:")
    lgcn_results = []
    for i, seed in enumerate(seeds):
        emb, val_auc = train_lightgcn(graph_data, epochs=epochs, lr=0.005,
                                       emb_dim=64, num_layers=3, seed=seed,
                                       verbose=False)
        res = eval_model_ranking(emb, user_histories, prod_gids, gid_to_pid, node_id_map, ks)
        res["Val_AUC"] = val_auc
        lgcn_results.append(res)
        print(f"    Seed {seed:>5} — AUC: {val_auc:.4f} | Rec@10: {res['Recall@10']:.4f} | NDCG@10: {res['NDCG@10']:.4f}")

    #  Summary table 
    print(f"\n{'='*75}")
    print(f"  SUMMARY — Mean ± Std over {n_seeds} seeds")
    print(f"{'='*75}\n")

    header = f"  {'Method':<14}"
    for k in ks:
        header += f" {'Rec@'+str(k):>12} {'NDCG@'+str(k):>12}"
    header += f" {'AUC':>12}"
    print(header)
    print(f"  {''*14}" + f" {''*12}" * (len(ks) * 2 + 1))

    # Baselines
    print(f"  {'Random':<14}", end="")
    for k in ks:
        print(f" {res_random[f'Recall@{k}']:>12.4f} {res_random[f'NDCG@{k}']:>12.4f}", end="")
    print(f" {'—':>12}")

    print(f"  {'Popularity':<14}", end="")
    for k in ks:
        print(f" {res_pop[f'Recall@{k}']:>12.4f} {res_pop[f'NDCG@{k}']:>12.4f}", end="")
    print(f" {'—':>12}")

    # VGAE
    print(f"  {'VGAE(BPR)':<14}", end="")
    for k in ks:
        vals_r = [r[f"Recall@{k}"] for r in vgae_results]
        vals_n = [r[f"NDCG@{k}"] for r in vgae_results]
        print(f" {np.mean(vals_r):.4f}±{np.std(vals_r):.4f}", end="")
        print(f" {np.mean(vals_n):.4f}±{np.std(vals_n):.4f}", end="")
    aucs = [r["Val_AUC"] for r in vgae_results]
    print(f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}")

    # LightGCN
    print(f"  {'LightGCN':<14}", end="")
    for k in ks:
        vals_r = [r[f"Recall@{k}"] for r in lgcn_results]
        vals_n = [r[f"NDCG@{k}"] for r in lgcn_results]
        print(f" {np.mean(vals_r):.4f}±{np.std(vals_r):.4f}", end="")
        print(f" {np.mean(vals_n):.4f}±{np.std(vals_n):.4f}", end="")
    aucs = [r["Val_AUC"] for r in lgcn_results]
    print(f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}")

    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed variance analysis")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--Ks", default="5,10,20")
    args = parser.parse_args()

    ks = [int(k) for k in args.Ks.split(",")]
    run_multiseed(args.data_dir, args.seeds, args.epochs, ks)
