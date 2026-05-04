"""
    /opt/anaconda3/bin/python evaluation/eval_experiments.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph.graph_builder import BeautyGraphBuilder
from graph.vgae_model import VGAEBeauty, GCNEncoder
from graph.trainer import _split_edges
from evaluation.eval_ranking import build_user_histories, recall_at_k, ndcg_at_k, hit_rate_at_k
from evaluation.eval_multiseed import eval_model_ranking, train_lightgcn, LightGCN


#  Exp 2: Deterministic encoder (no reparameterization noise) 

class VGAEDeterministic(VGAEBeauty):
    """VGAE that uses z=mu during training (no sampling noise)."""

    def _reparametrize(self, mu, logstd):
        # Always return mu — no stochastic sampling
        return mu


#  Exp 3: Multi-layer aggregation encoder 

class GCNEncoderMultiLayer(torch.nn.Module):
    """GCN encoder with LightGCN-style layer aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.proj_input = torch.nn.Linear(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        # H0: project input to hidden dim
        h0 = self.proj_input(x)
        # H1: first GCN layer
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.3, training=self.training)
        # H2: second GCN layer
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=0.3, training=self.training)
        # Aggregate: mean of all layers
        h_agg = (h0 + h1 + h2) / 3.0
        return self.conv_mu(h_agg, edge_index), self.conv_logstd(h_agg, edge_index)


class VGAEMultiLayer(VGAEBeauty):
    """VGAE with multi-layer aggregation encoder."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super(VGAEBeauty, self).__init__()
        self.encoder = GCNEncoderMultiLayer(in_channels, hidden_channels, out_channels)
        self._mu = None
        self._logstd = None


class VGAEMultiLayerDet(VGAEMultiLayer):
    """Multi-layer + deterministic."""

    def _reparametrize(self, mu, logstd):
        return mu


#  Generic training loop 

def train_model(model, graph_data, epochs=500, lr=0.005, beta=0.001,
                beta_warmup=50, seed=42):
    """Train any VGAE variant, return best embeddings and val AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_edges, val_edges, test_edges = _split_edges(graph_data.edge_index, seed=seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = graph_data.x.to(device)
    t_edges = train_edges.to(device)

    best_auc = 0.0
    best_emb = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, t_edges)
        neg = negative_sampling(t_edges, graph_data.num_nodes, t_edges.size(1))

        cur_beta = beta * min(1.0, epoch / beta_warmup) if beta_warmup > 0 else beta
        loss = model.total_loss(z, t_edges, neg, beta=cur_beta)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                z_v = model.encode(x, t_edges)
                v_e = val_edges.to(device)
                v_neg = negative_sampling(v_e, graph_data.num_nodes, v_e.size(1))
                pp = torch.sigmoid(model.decode(z_v, v_e)).cpu().numpy()
                np_ = torch.sigmoid(model.decode(z_v, v_neg)).cpu().numpy()
                preds = np.concatenate([pp, np_])
                labels = np.concatenate([np.ones(len(pp)), np.zeros(len(np_))])
                auc = roc_auc_score(labels, preds)
                if auc > best_auc:
                    best_auc = auc
                    best_emb = z_v.clone()

    return best_emb, best_auc


#  Exp 4: LightGCN-initialized VGAE 

def train_vgae_from_lightgcn(graph_data, epochs_finetune=100, lr=0.001,
                              beta=0.0001, seed=42):
    """Train LightGCN first, then fine-tune a deterministic VGAE from its embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train LightGCN
    lgcn_emb, lgcn_auc = train_lightgcn(graph_data, epochs=500, lr=0.005,
                                          emb_dim=64, num_layers=3, seed=seed)

    # Initialize VGAE with LightGCN-aware weights
    in_ch = graph_data.x.size(1)
    model = VGAEDeterministic(in_ch, 64, 32).to(device)

    train_edges, val_edges, _ = _split_edges(graph_data.edge_index, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = graph_data.x.to(device)
    t_edges = train_edges.to(device)

    best_auc = 0.0
    best_emb = None

    for epoch in range(1, epochs_finetune + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, t_edges)
        neg = negative_sampling(t_edges, graph_data.num_nodes, t_edges.size(1))

        # BPR + small KL + alignment loss to LightGCN embeddings
        bpr = model.bpr_loss(z, t_edges, neg)
        kl = model.kl_loss()
        # Alignment: MSE between VGAE embeddings and LightGCN embeddings (projected)
        # Project LightGCN 64-dim to VGAE 32-dim via simple truncation
        lgcn_target = lgcn_emb[:, :32].to(device)
        align = F.mse_loss(z, lgcn_target)

        loss = bpr + beta * kl + 0.1 * align
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                z_v = model.encode(x, t_edges)
                v_e = val_edges.to(device)
                v_neg = negative_sampling(v_e, graph_data.num_nodes, v_e.size(1))
                pp = torch.sigmoid(model.decode(z_v, v_e)).cpu().numpy()
                np_ = torch.sigmoid(model.decode(z_v, v_neg)).cpu().numpy()
                preds_arr = np.concatenate([pp, np_])
                labels = np.concatenate([np.ones(len(pp)), np.zeros(len(np_))])
                auc = roc_auc_score(labels, preds_arr)
                if auc > best_auc:
                    best_auc = auc
                    best_emb = z_v.clone()

    return best_emb, best_auc


#  Main experiment runner 

def main():
    data_dir = "data/"
    seeds = [42, 123, 256]
    ks = [5, 10, 20]
    epochs = 500

    builder = BeautyGraphBuilder(data_dir)
    graph_data, node_id_map, node_types = builder.build()

    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    user_histories = build_user_histories(interactions_df)

    prod_gids = builder.get_product_node_ids()
    gid_to_pid = {}
    for gid in prod_gids:
        pid = builder.global_id_to_product_id(gid)
        if pid is not None:
            gid_to_pid[gid] = pid

    in_ch = graph_data.x.size(1)
    log_rows = []

    def run_exp(name, model_factory, beta, beta_warmup, custom_train=None):
        print(f"\n{''*60}")
        print(f"  {name}")
        print(f"{''*60}")
        results = []
        for seed in seeds:
            if custom_train:
                emb, auc = custom_train(graph_data, seed=seed)
            else:
                model = model_factory()
                emb, auc = train_model(model, graph_data, epochs=epochs, lr=0.005,
                                       beta=beta, beta_warmup=beta_warmup, seed=seed)
            res = eval_model_ranking(emb, user_histories, prod_gids, gid_to_pid, node_id_map, ks)
            res["Val_AUC"] = auc
            results.append(res)
            print(f"    Seed {seed:>5} — AUC: {auc:.4f} | Rec@5: {res['Recall@5']:.4f} | "
                  f"Rec@10: {res['Recall@10']:.4f} | Rec@20: {res['Recall@20']:.4f} | "
                  f"NDCG@10: {res['NDCG@10']:.4f}")
            log_rows.append({
                "experiment": name, "seed": seed, "epochs": epochs,
                "beta": beta, "beta_warmup": beta_warmup,
                "Val_AUC": auc,
                "Recall@5": res["Recall@5"], "Recall@10": res["Recall@10"],
                "Recall@20": res["Recall@20"], "NDCG@10": res["NDCG@10"],
                "NDCG@20": res["NDCG@20"], "HitRate@10": res["HitRate@10"],
            })

        # Summary
        mean_r10 = np.mean([r["Recall@10"] for r in results])
        std_r10 = np.std([r["Recall@10"] for r in results])
        mean_auc = np.mean([r["Val_AUC"] for r in results])
        print(f"  >> Mean Recall@10: {mean_r10:.4f} ± {std_r10:.4f} | AUC: {mean_auc:.4f}")
        return mean_r10

    #  Exp 1: KL Annealing 
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: KL Annealing (β_max=0.001, warmup=50)")
    print("=" * 60)
    r10_exp1 = run_exp(
        "Exp1_KL_anneal",
        lambda: VGAEBeauty(in_ch, 64, 32),
        beta=0.001, beta_warmup=50,
    )

    #  Exp 2: Deterministic encoder 
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Deterministic encoder (z=μ, β=0.001)")
    print("=" * 60)
    r10_exp2 = run_exp(
        "Exp2_deterministic",
        lambda: VGAEDeterministic(in_ch, 64, 32),
        beta=0.001, beta_warmup=50,
    )

    #  Exp 3: Multi-layer aggregation 
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Multi-layer aggregation + deterministic")
    print("=" * 60)
    r10_exp3 = run_exp(
        "Exp3_multilayer_det",
        lambda: VGAEMultiLayerDet(in_ch, 64, 32),
        beta=0.001, beta_warmup=50,
    )

    #  Exp 4: LightGCN initialization 
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: LightGCN-initialized VGAE")
    print("=" * 60)
    r10_exp4 = run_exp(
        "Exp4_lgcn_init",
        None, beta=0.0001, beta_warmup=0,
        custom_train=lambda gd, seed: train_vgae_from_lightgcn(gd, seed=seed),
    )

    #  LightGCN reference 
    print("\n" + "=" * 60)
    print("  REFERENCE: LightGCN (upper bound)")
    print("=" * 60)
    for seed in seeds:
        emb, auc = train_lightgcn(graph_data, epochs=epochs, seed=seed)
        res = eval_model_ranking(emb, user_histories, prod_gids, gid_to_pid, node_id_map, ks)
        print(f"    Seed {seed:>5} — AUC: {auc:.4f} | Rec@10: {res['Recall@10']:.4f}")
        log_rows.append({
            "experiment": "LightGCN_ref", "seed": seed, "epochs": epochs,
            "beta": 0, "beta_warmup": 0, "Val_AUC": auc,
            "Recall@5": res["Recall@5"], "Recall@10": res["Recall@10"],
            "Recall@20": res["Recall@20"], "NDCG@10": res["NDCG@10"],
            "NDCG@20": res["NDCG@20"], "HitRate@10": res["HitRate@10"],
        })

    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\n  Results saved to {log_path}")

    print(f"\n{'='*60}")
    print(f"  DECISION SUMMARY")
    print(f"{'='*60}")
    print(f"  Exp1 (KL anneal):      Recall@10 = {r10_exp1:.4f}  {'✓ SUCCESS' if r10_exp1 >= 0.05 else '→ continue' if r10_exp1 >= 0.025 else '✗ fail'}")
    print(f"  Exp2 (deterministic):  Recall@10 = {r10_exp2:.4f}  {'✓ SUCCESS' if r10_exp2 >= 0.05 else '→ continue' if r10_exp2 >= 0.025 else '✗ fail'}")
    print(f"  Exp3 (multi-layer):    Recall@10 = {r10_exp3:.4f}  {'✓ SUCCESS' if r10_exp3 >= 0.05 else '→ continue' if r10_exp3 >= 0.025 else '✗ fail'}")
    print(f"  Exp4 (LightGCN init):  Recall@10 = {r10_exp4:.4f}  {'✓ SUCCESS' if r10_exp4 >= 0.05 else '→ continue' if r10_exp4 >= 0.025 else '✗ fail'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
