"""VGAE training pipeline with edge-split evaluation."""

import os
import torch
import numpy as np
from torch_geometric.utils import negative_sampling

from .vgae_model import VGAEBeauty


def _split_edges(edge_index, val_ratio=0.10, test_ratio=0.10, seed=42):
    """Randomly partition edges into train / val / test."""
    torch.manual_seed(seed)
    n = edge_index.size(1)
    perm = torch.randperm(n)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return edge_index[:, train_idx], edge_index[:, val_idx], edge_index[:, test_idx]


def train_vgae(graph_data, hidden_channels=64, out_channels=32, epochs=200,
               lr=0.01, beta=1.0, model_save_path="models/vgae_beauty.pt",
               verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_edges, val_edges, test_edges = _split_edges(graph_data.edge_index)

    in_channels = graph_data.x.size(1)
    model = VGAEBeauty(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = graph_data.x.to(device)
    all_edges = graph_data.edge_index.to(device)
    t_edges = train_edges.to(device)

    best_val_auc = 0.0
    best_state = None
    best_embeddings = None

    if verbose:
        print(f"\n  VGAE Training | Nodes: {graph_data.num_nodes} | Edges: {graph_data.num_edges}")
        print(f"  Features: {in_channels} | Hidden: {hidden_channels} | Latent: {out_channels} | Device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, t_edges)
        neg_edges = negative_sampling(t_edges, num_nodes=graph_data.num_nodes, num_neg_samples=t_edges.size(1))
        loss = model.total_loss(z, t_edges, neg_edges, beta=beta)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_val = model.encode(x, t_edges)
                v_edges = val_edges.to(device)
                v_neg = negative_sampling(v_edges, graph_data.num_nodes, v_edges.size(1))
                val_auc, val_ap = model.test(z_val, v_edges, v_neg)

            if verbose:
                print(f"  Epoch {epoch:4d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f} | Val AP {val_ap:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "in_channels": in_channels,
                    "hidden_channels": hidden_channels,
                    "out_channels": out_channels,
                    "val_auc": val_auc,
                }
                best_embeddings = z_val.cpu()

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save(best_state, model_save_path)
    if verbose:
        print(f"\n  Best model saved -> {model_save_path} (Val AUC: {best_val_auc:.4f})")

    model.load_state_dict({k: v.to(device) for k, v in best_state["model_state_dict"].items()})
    model.eval()
    with torch.no_grad():
        z_final = model.encode(x, all_edges)
        te_edges = test_edges.to(device)
        te_neg = negative_sampling(te_edges, graph_data.num_nodes, te_edges.size(1))
        test_auc, test_ap = model.test(z_final, te_edges, te_neg)

    if verbose:
        print(f"  Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}\n")

    return model.cpu(), best_embeddings
