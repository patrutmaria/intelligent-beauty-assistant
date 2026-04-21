"""
VGAE for beauty product recommendation.
Encoder: 2-layer GCN -> (mu, logstd) | Decoder: inner-product | Loss: BCE + beta*KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


class GCNEncoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VGAEBeauty(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self._mu = None
        self._logstd = None

    def encode(self, x, edge_index):
        self._mu, self._logstd = self.encoder(x, edge_index)
        self._logstd = self._logstd.clamp(max=10)
        return self._reparametrize(self._mu, self._logstd)

    def _reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * logstd.exp()
        return mu

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        return torch.sigmoid(z @ z.t())

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        EPS = 1e-15
        pos_loss = -torch.log(torch.sigmoid(self.decode(z, pos_edge_index)) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                pos_edge_index, num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1))
        neg_loss = -torch.log(1 - torch.sigmoid(self.decode(z, neg_edge_index)) + EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu=None, logstd=None):
        mu = self._mu if mu is None else mu
        logstd = self._logstd if logstd is None else logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))

    def total_loss(self, z, pos_edge_index, neg_edge_index=None, beta=1.0):
        return self.recon_loss(z, pos_edge_index, neg_edge_index) + beta * self.kl_loss()

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        import numpy as np
        pos_pred = torch.sigmoid(self.decode(z, pos_edge_index)).cpu().numpy()
        neg_pred = torch.sigmoid(self.decode(z, neg_edge_index)).cpu().numpy()
        preds = np.concatenate([pos_pred, neg_pred])
        labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
        return roc_auc_score(labels, preds), average_precision_score(labels, preds)
