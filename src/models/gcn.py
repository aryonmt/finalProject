from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SparseGraphConvolution


class StandardGCN(nn.Module):
    """Two-layer GCN on the original graph only (no skip graph)."""

    skip_kind = "none"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5):
        super().__init__()
        self.gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_dec)
        self.decoder2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor | None = None) -> torch.Tensor:
        del adj_skip
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        return self.gc2(h1, adj)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        feat = torch.cat((emb[u], emb[v]), dim=-1)
        logits = self.decoder2(F.relu(self.decoder1(feat)))
        return logits.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb
