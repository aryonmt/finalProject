from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SparseGraphConvolution


class SkipGNNBaseline(nn.Module):
    """Bug-free replica of Huang et al. SkipGNN (binary skip, additive fusion, concat decoder)."""

    skip_kind = "binary"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.o_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1_o = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.o_gc1_s = SparseGraphConvolution(nhid1, nhid1)
        self.o_gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.s_gc2_o = SparseGraphConvolution(nhid1, nhid2)
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_dec)
        self.decoder2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor) -> torch.Tensor:
        o_x = F.relu(self.o_gc1(x, adj) + self.s_gc1_o(x, adj_skip))
        s_x = F.relu(self.s_gc1(x, adj_skip) + self.o_gc1_s(o_x, adj))
        o_x = F.dropout(o_x, self.dropout, training=self.training)
        s_x = F.dropout(s_x, self.dropout, training=self.training)
        return self.o_gc2(o_x, adj) + self.s_gc2_o(s_x, adj_skip)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        feat = torch.cat((emb[u], emb[v]), dim=-1)
        return self.decoder2(F.relu(self.decoder1(feat))).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adj_skip is None:
            raise ValueError("SkipGNNBaseline requires adj_skip")
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb
