from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGATv2Layer(nn.Module):
    """Edge-sparse GATv2: attention on COO edges, O(E) memory."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if out_features % heads != 0:
            raise ValueError("out_features must be divisible by heads")
        self.heads = heads
        self.d_k = out_features // heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.w_src = nn.Linear(in_features, out_features, bias=False)
        self.w_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(heads, self.d_k))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_src.weight)
        nn.init.xavier_uniform_(self.w_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)
        nn.init.zeros_(self.bias)

    def _linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        if x.is_sparse:
            return torch.sparse.mm(x, layer.weight.t())
        return layer(x)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        if not adj_sparse.is_sparse:
            raise TypeError("SparseGATv2Layer expects a sparse COO adjacency")
        adj = adj_sparse.coalesce()
        n_nodes = x.size(0)
        h_src = self._linear(x, self.w_src).view(n_nodes, self.heads, self.d_k)
        h_dst = self._linear(x, self.w_dst).view(n_nodes, self.heads, self.d_k)

        indices = adj.indices()
        if indices.numel() == 0:
            return self.bias.expand(n_nodes, -1).clone()

        u_idx, v_idx = indices[0], indices[1]
        edge_feat = F.leaky_relu(h_src[u_idx] + h_dst[v_idx], negative_slope=self.negative_slope)
        score = (edge_feat * self.attn_vec.unsqueeze(0)).sum(dim=-1)
        score = score - score.max(dim=0, keepdim=True).values

        exp_score = torch.exp(score)
        denom = torch.zeros(n_nodes, self.heads, device=x.device, dtype=exp_score.dtype)
        denom.index_add_(0, u_idx, exp_score)
        alpha = exp_score / denom[u_idx].clamp_min(1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(n_nodes, self.heads, self.d_k, device=x.device, dtype=h_dst.dtype)
        out.index_add_(0, u_idx, h_dst[v_idx] * alpha.unsqueeze(-1))
        return out.reshape(n_nodes, self.heads * self.d_k) + self.bias


class SkipGATv2(nn.Module):
    """Gated original/skip encoder with sparse GATv2 instead of static Laplacian GCN."""

    skip_kind = "weighted"

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_dec: int,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__()
        self.dropout = dropout
        self.o_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.so_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.s_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.os_gat1 = SparseGATv2Layer(nhid1, nhid1, heads=heads, dropout=dropout)
        self.gate1 = nn.Linear(nhid1 * 2, 1)
        self.o_gat2 = SparseGATv2Layer(nhid1, nhid2, heads=heads, dropout=dropout)
        self.so_gat2 = SparseGATv2Layer(nhid1, nhid2, heads=heads, dropout=dropout)
        self.gate2 = nn.Linear(nhid2 * 2, 1)
        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor) -> torch.Tensor:
        h_o1 = self.o_gat1(x, adj)
        h_so1 = self.so_gat1(x, adj_skip)
        g1 = torch.sigmoid(self.gate1(torch.cat((h_o1, h_so1), dim=-1)))
        o1 = F.elu(g1 * h_o1 + (1.0 - g1) * h_so1)
        s1 = F.elu(self.s_gat1(x, adj_skip) + self.os_gat1(o1, adj))
        o1 = F.dropout(o1, self.dropout, training=self.training)
        s1 = F.dropout(s1, self.dropout, training=self.training)
        h_o2 = self.o_gat2(o1, adj)
        h_so2 = self.so_gat2(s1, adj_skip)
        g2 = torch.sigmoid(self.gate2(torch.cat((h_o2, h_so2), dim=-1)))
        return g2 * h_o2 + (1.0 - g2) * h_so2

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        e_u, e_v = emb[pairs[:, 0]], emb[pairs[:, 1]]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adj_skip is None:
            raise ValueError("SkipGATv2 requires adj_skip")
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb
