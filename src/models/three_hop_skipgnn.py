from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.normalization import build_three_hop_return
from src.models.layers import SparseGraphConvolution


class ThreeHopSkipGNN(nn.Module):
    """Fuses 1-hop, 2-hop (same-type skip), and 3-hop return walks with path attention."""

    skip_kind = "three_hop_bundle"

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_dec: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.gc_1hop_1 = SparseGraphConvolution(nfeat, nhid1)
        self.gc_2hop_1 = SparseGraphConvolution(nfeat, nhid1)
        self.gc_3hop_1 = SparseGraphConvolution(nfeat, nhid1)
        self.path_attn_vec1 = nn.Parameter(torch.empty(3, nhid1))
        self.gc_1hop_2 = SparseGraphConvolution(nhid1, nhid2)
        self.gc_2hop_2 = SparseGraphConvolution(nhid1, nhid2)
        self.gc_3hop_2 = SparseGraphConvolution(nhid1, nhid2)
        self.path_attn_vec2 = nn.Parameter(torch.empty(3, nhid2))
        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.path_attn_vec1)
        nn.init.xavier_uniform_(self.path_attn_vec2)

    def _path_fuse(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h3: torch.Tensor,
        attn_vec: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.cat(
            (
                (h1 * attn_vec[0]).sum(dim=-1, keepdim=True),
                (h2 * attn_vec[1]).sum(dim=-1, keepdim=True),
                (h3 * attn_vec[2]).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        gamma = F.softmax(scores, dim=-1)
        return gamma[:, 0:1] * h1 + gamma[:, 1:2] * h2 + gamma[:, 2:3] * h3

    def encode(
        self,
        x: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ) -> torch.Tensor:
        fused1 = self._path_fuse(
            F.relu(self.gc_1hop_1(x, f1)),
            F.relu(self.gc_2hop_1(x, f2)),
            F.relu(self.gc_3hop_1(x, f3)),
            self.path_attn_vec1,
        )
        fused1 = F.dropout(fused1, self.dropout, training=self.training)
        return self._path_fuse(
            self.gc_1hop_2(fused1, f1),
            self.gc_2hop_2(fused1, f2),
            self.gc_3hop_2(fused1, f3),
            self.path_attn_vec2,
        )

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        e_u, e_v = emb[pairs[:, 0]], emb[pairs[:, 1]]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        f1: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adj_skip is None or not isinstance(adj_skip, (tuple, list)) or len(adj_skip) != 2:
            raise ValueError("ThreeHopSkipGNN requires adj_skip=(f_2hop, f_3hop)")
        emb = self.encode(x, f1, adj_skip[0], adj_skip[1])
        return self.decode(emb, pairs), emb


def build_three_hop_matrices(adj):
    """Kept for the spec import path; loader uses normalization.build_three_hop_return."""
    return build_three_hop_return(adj)
