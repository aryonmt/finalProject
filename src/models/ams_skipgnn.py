from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SparseGraphConvolution


class AMSSkipGNN(nn.Module):
    """SkipGNN plus resource-allocation skip, gated fusion, and a 4-way decoder."""

    skip_kind = "weighted"

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_dec: int,
        dropout: float = 0.5,
        use_gate: bool = True,
        four_way_decoder: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_gate = use_gate
        self.four_way_decoder = four_way_decoder

        self.o_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1_o = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.o_gc1_s = SparseGraphConvolution(nhid1, nhid1)
        self.gate1 = nn.Linear(nhid1 * 2, 1)

        self.o_gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.s_gc2_o = SparseGraphConvolution(nhid1, nhid2)
        self.gate2 = nn.Linear(nhid2 * 2, 1)

        dec_in = nhid2 * 4 if four_way_decoder else nhid2 * 2
        self.decoder_fc1 = nn.Linear(dec_in, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def _fuse(self, a: torch.Tensor, b: torch.Tensor, gate: nn.Linear) -> torch.Tensor:
        if self.use_gate:
            g = torch.sigmoid(gate(torch.cat((a, b), dim=-1)))
            return g * a + (1.0 - g) * b
        return a + b

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor) -> torch.Tensor:
        h_orig1 = self.o_gc1(x, adj)
        h_skip_cross1 = self.s_gc1_o(x, adj_skip)
        o1 = F.relu(self._fuse(h_orig1, h_skip_cross1, self.gate1))
        s1 = F.relu(self.s_gc1(x, adj_skip) + self.o_gc1_s(o1, adj))
        o1 = F.dropout(o1, self.dropout, training=self.training)
        s1 = F.dropout(s1, self.dropout, training=self.training)
        h_orig2 = self.o_gc2(o1, adj)
        h_skip2 = self.s_gc2_o(s1, adj_skip)
        return self._fuse(h_orig2, h_skip2, self.gate2)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        e_u, e_v = emb[u], emb[v]
        if self.four_way_decoder:
            z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        else:
            z = torch.cat((e_u, e_v), dim=-1)
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
            raise ValueError("AMSSkipGNN requires adj_skip")
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb


def make_ablation(variant: str, **kwargs) -> nn.Module:
    """Stepwise AMS ablations: binary skip → weighted skip → gate → 4-way decoder."""
    variant = variant.lower()
    if variant in {"0", "ablation_0_skipgnn", "skipgnn"}:
        from src.models.skipgnn import SkipGNNBaseline

        model = SkipGNNBaseline(**kwargs)
        model.skip_kind = "binary"
        return model
    if variant in {"1", "ablation_1_weighted", "weighted"}:
        from src.models.skipgnn import SkipGNNBaseline

        model = SkipGNNBaseline(**kwargs)
        model.skip_kind = "weighted"
        return model
    if variant in {"2", "ablation_2_gated", "gated"}:
        return AMSSkipGNN(use_gate=True, four_way_decoder=False, **kwargs)
    if variant in {"3", "ablation_3_full", "full", "ams"}:
        return AMSSkipGNN(use_gate=True, four_way_decoder=True, **kwargs)
    raise ValueError(f"Unknown ablation variant: {variant}")
