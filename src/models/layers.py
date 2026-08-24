from __future__ import annotations

import torch
import torch.nn as nn


class SparseGraphConvolution(nn.Module):
    """GCN layer: `X W` then sparse `A @ support`. Accepts dense or sparse `X`."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weight)
        else:
            support = x @ self.weight
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output
