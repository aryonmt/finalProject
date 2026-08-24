from __future__ import annotations

import torch.nn as nn

from src.data.types import DatasetBundle
from src.models.ams_skipgnn import AMSSkipGNN, make_ablation
from src.models.gcn import StandardGCN
from src.models.skipgnn import SkipGNNBaseline


def build_model(
    name: str,
    bundle: DatasetBundle,
    hidden1: int = 64,
    hidden2: int = 64,
    decoder_hidden: int = 64,
    dropout: float = 0.5,
) -> nn.Module:
    kwargs = dict(
        nfeat=bundle.n_features,
        nhid1=hidden1,
        nhid2=hidden2,
        nhid_dec=decoder_hidden,
        dropout=dropout,
    )
    key = name.lower()
    if key in {"gcn", "standard_gcn", "baseline_gcn"}:
        return StandardGCN(**kwargs)
    if key in {"skipgnn", "skipgnn", "skipgnn_baseline", "baseline"}:
        return SkipGNNBaseline(**kwargs)
    if key in {"ams", "ams_skipgnn", "ams-skipgnn"}:
        return AMSSkipGNN(**kwargs)
    return make_ablation(key, **kwargs)


