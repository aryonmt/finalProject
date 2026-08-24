from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass
class SplitArrays:
    """Edge list for one split: `pairs` is (N, 2) int64, `labels` is (N,) float/int."""

    pairs: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    """Leak-free graph + splits used by every trainer and metric helper.

    `adj_train` and the Laplacian tensors (`f_orig`, `f_skip_*`, `f_3hop`) are
    built from **train positives only**. Val/test edges never enter the graph.
    For bipartite datasets, sources occupy `[0, n_source)` and targets occupy
    `[n_source, n_nodes)`.
    """

    name: str
    n_nodes: int
    n_source: int
    n_target: int
    bipartite: bool
    features: torch.Tensor
    adj_train: sp.csr_matrix
    f_orig: torch.Tensor
    f_skip_bin: torch.Tensor
    f_skip_weighted: torch.Tensor
    f_3hop: torch.Tensor
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    known_positives: set[tuple[int, int]]
    input_type: str
    idx_map: dict[str, int] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


class PairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, labels: np.ndarray):
        self.pairs = torch.as_tensor(pairs, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[index], self.labels[index]
