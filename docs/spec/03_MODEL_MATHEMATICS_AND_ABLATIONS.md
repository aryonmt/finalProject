# Document 03: Model Mathematics, Layer Specifications & Ablation Hierarchy

---

## 1. Weighted Topological Skip Graph Formulation ($A_s$)

### 1.1 Mathematical Definition
Let $A \in \{0, 1\}^{N \times N}$ be the unweighted adjacency matrix constructed strictly from training interactions ($\mathcal{E}_{train}$). Let $d_k = \sum_j A_{kj}$ denote the degree of node $k$.

The continuous degree-weighted **Resource Allocation Skip Matrix** $W \in \mathbb{R}^{N \times N}$ is defined as:
$$W_{uv} = \sum_{k \in \mathcal{N}(u) \cap \mathcal{N}(v)} \frac{A_{uk} A_{kv}}{\max(1, d_k)}$$

In sparse matrix algebra, with $D = \text{diag}(\max(1, d_1), \dots, \max(1, d_N))$:
$$W = A \cdot D^{-1} \cdot A^T$$

To eliminate artificial 2-hop self-loops, the diagonal is set to zero:
$$A_s = W - \text{diag}(W)$$

### 1.2 Symmetric Laplacian Normalization
Both original graph $A$ and skip graph $A_s$ are symmetrically normalized:
$$\tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_{j=1}^N \tilde{A}_{ij}, \quad F = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
$$\tilde{D}_{s, ii} = \max\left(10^{-5}, \sum_{j=1}^N (A_s)_{ij}\right), \quad F_s = \tilde{D}_s^{-1/2} A_s \tilde{D}_s^{-1/2}$$

---

## 2. Neural Architecture Catalog

### 2.1 Model 1: `StandardGCN` (Baseline 1)
A standard 2-layer Graph Convolutional Network operating solely on the original graph $F$, serving as the primary non-skip deep learning baseline.

- **Propagation:**
  $$H^{(1)} = \text{ReLU}(F X W^{(0)})$$
  $$\hat{H}^{(1)} = \text{Dropout}(H^{(1)}, p)$$
  $$E = F \hat{H}^{(1)} W^{(1)}$$
- **Decoder:**
  $$\text{Logit}(i, j) = W_{d2} \cdot \text{ReLU}(W_{d1} [E_i \parallel E_j] + b_{d1}) + b_{d2}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import SparseGraphConvolution

class StandardGCN(nn.Module):
    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5):
        super().__init__()
        self.gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_dec)
        self.decoder2 = nn.Linear(nhid_dec, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        emb = self.gc2(h1, adj)
        
        u, v = pairs[:, 0], pairs[:, 1]
        feat = torch.cat((emb[u], emb[v]), dim=-1)
        logits = self.decoder2(F.relu(self.decoder1(feat)))
        return logits.squeeze(-1), emb
```

---

### 2.2 Model 2: `SkipGNNBaseline` (Baseline 2 - Bug-Free Replica)
Bug-free reimplementation of Huang et al. using unweighted binary skip matrix $A_s^{bin} = \text{sign}(A A^T)$, isotropic additive fusion, and a linear concatenation decoder.

- **Layer 1:**
  $$H_{orig}^{(1)} = \text{ReLU}\left(F X W_o^{(0)} + F_s^{bin} X W_{so}^{(0)}\right)$$
  $$S^{(1)} = \text{ReLU}\left(F_s^{bin} X W_s^{(0)} + F H_{orig}^{(1)} W_{os}^{(0)}\right)$$
- **Layer 2:**
  $$\hat{H}_{orig}^{(1)} = \text{Dropout}(H_{orig}^{(1)}, p), \quad \hat{S}^{(1)} = \text{Dropout}(S^{(1)}, p)$$
  $$E = F \hat{H}_{orig}^{(1)} W_o^{(1)} + F_s^{bin} \hat{S}^{(1)} W_{s2o}^{(1)}$$
- **Decoder:**
  $$\text{Logit}(i, j) = W_{d2} \cdot \text{ReLU}(W_{d1} [E_i \parallel E_j] + b_{d1}) + b_{d2}$$

---

### 2.3 Model 3: `AMSSkipGNN` (Proposed Champion Model)
Combines degree-weighted skip propagation ($F_s$), Adaptive Gated Residual fusion ($g_1, g_2$), and a 4-Way Interactive Tensor Decoder.

```text
Input Features X (N x d_in)
    │
    ├─────────────────────────────┬─────────────────────────────┐
    ▼                             ▼                             ▼
GCN_o(X, F)                 GCN_so(X, F_s)                GCN_s(X, F_s)
    │                             │                             │
    └──────────────┬──────────────┘                             │
                   ▼                                            │
        Adaptive Gate 1 (g_1)                                   │
                   ▼                                            │
        O^(1) = g_1 ⊙ H_o + (1-g_1) ⊙ H_so                      │
                   │                                            │
                   ├──────────────────────┐                     │
                   │                      ▼                     │
                   │                GCN_os(O^(1), F)            │
                   │                      │                     │
                   │                      └──────────┬──────────┘
                   ▼                                 ▼
             Dropout(O^(1))                   S^(1) = ReLU(GCN_s + GCN_os)
                   │                                 │
                   ├──────────────┬──────────────┐   │
                   ▼              ▼              │   ▼
             GCN_o2(O^(1), F) GCN_s2o(S^(1), F_s)│ Dropout(S^(1))
                   │              │              │
                   └──────┬───────┘              │
                          ▼                      │
                Adaptive Gate 2 (g_2)            │
                          ▼                      │
                Embedding Matrix E ◄─────────────┘
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
Retrieve E_u (Source)                 Retrieve E_v (Target)
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
            4-Way Interaction Tensor:
      Z_uv = [E_u || E_v || E_u ⊙ E_v || |E_u - E_v|]
                          │
                          ▼
    Linear(4*d_hid2 -> d_dec) -> BatchNorm1d -> ReLU -> Dropout
                          │
                          ▼
             Linear(d_dec -> 1) -> Logit_uv
```

- **Propagation Equations:**
  $$H_{orig}^{(1)} = F X W_o^{(0)}, \quad H_{skip\_cross}^{(1)} = F_s X W_{so}^{(0)}$$
  $$g_1 = \sigma\left([H_{orig}^{(1)} \parallel H_{skip\_cross}^{(1)}] \cdot w_{g1} + b_{g1}\right) \in \mathbb{R}^{N \times 1}$$
  $$O^{(1)} = \text{ReLU}\left( g_1 \odot H_{orig}^{(1)} + (1 - g_1) \odot H_{skip\_cross}^{(1)} \right)$$
  $$S^{(1)} = \text{ReLU}\left( F_s X W_s^{(0)} + F O^{(1)} W_{os}^{(0)} \right)$$
  $$\hat{O}^{(1)} = \text{Dropout}(O^{(1)}, p), \quad \hat{S}^{(1)} = \text{Dropout}(S^{(1)}, p)$$
  $$H_{orig}^{(2)} = F \hat{O}^{(1)} W_o^{(1)}, \quad H_{skip2} = F_s \hat{S}^{(1)} W_{s2o}^{(1)}$$
  $$g_2 = \sigma\left([H_{orig}^{(2)} \parallel H_{skip2}] \cdot w_{g2} + b_{g2}\right) \in \mathbb{R}^{N \times 1}$$
  $$E = g_2 \odot H_{orig}^{(2)} + (1 - g_2) \odot H_{skip2} \in \mathbb{R}^{N \times d_2}$$

- **Decoder Equations:**
  $$Z_{uv} = \left[ E_u \;\parallel\; E_v \;\parallel\; (E_u \odot E_v) \;\parallel\; |E_u - E_v| \right] \in \mathbb{R}^{4 d_2}$$
  $$H_{dec} = \text{Dropout}\left(\text{ReLU}\left(\text{BatchNorm1d}(Z_{uv} W_{d1} + b_{d1})\right), p\right)$$
  $$\text{Logit}(u, v) = H_{dec} W_{d2} + b_{d2} \in \mathbb{R}$$

```python
class AMSSkipGNN(nn.Module):
    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        # Layer 1 operators
        self.o_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1_o = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.o_gc1_s = SparseGraphConvolution(nhid1, nhid1)
        self.gate1 = nn.Linear(nhid1 * 2, 1)

        # Layer 2 operators
        self.o_gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.s_gc2_o = SparseGraphConvolution(nhid1, nhid2)
        self.gate2 = nn.Linear(nhid2 * 2, 1)

        # 4-Way Tensor Decoder
        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def forward(self, x: torch.Tensor, f_orig: torch.Tensor, f_skip: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Layer 1
        h_orig1 = self.o_gc1(x, f_orig)
        h_skip_cross1 = self.s_gc1_o(x, f_skip)
        g1 = torch.sigmoid(self.gate1(torch.cat((h_orig1, h_skip_cross1), dim=-1)))
        o1 = F.relu(g1 * h_orig1 + (1.0 - g1) * h_skip_cross1)

        s1_cross = self.o_gc1_s(o1, f_orig)
        s1_direct = self.s_gc1(x, f_skip)
        s1 = F.relu(s1_direct + s1_cross)

        o1_drop = F.dropout(o1, self.dropout, training=self.training)
        s1_drop = F.dropout(s1, self.dropout, training=self.training)

        # Layer 2
        h_orig2 = self.o_gc2(o1_drop, f_orig)
        h_skip2 = self.s_gc2_o(s1_drop, f_skip)
        g2 = torch.sigmoid(self.gate2(torch.cat((h_orig2, h_skip2), dim=-1)))
        emb = g2 * h_orig2 + (1.0 - g2) * h_skip2

        # 4-Way Interaction Retrieval
        u, v = pairs[:, 0], pairs[:, 1]
        e_u, e_v = emb[u], emb[v]
        z_uv = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)

        # Decode
        h_dec = F.relu(self.decoder_bn(self.decoder_fc1(z_uv)))
        h_dec = F.dropout(h_dec, self.dropout, training=self.training)
        logits = self.decoder_fc2(h_dec).squeeze(-1)

        return logits, emb
```

---

## 3. Modular Ablation Ladder Specifications

To isolate the contribution of each architectural component, `AMSAblationLadder` provides explicit ablation variants:

| Model ID | Skip Graph Matrix ($A_s$) | Fusion Mechanism | Decoder Architecture |
| :--- | :--- | :--- | :--- |
| `Ablation_0_SkipGNN` | Binary: $\text{sign}(AA^T)$ | Unweighted Addition ($H_o + H_s$) | Concat Linear: $[E_u \parallel E_v]$ |
| `Ablation_1_Weighted` | Weighted Resource Allocation ($W$) | Unweighted Addition ($H_o + H_s$) | Concat Linear: $[E_u \parallel E_v]$ |
| `Ablation_2_Gated` | Weighted Resource Allocation ($W$) | Adaptive Gating ($g_1, g_2$) | Concat Linear: $[E_u \parallel E_v]$ |
| `Ablation_3_Full` | Weighted Resource Allocation ($W$) | Adaptive Gating ($g_1, g_2$) | 4-Way Tensor Decoder ($Z_{uv}$) |

---

## 4. Analytical Topological Heuristics (`src.models.heuristics`)

Implemented for strict non-parametric performance reference:

```python
import numpy as np
import scipy.sparse as sp

def compute_heuristic_scores(adj: sp.csr_matrix, pairs: np.ndarray, method: str = "resource_allocation") -> np.ndarray:
    """
    Computes heuristic link prediction scores for given (u, v) pairs.
    Supported methods: 'common_neighbors', 'jaccard', 'adamic_adar', 'resource_allocation'.
    """
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    scores = np.zeros(len(pairs), dtype=np.float32)

    for idx, (u, v) in enumerate(pairs):
        neighbors_u = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
        neighbors_v = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
        common = np.intersect1d(neighbors_u, neighbors_v, assume_unique=True)

        if len(common) == 0:
            scores[idx] = 0.0
            continue

        if method == "common_neighbors":
            scores[idx] = float(len(common))
        elif method == "jaccard":
            union_len = len(np.union1d(neighbors_u, neighbors_v))
            scores[idx] = float(len(common)) / max(1, union_len)
        elif method == "adamic_adar":
            weights = 1.0 / np.log(np.maximum(2.0, degrees[common]))
            scores[idx] = float(np.sum(weights))
        elif method == "resource_allocation":
            weights = 1.0 / np.maximum(1.0, degrees[common])
            scores[idx] = float(np.sum(weights))
        else:
            raise ValueError(f"Unknown heuristic method: {method}")

    return scores
```