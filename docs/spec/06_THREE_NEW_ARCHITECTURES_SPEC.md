# Document 06: Specification for Three New Neural Architectures

- **Module Targets:**
  1. `src/models/skip_gat.py` (`SkipGATv2`: Dynamic Cross-Graph Edge Attention)
  2. `src/models/three_hop_skipgnn.py` (`ThreeHopSkipGNN`: Multi-Order Bipartite-Aware Paths)
  3. `src/models/contrastive_skipgnn.py` (`ContrastiveSkipGNN`: Dual-View Self-Supervised InfoNCE Alignment)
- **Integration Targets:** `src/models/factory.py`, `src/training/trainer.py`, `src/data/loader.py`, `src/data/types.py`
- **Global Hyperparameters:** `lr: 0.001` (or `0.0005`), `epochs: 30` (or `45`), `hidden_dim: 64`, `decoder_hidden: 64`, `dropout: 0.5`

---

## 1. Architecture 1: `SkipGATv2` (Dynamic Cross-Graph Attention)

### 1.1 Mathematical Formulation
Static Laplacian matrix multiplication ($F X W$) is replaced by dynamic, learnable edge attention (GATv2).

For an edge $(i, j)$ in graph $\mathcal{G} \in \{A_{orig}, A_{skip}\}$, with input representations $h_i, h_j \in \mathbb{R}^{d_{in}}$:
$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left( W_l h_i + W_r h_j + b \right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

With $K=4$ attention heads ($d_k = 16 \implies d_{out} = 64$):
$$h_i^{(l+1)} = \Big\|_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W_v^k h_j^{(l)}$$

- **Layer 1 Propagation:**
  $$H_{orig}^{(1)} = \text{MultiHeadGATv2}_o(X, A)$$
  $$H_{skip\_cross}^{(1)} = \text{MultiHeadGATv2}_{so}(X, A_s)$$
  $$g_1 = \sigma\left( [H_{orig}^{(1)} \parallel H_{skip\_cross}^{(1)}] w_{g1} + b_{g1} \right)$$
  $$O^{(1)} = \text{ELU}\left( g_1 \odot H_{orig}^{(1)} + (1 - g_1) \odot H_{skip\_cross}^{(1)} \right)$$
  $$S^{(1)} = \text{ELU}\left( \text{MultiHeadGATv2}_s(X, A_s) + \text{MultiHeadGATv2}_{os}(O^{(1)}, A) \right)$$

- **Layer 2 Propagation:**
  $$E = g_2 \odot \text{MultiHeadGATv2}_{o2}(O^{(1)}, A) + (1 - g_2) \odot \text{MultiHeadGATv2}_{s2o}(S^{(1)}, A_s)$$

- **Decoder:** 4-Way Interactive Tensor Decoder `[E_u || E_v || E_u ⊙ E_v || |E_u - E_v|]` with `BatchNorm1d`.

### 1.2 Python Implementation (`src/models/skip_gat.py`)

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGATv2Layer(nn.Module):
    """Memory-efficient Sparse Multi-Head GATv2 Layer."""
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.5, negative_slope: float = 0.2):
        super().__init__()
        assert out_features % heads == 0, "out_features must be divisible by heads"
        self.heads = heads
        self.d_k = out_features // heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.w_src = nn.Linear(in_features, out_features, bias=False)
        self.w_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(self.heads, self.d_k))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_src.weight)
        nn.init.xavier_uniform_(self.w_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            h_src = torch.sparse.mm(x, self.w_src.weight.t())
            h_dst = torch.sparse.mm(x, self.w_dst.weight.t())
        else:
            h_src = self.w_src(x)
            h_dst = self.w_dst(x)

        n_nodes = x.size(0)
        h_src_h = h_src.view(n_nodes, self.heads, self.d_k)
        h_dst_h = h_dst.view(n_nodes, self.heads, self.d_k)

        indices = adj_sparse.indices()  # (2, E)
        u_idx, v_idx = indices[0], indices[1]

        edge_feat = h_src_h[u_idx] + h_dst_h[v_idx]  # (E, heads, d_k)
        edge_feat = F.leaky_relu(edge_feat, negative_slope=self.negative_slope)
        score = (edge_feat * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # (E, heads)

        score = score - score.max()
        exp_score = torch.exp(score)

        denom = torch.zeros(n_nodes, self.heads, device=x.device)
        denom.index_add_(0, u_idx, exp_score)
        denom = denom.clamp_min(1e-12)
        alpha = exp_score / denom[u_idx]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(n_nodes, self.heads, self.d_k, device=x.device)
        msg = h_dst_h[v_idx] * alpha.unsqueeze(-1)
        out.index_add_(0, u_idx, msg)
        return out.view(n_nodes, self.heads * self.d_k) + self.bias


class SkipGATv2(nn.Module):
    skip_kind = "weighted"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5, heads: int = 4):
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

        s1_cross = self.os_gat1(o1, adj)
        s1_direct = self.s_gat1(x, adj_skip)
        s1 = F.elu(s1_direct + s1_cross)

        o1_drop = F.dropout(o1, self.dropout, training=self.training)
        s1_drop = F.dropout(s1, self.dropout, training=self.training)

        h_o2 = self.o_gat2(o1_drop, adj)
        h_so2 = self.so_gat2(s1_drop, adj_skip)
        g2 = torch.sigmoid(self.gate2(torch.cat((h_o2, h_so2), dim=-1)))
        return g2 * h_o2 + (1.0 - g2) * h_so2

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        e_u, e_v = emb[u], emb[v]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor, adj_skip: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if adj_skip is None:
            raise ValueError("SkipGATv2 requires adj_skip")
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb
```

---

## 2. Architecture 2: `ThreeHopSkipGNN` (Multi-Order Bipartite Paths)

### 2.1 Mathematical Formulation
Constructs three structural walk operators:
1. **$F_1$ (1-Hop Direct):** Normalized $\tilde{A} = A + I_N$.
2. **$F_2$ (2-Hop Same-Type):** $W_2 = A D_{tgt}^{-1} A^T, \quad \text{diag}(W_2) = 0$.
3. **$F_3$ (3-Hop Target Return):** $W_3 = W_2 D_{src}^{-1} A, \quad F_3 = \tilde{D}_3^{-1/2} W_3 \tilde{D}_3^{-1/2}$.

- **Path Attention Gate:**
  $$\mathbf{s}_i = \left[ \mathbf{v}_1^T H_{1, i}, \;\mathbf{v}_2^T H_{2, i}, \;\mathbf{v}_3^T H_{3, i} \right] \in \mathbb{R}^3$$
  $$\gamma_i = \text{Softmax}(\mathbf{s}_i) \in \mathbb{R}^3$$
  $$H_{fused, i} = \gamma_{i, 1} H_{1, i} + \gamma_{i, 2} H_{2, i} + \gamma_{i, 3} H_{3, i}$$

### 2.2 Python Implementation (`src/models/three_hop_skipgnn.py`)

```python
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import SparseGraphConvolution

def build_three_hop_matrices(adj: sp.spmatrix) -> tuple[sp.coo_matrix, sp.coo_matrix]:
    """Builds continuous 2-hop RA and 3-hop return walk matrices."""
    csr = adj.tocsr().astype(np.float32)
    deg = np.asarray(csr.sum(axis=1)).flatten()
    inv_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    d_inv = sp.diags(inv_deg.astype(np.float32))

    w2 = csr @ d_inv @ csr.T
    w2 = w2.tolil()
    w2.setdiag(0)
    w2_csr = w2.tocsr()

    w3_csr = w2_csr @ d_inv @ csr
    return w2_csr.tocoo(), w3_csr.tocoo()


class ThreeHopSkipGNN(nn.Module):
    skip_kind = "three_hop_bundle"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5):
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

    def _path_fuse(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor, attn_vec: torch.Tensor) -> torch.Tensor:
        s1 = (h1 * attn_vec[0]).sum(dim=-1, keepdim=True)
        s2 = (h2 * attn_vec[1]).sum(dim=-1, keepdim=True)
        s3 = (h3 * attn_vec[2]).sum(dim=-1, keepdim=True)
        gamma = F.softmax(torch.cat((s1, s2, s3), dim=-1), dim=-1)
        return gamma[:, 0:1] * h1 + gamma[:, 1:2] * h2 + gamma[:, 2:3] * h3

    def encode(self, x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        h1_1 = F.relu(self.gc_1hop_1(x, f1))
        h2_1 = F.relu(self.gc_2hop_1(x, f2))
        h3_1 = F.relu(self.gc_3hop_1(x, f3))
        fused1 = self._path_fuse(h1_1, h2_1, h3_1, self.path_attn_vec1)
        fused1 = F.dropout(fused1, self.dropout, training=self.training)

        h1_2 = self.gc_1hop_2(fused1, f1)
        h2_2 = self.gc_2hop_2(fused1, f2)
        h3_2 = self.gc_3hop_2(fused1, f3)
        return self._path_fuse(h1_2, h2_2, h3_2, self.path_attn_vec2)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        e_u, e_v = emb[u], emb[v]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(self, x: torch.Tensor, f1: torch.Tensor, pairs: torch.Tensor, adj_skip: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if adj_skip is None or not isinstance(adj_skip, (tuple, list)):
            raise ValueError("ThreeHopSkipGNN requires adj_skip=(f_2hop, f_3hop)")
        f2, f3 = adj_skip[0], adj_skip[1]
        emb = self.encode(x, f1, f2, f3)
        return self.decode(emb, pairs), emb
```

---

## 3. Architecture 3: `ContrastiveSkipGNN` (Dual-View InfoNCE Alignment)

### 3.1 Mathematical Formulation
Dual views of the interactome are encoded simultaneously:
- **Direct View:** $H = \text{GNN}_o(X, F)$.
- **Skip View:** $S = \text{GNN}_s(X, F_s)$.

Each node representation is projected to a normalized contrastive space:
$$z_i = \frac{g_o(H_i)}{\|g_o(H_i)\|_2}, \quad \tilde{z}_i = \frac{g_s(S_i)}{\|g_s(S_i)\|_2}$$

- **Symmetric Cross-View InfoNCE Loss:**
  $$\mathcal{L}_{CL} = \frac{1}{2} \left[ -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \log \frac{\exp(z_i^T \tilde{z}_i / \tau)}{\sum_{j \in \mathcal{B}} \exp(z_i^T \tilde{z}_j / \tau)} + -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \log \frac{\exp(\tilde{z}_i^T z_i / \tau)}{\sum_{j \in \mathcal{B}} \exp(\tilde{z}_i^T z_j / \tau)} \right]$$

- **Total Objective:** $\mathcal{L}_{total} = \mathcal{L}_{BCEWithLogits} + \lambda \cdot \mathcal{L}_{CL}$ ($\tau = 0.2, \lambda = 0.1$).

### 3.2 Python Implementation (`src/models/contrastive_skipgnn.py`)

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import SparseGraphConvolution

class ContrastiveSkipGNN(nn.Module):
    skip_kind = "weighted"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5, proj_dim: int = 32, tau: float = 0.2, lambda_cl: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.tau = tau
        self.lambda_cl = lambda_cl

        self.o_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.o_gc2 = SparseGraphConvolution(nhid1, nhid2)

        self.s_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc2 = SparseGraphConvolution(nhid1, nhid2)

        self.fuse_gate = nn.Linear(nhid2 * 2, 1)

        self.proj_o = nn.Sequential(nn.Linear(nhid2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.proj_s = nn.Sequential(nn.Linear(nhid2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, f_orig: torch.Tensor, f_skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_o1 = F.relu(self.o_gc1(x, f_orig))
        h_o1 = F.dropout(h_o1, self.dropout, training=self.training)
        h_o2 = self.o_gc2(h_o1, f_orig)

        h_s1 = F.relu(self.s_gc1(x, f_skip))
        h_s1 = F.dropout(h_s1, self.dropout, training=self.training)
        h_s2 = self.s_gc2(h_s1, f_skip)

        g = torch.sigmoid(self.fuse_gate(torch.cat((h_o2, h_s2), dim=-1)))
        emb = g * h_o2 + (1.0 - g) * h_s2
        return emb, h_o2, h_s2

    def compute_contrastive_loss(self, h_o2: torch.Tensor, h_s2: torch.Tensor, active_nodes: torch.Tensor) -> torch.Tensor:
        unique_nodes = torch.unique(active_nodes)
        if unique_nodes.numel() <= 1:
            return torch.tensor(0.0, device=h_o2.device)

        z_o = F.normalize(self.proj_o(h_o2[unique_nodes]), dim=-1)
        z_s = F.normalize(self.proj_s(h_s2[unique_nodes]), dim=-1)

        sim_matrix = torch.mm(z_o, z_s.t()) / self.tau
        labels = torch.arange(unique_nodes.size(0), device=h_o2.device)
        loss_o2s = F.cross_entropy(sim_matrix, labels)
        loss_s2o = F.cross_entropy(sim_matrix.t(), labels)
        return 0.5 * (loss_o2s + loss_s2o)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u, v = pairs[:, 0], pairs[:, 1]
        e_u, e_v = emb[u], emb[v]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor, adj_skip: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if adj_skip is None:
            raise ValueError("ContrastiveSkipGNN requires adj_skip")
        emb, h_o2, h_s2 = self.encode(x, adj, adj_skip)
        logits = self.decode(emb, pairs)
        if self.training:
            cl_loss = self.compute_contrastive_loss(h_o2, h_s2, pairs.view(-1))
            return logits, emb, cl_loss
        return logits, emb
```

---

## 4. Pipeline Updates (`src/training/trainer.py` & `src/data/loader.py`)

### 4.1 Update `src/data/types.py`
Add `f_3hop` attribute to `DatasetBundle`:
```python
@dataclass
class DatasetBundle:
    # ... existing fields ...
    f_3hop: torch.Tensor | None = None
```

### 4.2 Update `src/data/loader.py`
Compute and store `f_3hop` during dataset loading:
```python
from src.models.three_hop_skipgnn import build_three_hop_matrices

# Inside load_dataset_splits():
w2_coo, w3_coo = build_three_hop_matrices(adj_train)
f_3hop = laplacian_normalize(w3_coo, add_self_loops=False)
f_3hop_tensor = scipy_to_torch_sparse(f_3hop, device)

# In return DatasetBundle:
f_3hop=f_3hop_tensor
```

### 4.3 Update `predict_probs` and `train_one_model` in `src/training/trainer.py`

```python
def _skip_matrix(bundle: DatasetBundle, model: torch.nn.Module) -> Any:
    kind = getattr(model, "skip_kind", "none")
    if kind == "binary":
        return bundle.f_skip_bin
    if kind == "weighted":
        return bundle.f_skip_weighted
    if kind == "three_hop_bundle":
        return (bundle.f_skip_weighted, bundle.f_3hop)
    return None

@torch.no_grad()
def predict_probs(model: torch.nn.Module, bundle: DatasetBundle, pairs: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    device = bundle.features.device
    skip = _skip_matrix(bundle, model)
    outs: list[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        batch = torch.as_tensor(pairs[start : start + batch_size], device=device)
        out = model(bundle.features, bundle.f_orig, batch, skip)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)

# Inside training loop of train_one_model():
for pairs, labels in loader:
    pairs = pairs.to(device)
    labels = labels.to(device)
    opt.zero_grad(set_to_none=True)
    out = model(bundle.features, bundle.f_orig, pairs, skip)
    if isinstance(out, tuple) and len(out) == 3:
        logits, _, cl_loss = out
        loss = loss_fn(logits, labels) + getattr(model, "lambda_cl", 0.1) * cl_loss
    else:
        logits = out[0]
        loss = loss_fn(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
```

### 4.4 Update `src/models/factory.py`
Register new model keys:
```python
from src.models.skip_gat import SkipGATv2
from src.models.three_hop_skipgnn import ThreeHopSkipGNN
from src.models.contrastive_skipgnn import ContrastiveSkipGNN

def build_model(name: str, bundle: DatasetBundle, hidden1: int = 64, hidden2: int = 64, decoder_hidden: int = 64, dropout: float = 0.5) -> nn.Module:
    kwargs = dict(nfeat=bundle.n_features, nhid1=hidden1, nhid2=hidden2, nhid_dec=decoder_hidden, dropout=dropout)
    key = name.lower()
    if key in {"gcn", "standard_gcn", "baseline_gcn"}:
        return StandardGCN(**kwargs)
    if key in {"skipgnn", "skipgnn_baseline", "baseline"}:
        return SkipGNNBaseline(**kwargs)
    if key in {"ams", "ams_skipgnn", "ams-skipgnn"}:
        return AMSSkipGNN(**kwargs)
    if key in {"gat", "skip_gat", "skipgatv2"}:
        return SkipGATv2(**kwargs)
    if key in {"3hop", "three_hop", "three_hop_skipgnn"}:
        return ThreeHopSkipGNN(**kwargs)
    if key in {"contrastive", "contrastive_skipgnn", "cl_skip"}:
        return ContrastiveSkipGNN(**kwargs)
    return make_ablation(key, **kwargs)
```

---

## 5. Execution Command for the Complete 7-Model Benchmark Suite

```bash
python scripts/run_benchmark.py --dataset DTI --models gcn skipgnn ams gat 3hop contrastive heuristic
```