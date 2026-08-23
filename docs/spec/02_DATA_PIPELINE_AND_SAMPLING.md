# Document 02: Data Pipeline, Normalization & Negative Sampling Specification

---

## 1. Raw Data Origin & Entity Mapping

All raw interaction datasets are sourced directly from the original SkipGNN repository (`https://github.com/kexinhuang12345/SkipGNN.git` [1]).

### 1.1 Dataset Properties & Schema

| Dataset | Type | Entity 1 Type ($u$) | Entity 2 Type ($v$) | Entity Count ($N$) | Entity Offset Indexing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `DDI` | Homogeneous | Drug | Drug | 1,514 | Single space: $[0, 1513]$ |
| `PPI` | Homogeneous | Protein | Protein | 5,604 | Single space: $[0, 5603]$ |
| `DTI` | Bipartite | Drug | Protein | 7,343 | Drug: $[0, 5017]$, Protein: $[5018, 7342]$ |
| `GDI` | Bipartite | Gene | Disease | 19,783 | Gene: $[0, 9412]$, Disease: $[9413, 19782]$ |

### 1.2 Upstream Folder Structure
The raw data files must be placed in `data/raw/` with the following structure:
```text
data/raw/
├── DDI/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── ddi.emb
│   └── ddi_unique_smiles.csv
├── PPI/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── ppi.emb
│   └── protein_list.csv
├── DTI/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── dti.emb
│   └── entity_list.csv
└── GDI/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── gdi.emb
    └── entity_list.csv
```

---

## 2. Leak-Free Graph Construction & Legacy Fixes

### 2.1 Transductive Leakage-Free Constraint
- The graph adjacency matrix $A$ and skip matrix $A_s$ are constructed **strictly from `train.csv` rows with `label == 1`**.
- Pairs in `val.csv` and `test.csv` (both positive and negative) are strictly excluded from adjacency matrix generation.

### 2.2 Feature Standardization (Z-Score Fix for Node2Vec)
- When `input_type == 'node2vec'`, dense embedding features contain negative values. L1 row normalization (`x / sum(x)`) is strictly prohibited.
- **Specification:** Node2Vec features must be column-wise Z-score standardized:

```python
import numpy as np

def standardize_node2vec_features(features: np.ndarray) -> np.ndarray:
    """
    Applies column-wise Z-score standardization: (X - mean) / std.
    Safe against zero-variance columns and negative values.
    """
    features = np.asarray(features, dtype=np.float32)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return np.ascontiguousarray((features - mean) / std, dtype=np.float32)
```

### 2.3 Sparse COO Tensor Conversion
All adjacency matrices must be converted to PyTorch sparse COO tensors:

```python
import torch
import scipy.sparse as sp

def scipy_coo_to_torch_sparse(matrix: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    """
    Converts scipy.sparse.coo_matrix to torch.sparse_coo_tensor on target device.
    """
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((matrix.row, matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(matrix.data.astype(np.float32))
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
```

---

## 3. Negative Sampling Protocols

### 3.1 Protocol A: Uniform Random Negative Bank
For homogeneous and bipartite datasets, uniform random negative pairs are extracted directly from pre-split `val.csv` and `test.csv` where `label == 0`, maintaining a 1:1 balanced ratio against positive pairs.

### 3.2 Protocol B: Bipartite-Aware & Symmetric Hard Negative Sampler

#### Crucial Fix 1: Per-Entity-Type Quartile Partitioning
- In bipartite graphs (`DTI`, `GDI`), target nodes ($v$) belong to Entity Type 2 (Proteins or Diseases).
- Degree quartile bucketing must be performed **strictly over nodes of Entity Type 2**, ensuring negative candidates $v^-$ are always valid target-type entities.

#### Crucial Fix 2: Symmetric Edge Non-Existence Check
- For homogeneous graphs (`DDI`, `PPI`), non-existence of an edge must check **both directions**: `(u, v) not in positives AND (v, u) not in positives`.
- For bipartite graphs (`DTI`, `GDI`), the directed check `(u, v) not in positives` is applied.

```python
import numpy as np
import scipy.sparse as sp

def generate_bipartite_aware_hard_negatives(
    positive_pairs: np.ndarray,
    train_adj: sp.coo_matrix,
    known_positives: set[tuple[int, int]],
    is_bipartite: bool,
    num_source_nodes: int,
    num_target_nodes: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generates quartile degree-matched negative pairs with strict entity-type typing.
    
    Args:
        positive_pairs: Array of shape (M, 2) containing positive (u, v) pairs.
        train_adj: Training adjacency matrix (N x N).
        known_positives: Global set of all known positive edges across train/val/test.
        is_bipartite: True for DTI and GDI, False for DDI and PPI.
        num_source_nodes: Number of source entities (e.g. Drugs in DTI = 5018).
        num_target_nodes: Number of target entities (e.g. Proteins in DTI = 2325).
        seed: Deterministic random seed.
    Returns:
        np.ndarray: Array of shape (M, 2) containing hard negative pairs.
    """
    rng = np.random.default_rng(seed)
    total_nodes = train_adj.shape[0]
    degrees = np.asarray(train_adj.sum(axis=1)).flatten()

    if is_bipartite:
        # Target nodes reside in offset slice [num_source_nodes, total_nodes)
        target_offset = num_source_nodes
        target_indices = np.arange(target_offset, total_nodes)
        target_degrees = degrees[target_indices]
    else:
        # Homogeneous: all nodes are valid targets
        target_offset = 0
        target_indices = np.arange(total_nodes)
        target_degrees = degrees

    # Compute quartile thresholds on target entity subset
    q25, q50, q75 = np.percentile(target_degrees, [25, 50, 75])

    target_bins = np.zeros(len(target_indices), dtype=np.int32)
    target_bins[target_degrees > q25] = 1
    target_bins[target_degrees > q50] = 2
    target_bins[target_degrees > q75] = 3

    bucket_to_target_nodes = {
        b: target_indices[target_bins == b] for b in range(4)
    }

    hard_negatives = []
    for u, v in positive_pairs:
        # Determine quartile bucket of ground-truth target v
        v_local_idx = v - target_offset
        v_bucket = target_bins[v_local_idx]
        candidate_pool = bucket_to_target_nodes[v_bucket]

        found = False
        for _ in range(100):
            neg_v = int(rng.choice(candidate_pool))
            if is_bipartite:
                valid = (u, neg_v) not in known_positives and u != neg_v
            else:
                valid = (u, neg_v) not in known_positives and (neg_v, u) not in known_positives and u != neg_v

            if valid:
                hard_negatives.append((u, neg_v))
                found = True
                break

        if not found:
            # Fallback within target entity space
            while True:
                neg_v = int(rng.choice(target_indices))
                if is_bipartite:
                    valid = (u, neg_v) not in known_positives and u != neg_v
                else:
                    valid = (u, neg_v) not in known_positives and (neg_v, u) not in known_positives and u != neg_v

                if valid:
                    hard_negatives.append((u, neg_v))
                    break

    return np.asarray(hard_negatives, dtype=np.int64)
```