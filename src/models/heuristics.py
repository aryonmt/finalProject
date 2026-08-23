from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def compute_heuristic_scores(
    adj: sp.spmatrix,
    pairs: np.ndarray,
    method: str = "resource_allocation",
) -> np.ndarray:
    csr = adj.tocsr()
    degrees = np.asarray(csr.sum(axis=1)).flatten()
    scores = np.zeros(len(pairs), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        nu = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
        nv = csr.indices[csr.indptr[v] : csr.indptr[v + 1]]
        common = np.intersect1d(nu, nv, assume_unique=True)
        if common.size == 0:
            continue
        if method == "common_neighbors":
            scores[i] = float(common.size)
        elif method == "jaccard":
            union = np.union1d(nu, nv)
            scores[i] = float(common.size) / max(1, len(union))
        elif method == "adamic_adar":
            scores[i] = float(np.sum(1.0 / np.log(np.maximum(2.0, degrees[common]))))
        elif method == "resource_allocation":
            scores[i] = float(np.sum(1.0 / np.maximum(1.0, degrees[common])))
        else:
            raise ValueError(f"Unknown heuristic method: {method}")
    return scores


compute_heuristic_scores = compute_heuristic_scores

