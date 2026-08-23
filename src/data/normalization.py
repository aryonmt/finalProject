from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def scipy_to_torch_sparse(mat: sp.spmatrix, device: torch.device | None = None) -> torch.Tensor:
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape)).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def identity_sparse(n: int, device: torch.device | None = None) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.int64)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def laplacian_normalize(adj: sp.spmatrix, add_self_loops: bool) -> sp.coo_matrix:
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(adj.sum(axis=1)).flatten()
    if add_self_loops:
        inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    else:
        inv_sqrt = np.power(np.maximum(deg, 1e-5), -0.5)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    d = sp.diags(inv_sqrt)
    return (d @ adj @ d).tocoo()


def build_binary_skip(adj: sp.spmatrix) -> sp.coo_matrix:
    a = adj.tocsr().astype(np.float32)
    skip = a @ a.T
    skip = skip.sign().tolil()
    skip.setdiag(0)
    return skip.tocoo()


def build_weighted_skip(adj: sp.spmatrix) -> sp.coo_matrix:
    """Resource-allocation skip: W = A D^{-1} A^T with zero diagonal."""
    a = adj.tocsr().astype(np.float32)
    deg = np.asarray(a.sum(axis=1)).flatten()
    inv_deg = 1.0 / np.maximum(deg, 1.0)
    d_inv = sp.diags(inv_deg.astype(np.float32))
    w = a @ d_inv @ a.T
    w = w.tolil()
    w.setdiag(0)
    return w.tocoo()
