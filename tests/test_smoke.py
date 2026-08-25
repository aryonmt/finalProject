from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data.normalization import build_binary_skip, build_weighted_skip, laplacian_normalize
from src.data.samplers import generate_bipartite_aware_hard_negatives
from src.eval.metrics import evaluate_at_threshold, find_optimal_f1_threshold
from src.models.ams_skipgnn import AMSSkipGNN
from src.models.contrastive_skipgnn import ContrastiveSkipGNN
from src.models.factory import build_model
from src.models.gcn import StandardGCN
from src.models.skip_gat import SkipGATv2
from src.models.skipgnn import SkipGNNBaseline
from src.models.three_hop_skipgnn import ThreeHopSkipGNN


def _tiny_adj(n: int = 12) -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, n, size=(20, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj.tocsr()


def test_skip_graphs_zero_diag():
    adj = _tiny_adj()
    binary = build_binary_skip(adj).tocsr()
    weighted = build_weighted_skip(adj).tocsr()
    assert np.allclose(binary.diagonal(), 0)
    assert np.allclose(weighted.diagonal(), 0)
    f = laplacian_normalize(adj, add_self_loops=True)
    assert f.shape == adj.shape


def test_models_forward_no_nan():
    n, d = 16, 16
    x = torch.eye(n)
    idx = torch.stack([torch.arange(n), torch.arange(n)])
    adj = torch.sparse_coo_tensor(idx, torch.ones(n), (n, n)).coalesce()
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    for model in (
        StandardGCN(d, 8, 8, 8, 0.1),
        SkipGNNBaseline(d, 8, 8, 8, 0.1),
        AMSSkipGNN(d, 8, 8, 8, 0.1),
        SkipGATv2(d, 8, 8, 8, 0.1, heads=2),
        ThreeHopSkipGNN(d, 8, 8, 8, 0.1),
        ContrastiveSkipGNN(d, 8, 8, 8, 0.1),
    ):
        model.eval()
        skip = (adj, adj) if isinstance(model, ThreeHopSkipGNN) else adj
        out = model(x, adj, pairs, skip)
        logits, emb = out[0], out[1]
        assert torch.isfinite(logits).all()
        assert torch.isfinite(emb).all()
        assert logits.shape == (4,)


def test_hard_negatives_respect_bipartite_types():
    n_src, n_tgt = 10, 8
    n = n_src + n_tgt
    pos = np.array([[0, 10], [1, 11], [2, 12]], dtype=np.int64)
    adj = sp.csr_matrix((n, n), dtype=np.float32)
    known = {(int(u), int(v)) for u, v in pos}
    hard = generate_bipartite_aware_hard_negatives(pos, adj, known, True, n_src, n_tgt, seed=0)
    assert hard.shape == (3, 2)
    assert np.all(hard[:, 0] < n_src)
    assert np.all(hard[:, 1] >= n_src)
    for u, v in hard:
        assert (int(u), int(v)) not in known


def test_threshold_is_validation_only():
    rng = np.random.default_rng(1)
    val_y = np.array([0, 0, 0, 1, 1, 1])
    val_p = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    tau = find_optimal_f1_threshold(val_p, val_y)
    test_p = rng.random(20)
    test_y = rng.integers(0, 2, size=20)
    m = evaluate_at_threshold(test_p, test_y, tau)
    assert 0.0 <= m["auprc"] <= 1.0
    assert 0.05 <= tau <= 0.95
    # function signature forbids coupling: tau computed before seeing test labels
    assert tau == find_optimal_f1_threshold(val_p, val_y)


@pytest.mark.skipif(
    not (Path := __import__("pathlib").Path)("data/raw/DDI/train.csv").exists(),
    reason="raw DDI splits not fetched yet",
)
def test_ddi_shapes_and_no_leakage():
    from src.data.loader import load_dataset_splits

    bundle = load_dataset_splits("DDI", device=torch.device("cpu"))
    assert bundle.n_nodes == 1514
    assert bundle.adj_train.shape == (bundle.n_nodes, bundle.n_nodes)
    train_pos = {tuple(map(int, p)) for p in bundle.train.pairs[bundle.train.labels >= 0.5]}
    val_pos = {tuple(map(int, p)) for p in bundle.val.pairs[bundle.val.labels >= 0.5]}
    test_pos = {tuple(map(int, p)) for p in bundle.test.pairs[bundle.test.labels >= 0.5]}
    assert train_pos.isdisjoint(val_pos)
    assert train_pos.isdisjoint(test_pos)
    # adjacency constructed from train positives only
    coo = bundle.adj_train.tocoo()
    adj_edges = set(zip(map(int, coo.row), map(int, coo.col)))
    for u, v in val_pos | test_pos:
        if (u, v) not in train_pos and (v, u) not in train_pos:
            assert (u, v) not in adj_edges


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/DTI/train.csv").exists(),
    reason="raw DTI splits not fetched yet",
)
def test_dti_bipartite_ranges():
    from src.data.loader import load_dataset_splits
    from src.data.samplers import generate_bipartite_aware_hard_negatives

    bundle = load_dataset_splits("DTI", device=torch.device("cpu"))
    assert bundle.n_source in {5017, 5018}
    assert bundle.n_nodes == 7343
    pos = bundle.test.pairs[bundle.test.labels >= 0.5]
    hard = generate_bipartite_aware_hard_negatives(
        pos[:32],
        bundle.adj_train,
        bundle.known_positives,
        True,
        bundle.n_source,
        bundle.n_target,
        seed=0,
    )
    assert np.all(hard[:, 0] < bundle.n_source)
    assert np.all(hard[:, 1] >= bundle.n_source)


def test_laplacian_isolated_nodes_are_zero():
    adj = sp.csr_matrix(np.zeros((4, 4), dtype=np.float32))
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    norm = laplacian_normalize(adj, add_self_loops=False).tocsr()
    # degree-0 nodes must not be inflated by 1e-5 clipping
    assert float(norm[2, 2]) == 0.0
    assert float(norm[3, 3]) == 0.0
    assert np.isfinite(norm.toarray()).all()


def test_bipartite_heuristics_are_nonzero():
    from src.models.heuristics import compute_heuristic_scores

    n_src, n_tgt = 6, 5
    n = n_src + n_tgt
    # undirected bipartite edges: 0-6, 0-7, 1-6, 2-8, 3-9
    rows = np.array([0, 0, 1, 2, 3, 6, 7, 6, 8, 9], dtype=np.int64)
    cols = np.array([6, 7, 6, 8, 9, 0, 0, 1, 2, 3], dtype=np.int64)
    adj = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    # 1->6->0->7 is a 3-walk; 1-hop common neighbors are empty on bipartite graphs
    pairs = np.array([[1, 7], [0, 8]], dtype=np.int64)
    hop1 = compute_heuristic_scores(adj, pairs, bipartite=False)
    hop3 = compute_heuristic_scores(adj, pairs, bipartite=True)
    assert np.allclose(hop1, 0.0)
    assert hop3[0] > 0


def test_new_models_train_step_and_factory():
    n, d = 12, 12
    x = torch.eye(n)
    idx = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n])
    adj = torch.sparse_coo_tensor(idx, torch.ones(n), (n, n)).coalesce()
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    for name in ("gat", "3hop", "contrastive"):
        class _B:
            n_features = d

        model = build_model(name, _B(), hidden1=8, hidden2=8, decoder_hidden=8, dropout=0.0)
        model.train()
        skip = (adj, adj) if name == "3hop" else adj
        out = model(x, adj, pairs, skip)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out[0], y)
        if len(out) == 3:
            loss = loss + out[2]
        loss.backward()
        assert torch.isfinite(loss).item()


def test_three_hop_matrix_is_finite():
    from src.data.normalization import build_three_hop_return, laplacian_normalize

    adj = _tiny_adj(10)
    w3 = build_three_hop_return(adj)
    f3 = laplacian_normalize(w3, add_self_loops=False)
    assert f3.shape == adj.shape
    assert np.isfinite(f3.toarray()).all()


def test_extract_node_embeddings_shape():
    from src.data.types import DatasetBundle, SplitArrays
    from src.training.trainer import extract_node_embeddings, set_seed

    set_seed(0)
    n, d = 10, 10
    x = torch.eye(n)
    idx = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n])
    adj = torch.sparse_coo_tensor(idx, torch.ones(n), (n, n)).coalesce()
    zeros = np.zeros((2, 2), dtype=np.int64)
    labels = np.array([1, 0], dtype=np.int32)
    bundle = DatasetBundle(
        name="toy",
        n_nodes=n,
        n_source=5,
        n_target=5,
        bipartite=True,
        features=x,
        adj_train=sp.csr_matrix((n, n), dtype=np.float32),
        f_orig=adj,
        f_skip_bin=adj,
        f_skip_weighted=adj,
        f_3hop=adj,
        train=SplitArrays(zeros, labels),
        val=SplitArrays(zeros, labels),
        test=SplitArrays(zeros, labels),
        known_positives=set(),
        input_type="one_hot",
    )
    model = StandardGCN(d, 8, 8, 8, 0.0)
    model.eval()
    emb = extract_node_embeddings(model, bundle)
    assert emb.shape[0] == n
    assert np.isfinite(emb).all()


def test_chunked_gat_matches_large_chunk():
    from src.models.skip_gat import _ChunkedSparseGATv2, _softmax_gatv2

    torch.manual_seed(0)
    n_nodes, heads, d_k = 20, 2, 4
    n_edges = 80
    h_src = torch.randn(n_nodes, heads, d_k, requires_grad=True)
    h_dst = torch.randn(n_nodes, heads, d_k, requires_grad=True)
    attn = torch.randn(heads, d_k, requires_grad=True)
    u = torch.randint(0, n_nodes, (n_edges,))
    v = torch.randint(0, n_nodes, (n_edges,))
    out_a = _ChunkedSparseGATv2.apply(h_src, h_dst, attn, u, v, 0.2, 0.0, False, 7)
    out_b = _ChunkedSparseGATv2.apply(h_src, h_dst, attn, u, v, 0.2, 0.0, False, 10_000)
    out_c = _softmax_gatv2(h_src, h_dst, attn, u, v, 0.2, 0.0, False)
    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_a, out_c, atol=1e-5, rtol=1e-5)
    out_a.sum().backward()
    assert torch.isfinite(h_src.grad).all()
    assert torch.isfinite(h_dst.grad).all()
    assert torch.isfinite(attn.grad).all()


def test_encode_once_trains_new_models():
    from src.data.types import DatasetBundle, SplitArrays
    from src.training.trainer import train_one_model

    n, d = 16, 16
    x = torch.eye(n)
    idx = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n])
    adj = torch.sparse_coo_tensor(idx, torch.ones(n), (n, n)).coalesce()
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, n, size=(32, 2))
    labels = rng.integers(0, 2, size=32).astype(np.int32)
    bundle = DatasetBundle(
        name="toy",
        n_nodes=n,
        n_source=8,
        n_target=8,
        bipartite=False,
        features=x,
        adj_train=sp.csr_matrix((n, n), dtype=np.float32),
        f_orig=adj,
        f_skip_bin=adj,
        f_skip_weighted=adj,
        f_3hop=adj,
        train=SplitArrays(pairs, labels),
        val=SplitArrays(pairs[:8], np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)),
        test=SplitArrays(pairs[:8], np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)),
        known_positives=set(),
        input_type="one_hot",
    )
    for name in ("gat", "3hop", "contrastive"):
        class _B:
            n_features = d

        model = build_model(name, _B(), hidden1=8, hidden2=8, decoder_hidden=8, dropout=0.0)
        metrics = train_one_model(
            model,
            bundle,
            epochs=1,
            batch_size=8,
            patience=9,
            seed=0,
            encode_once=True,
        )
        assert np.isfinite(metrics["best_val_auprc"])
