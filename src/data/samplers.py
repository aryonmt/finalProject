"""Uniform and degree-biased negative pair sampling."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def generate_uniform_negatives(
    n_nodes: int,
    n_samples: int,
    known_positives: set[tuple[int, int]],
    rng: np.random.Generator,
    bipartite: bool = False,
    n_source: int | None = None,
) -> np.ndarray:
    """Sample negatives uniformly, never repeating a known positive."""
    pairs: list[tuple[int, int]] = []
    n_source = n_nodes if n_source is None else n_source
    while len(pairs) < n_samples:
        if bipartite:
            u = int(rng.integers(0, n_source))
            v = int(rng.integers(n_source, n_nodes))
            if (u, v) in known_positives or u == v:
                continue
        else:
            u = int(rng.integers(0, n_nodes))
            v = int(rng.integers(0, n_nodes))
            if u == v:
                continue
            if (u, v) in known_positives or (v, u) in known_positives:
                continue
        pairs.append((u, v))
    return np.asarray(pairs, dtype=np.int64)


def generate_bipartite_aware_hard_negatives(
    positive_pairs: np.ndarray,
    train_adj: sp.spmatrix,
    known_positives: set[tuple[int, int]],
    is_bipartite: bool,
    num_source_nodes: int,
    num_target_nodes: int,
    seed: int = 42,
) -> np.ndarray:
    """Degree-quartile negatives that stay on the correct side of a bipartite cut."""
    rng = np.random.default_rng(seed)
    total_nodes = train_adj.shape[0]
    degrees = np.asarray(train_adj.sum(axis=1)).flatten()

    if is_bipartite:
        target_offset = num_source_nodes
        target_indices = np.arange(target_offset, total_nodes)
        target_degrees = degrees[target_indices]
    else:
        target_offset = 0
        target_indices = np.arange(total_nodes)
        target_degrees = degrees

    q25, q50, q75 = np.percentile(target_degrees, [25, 50, 75])
    target_bins = np.zeros(len(target_indices), dtype=np.int32)
    target_bins[target_degrees > q25] = 1
    target_bins[target_degrees > q50] = 2
    target_bins[target_degrees > q75] = 3
    bucket_to_target_nodes = {b: target_indices[target_bins == b] for b in range(4)}

    hard_negatives: list[tuple[int, int]] = []
    for u, v in positive_pairs:
        u = int(u)
        v = int(v)
        v_local_idx = v - target_offset
        if v_local_idx < 0 or v_local_idx >= len(target_bins):
            nonempty = [b for b, nodes in bucket_to_target_nodes.items() if nodes.size > 0]
            v_bucket = int(rng.choice(nonempty)) if nonempty else 0
        else:
            v_bucket = int(target_bins[v_local_idx])
        candidate_pool = bucket_to_target_nodes[v_bucket]
        if candidate_pool.size == 0:
            candidate_pool = target_indices

        found = False
        for _ in range(100):
            neg_v = int(rng.choice(candidate_pool))
            if is_bipartite:
                valid = (u, neg_v) not in known_positives and u != neg_v
            else:
                valid = (
                    (u, neg_v) not in known_positives
                    and (neg_v, u) not in known_positives
                    and u != neg_v
                )
            if valid:
                hard_negatives.append((u, neg_v))
                found = True
                break
        if not found:
            for _ in range(1000):
                neg_v = int(rng.choice(target_indices))
                if is_bipartite:
                    valid = (u, neg_v) not in known_positives and u != neg_v
                else:
                    valid = (
                        (u, neg_v) not in known_positives
                        and (neg_v, u) not in known_positives
                        and u != neg_v
                    )
                if valid:
                    hard_negatives.append((u, neg_v))
                    break
            else:
                hard_negatives.append((u, int(rng.choice(target_indices))))

    _ = num_target_nodes
    return np.asarray(hard_negatives, dtype=np.int64)
