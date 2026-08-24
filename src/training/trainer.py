from __future__ import annotations

import copy
import random
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.samplers import generate_bipartite_aware_hard_negatives
from src.data.types import DatasetBundle, PairDataset
from src.eval.metrics import evaluate_dual_bank


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _skip_matrix(bundle: DatasetBundle, model: torch.nn.Module) -> torch.Tensor | None:
    kind = getattr(model, "skip_kind", "none")
    if kind == "binary":
        return bundle.f_skip_bin
    if kind == "weighted":
        return bundle.f_skip_weighted
    return None


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    pairs: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    device = bundle.features.device
    skip = _skip_matrix(bundle, model)
    outs: list[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        batch = torch.as_tensor(pairs[start : start + batch_size], device=device)
        logits, _ = model(bundle.features, bundle.f_orig, batch, skip)
        outs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


def train_one_model(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 8,
    grad_clip: float = 5.0,
    seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = bundle.features.device
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        PairDataset(bundle.train.pairs, bundle.train.labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    skip = _skip_matrix(bundle, model)
    best_state = copy.deepcopy(model.state_dict())
    best_auprc = -1.0
    stall = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for pairs, labels in loader:
            pairs = pairs.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(bundle.features, bundle.f_orig, pairs, skip)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            bs = int(labels.size(0))
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(1, n_seen)
        val_probs = predict_probs(model, bundle, bundle.val.pairs)
        from sklearn.metrics import average_precision_score

        val_auprc = float(average_precision_score(bundle.val.labels, val_probs))
        history.append({"epoch": epoch, "loss": epoch_loss, "val_auprc": val_auprc})
        msg = f"epoch={epoch:02d} loss={epoch_loss:.4f} val_auprc={val_auprc:.4f}"
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    model.load_state_dict(best_state)
    pos_test = bundle.test.pairs[bundle.test.labels >= 0.5]
    hard_neg = generate_bipartite_aware_hard_negatives(
        pos_test,
        bundle.adj_train,
        bundle.known_positives,
        bundle.bipartite,
        bundle.n_source,
        bundle.n_target,
        seed=seed,
    )
    hard_pairs = np.concatenate([pos_test, hard_neg], axis=0)
    hard_labels = np.concatenate(
        [np.ones(len(pos_test), dtype=np.int32), np.zeros(len(hard_neg), dtype=np.int32)]
    )
    val_p = predict_probs(model, bundle, bundle.val.pairs)
    test_p = predict_probs(model, bundle, bundle.test.pairs)
    hard_p = predict_probs(model, bundle, hard_pairs)
    metrics = evaluate_dual_bank(
        val_p, bundle.val.labels, test_p, bundle.test.labels, hard_p, hard_labels
    )
    metrics["best_val_auprc"] = best_auprc
    metrics["history"] = history
    metrics["hard_pairs"] = hard_pairs
    metrics["hard_labels"] = hard_labels
    metrics["hard_probs"] = hard_p
    metrics["test_probs"] = test_p
    return metrics

