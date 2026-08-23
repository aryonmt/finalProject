# Document 04: Training, Leak-Free Metric Protocol & Multi-Seed Matrix

---

## 1. Loss Formulation & Numerical Stability

### 1.1 Numerically Stable Objective
All network heads output unconstrained real logits $\hat{l} \in \mathbb{R}$. Training utilizes `torch.nn.BCEWithLogitsLoss()` directly, resolving legacy arithmetic overflows:

$$\mathcal{L}_{batch} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \cdot \log \sigma(\hat{l}_i) + (1 - y_i) \cdot \log(1 - \sigma(\hat{l}_i)) \right]$$

### 1.2 Full-Epoch Averaged Loss Calculation
The trainer maintains exact sample-weighted loss accumulation across minibatches:

```python
total_loss = 0.0
total_samples = 0

for labels, pairs in data_loader:
    optimizer.zero_grad()
    logits, _ = model(features, f_orig, f_skip, pairs)
    loss = loss_fn(logits, labels.float())
    loss.backward()
    
    # Optional Gradient Clipping for Numerical Guardrail
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    batch_size = labels.size(0)
    total_loss += loss.item() * batch_size
    total_samples += batch_size

epoch_avg_loss = total_loss / max(1, total_samples)
```

---

## 2. Leak-Free Metric Evaluation Protocol

### 2.1 API Separation of Threshold Selection and Evaluation
To guarantee that threshold tuning on test splits is architecturally impossible, the metric module exposes two distinct functions:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def find_optimal_f1_threshold(val_probs: np.ndarray, val_labels: np.ndarray, num_steps: int = 91) -> float:
    """
    Finds the probability threshold in [0.05, 0.95] that maximizes F1 on VALIDATION data.
    Strictly forbidden from receiving test data.
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    best_f1 = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (val_probs >= t).astype(np.int32)
        score = float(f1_score(val_labels, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    return best_threshold


def evaluate_predictions_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> dict[str, float]:
    """
    Evaluates predictions at a strictly pre-determined probability threshold.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    auroc = float(roc_auc_score(labels, probs))
    auprc = float(average_precision_score(labels, probs))

    preds = (probs >= threshold).astype(np.int32)
    f1 = float(f1_score(labels, preds, zero_division=0))
    brier = float(np.mean((probs - labels) ** 2))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "brier_score": brier,
        "threshold": float(threshold)
    }
```

---

## 3. Multi-Seed Benchmarking Protocol

### 3.1 Deterministic Seed Matrix
All experiments must be executed over five fixed seeds:
$$\text{Seeds} = [42, 13, 29, 71, 101]$$

### 3.2 Evaluation Lifecycle per Seed Run
1. **Train Loop:** Train on `train.csv` (1:1 positive + negative rows).
2. **Epoch Validation:** At each epoch end, evaluate on `val.csv` (Uniform) to compute $\text{AUPRC}_{val}$.
3. **Model Selection:** Save the model checkpoint `best_model.pt` at $\arg\max_{epoch} \text{AUPRC}_{val}$.
4. **Validation Threshold Tuning:** Load `best_model.pt`, predict probabilities on `val.csv`, and compute $\tau^* = \text{find\_optimal\_f1\_threshold}(p_{val}, y_{val})$.
5. **Test Scoring (Dual-Bank):**
   - **Bank A (Uniform Test):** Evaluate `best_model.pt` on `test.csv` (Uniform) at default $\tau = 0.5$ and optimal $\tau^*$.
   - **Bank B (Hard Test):** Evaluate `best_model.pt` on generated `test_hard` at default $\tau = 0.5$ and optimal $\tau^*$.
6. **Aggregate Statistics:** Record all metrics and report $\text{Mean} \pm \text{Std}$ across seeds.

---

## 4. Hyperparameter Specifications (YAML Locked)

```yaml
# configs/default_hyperparams.yaml
optimization:
  learning_rate: 0.0005
  weight_decay: 0.0005
  batch_size: 128
  max_epochs: 30
  early_stopping_patience: 8

architecture:
  hidden_dim_1: 64
  hidden_dim_2: 32
  decoder_hidden_dim: 16
  dropout: 0.5

evaluation:
  dual_bank: true
  num_threshold_steps: 91
  seeds: [42, 13, 29, 71, 101]
```