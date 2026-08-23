# Document 01: System Architecture & Rapid 2-Day Execution Blueprint

- **Project Title:** AMS-SkipGNN: Attentive Multi-Scale & Weighted Skip Graph Neural Networks for Molecular Interaction Prediction
- **Target Tasks:** Drug-Target Interaction (DTI), Drug-Drug Interaction (DDI), Protein-Protein Interaction (PPI), Gene-Disease Interaction (GDI)
- **Target Hardware:** Local (Development / Lint / CPU Smoke Test) -> GitHub -> Kaggle GPU (Tesla T4)
- **Execution Timeline:** 2-Day High-Velocity Staged Execution

---

## 1. Project Charter & Executive Summary

### 1.1 Objective
This project re-engineers, optimizes, and proves the superiority of **AMS-SkipGNN** (*Attentive Multi-Scale & Weighted SkipGNN*) over Huang et al. (*Nature Scientific Reports* 2020) on canonical biomedical interaction benchmarks.

The codebase fixes all fundamental structural and evaluation bugs present in legacy implementations:
1. **Topological Density Fix:** Replaces binary `sign(A A^T)` skip graph with a degree-weighted continuous Resource Allocation skip operator.
2. **Dynamic Cross-Graph Attention:** Introduces Adaptive Gated Residual units between direct interaction and skip paths.
3. **4-Way Interactive Tensor Decoder:** Combines representations via `[u || v || u ⊙ v || |u - v|]` with batch normalization.
4. **Leak-Free Dual-Bank Evaluation:** Evaluates both standard Uniform AUPRC and entity-aware Degree-Matched Hard AUPRC without metric tuning leakage.

### 1.2 Two-Day Staged Execution Strategy (Risk Mitigation)

To eliminate GPU timeout and debugging overhead under a strict 2-day deadline, execution is partitioned into two distinct stages:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Fast MVP (MANDATORY - Day 1)                                      │
│ • Datasets: DTI (Bipartite) & DDI (Homogeneous)                            │
│ • Models: Heuristic Baselines, Standard GCN, SkipGNN Baseline, AMS-SkipGNN │
│ • Seeds: 3 Seeds (42, 13, 29)                                              │
│ • Output: Working pipeline, bug-free numbers beating SkipGNN, core plots   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ (If Day 1 passes with zero errors)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Extended Grid & Ablation (COMPLETION - Day 2)                     │
│ • Datasets: Add PPI & GDI (All 4 datasets)                                 │
│ • Models: Full 5-Seed Evaluation (42, 13, 29, 71, 101)                     │
│ • Ablation Study: 4-step progressive ladder on DTI                         │
│ • Missing-Edge Robustness: 10% to 90% edge removal stress-test             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory & File Organization

The repository layout is streamlined for rapid delivery and automated Kaggle execution:

```text
AMS-SkipGNN/
├── configs/
│   ├── dti_config.yaml
│   ├── ddi_config.yaml
│   ├── ppi_config.yaml
│   └── gdi_config.yaml
├── data/
│   ├── raw/                  # Cloned directly from SkipGNN upstream repository
│   │   ├── DTI/
│   │   ├── DDI/
│   │   ├── PPI/
│   │   └── GDI/
│   └── processed/            # Cached sparse matrices and standardized features
├── notebooks/
│   └── ams_skipgnn_kaggle_runner.ipynb  # Self-contained Kaggle execution notebook
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Leak-free graph & split loader
│   │   ├── normalization.py  # Z-Score standardization & sparse Laplacian norm
│   │   └── samplers.py       # Entity-type-aware Hard & Uniform negative samplers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers.py         # PyTorch Sparse COO Graph Convolution operator
│   │   ├── gcn.py            # Standard 2-Layer GCN baseline
│   │   ├── skipgnn_baseline.py # Re-implemented SkipGNN baseline
│   │   ├── ams_skipgnn.py    # AMS-SkipGNN champion model & ablation ladder
│   │   └── heuristics.py     # Non-parametric link prediction heuristics
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Complete epoch-averaged training loop
│   │   └── losses.py         # Numerically stable BCEWithLogitsLoss wrapper
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py        # Leak-free metric calculator with strict threshold separation
│       └── plotting.py       # 300-DPI publication figure generator
├── tests/
│   └── test_smoke.py         # Lightweight 30-second smoke test suite
├── scripts/
│   ├── run_benchmark.py      # Unified CLI runner (supports --quick MVP flag)
│   ├── run_ablation.py       # Ablation ladder runner
│   └── run_robustness.py     # Missing-edge resilience tester
├── pyproject.toml
├── environment.yml
└── README.md
```

---

## 3. Module Boundaries & Interfaces

| Module | Core Responsibility | Public Functions / Classes |
| :--- | :--- | :--- |
| `src.data.loader` | Loads raw data and builds sparse train-only graphs. | `load_dataset_splits(dataset_name, data_dir, input_type)` |
| `src.data.samplers` | Entity-aware negative candidate generation. | `generate_uniform_negatives()`, `generate_bipartite_aware_hard_negatives()` |
| `src.models.gcn` | 2-layer standard GCN baseline (no skip connections). | `StandardGCN(nfeat, nhid1, nhid2, nhid_dec, dropout)` |
| `src.models.skipgnn_baseline` | Reimplemented baseline with fixed Sparse SpMM. | `SkipGNNBaseline(...)` |
| `src.models.ams_skipgnn` | Full AMS-SkipGNN with Weighted Skip & Adaptive Gate. | `AMSSkipGNN(...)`, `AMSAblationLadder(...)` |
| `src.evaluation.metrics` | Strict validation-to-test thresholding & evaluation. | `find_optimal_f1_threshold(val_probs, val_labels)`, `evaluate_predictions(probs, labels, threshold)` |
| `src.evaluation.plotting` | Generates 4 publication figures at 300 DPI. | `plot_uniform_vs_hard_auprc()`, `plot_ablation_ladder()`, `plot_robustness_curve()`, `plot_pr_curves()` |

---

## 4. Software Environment & NumPy/PyTorch Compatibility

To avoid environment collisions on Kaggle GPUs:
- **PyTorch & CUDA:** Use Kaggle's pre-installed PyTorch (`torch >= 2.1.0` with CUDA acceleration).
- **NumPy 2.x Compatibility:** All code must be strictly NumPy 2.x safe (avoiding deprecated aliases like `np.float_`, using `np.asarray(..., dtype=np.float32)`, explicit casting, and keyword-safe `np.percentile`).
- **Packaging:** Install locally or on Kaggle with `pip install -e . --no-deps`.

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ams-skipgnn"
version = "1.0.0"
description = "Attentive Multi-Scale & Weighted Skip Graph Neural Networks"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.2.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0"
]
```

---

## 5. Lightweight Smoke Test Specification (`tests/test_smoke.py`)

The test suite must execute in **under 30 seconds** on CPU to verify pipeline integrity before pushing to GitHub:

1. **`test_data_loader_shapes()`:** Verifies that loaded sparse adjacency matrices match known entity counts for DDI (1,514) and DTI (7,343).
2. **`test_bipartite_negative_typing()`:** Verifies that negative pairs for DTI strictly pair drug entities ($0 \le u < 5018$) with protein entities ($5018 \le v < 7343$).
3. **`test_model_forward_pass()`:** Performs a dummy 2-step forward pass on CPU for `StandardGCN`, `SkipGNNBaseline`, and `AMSSkipGNN` to guarantee non-NaN logit tensors.
4. **`test_metric_threshold_isolation()`:** Asserts that `find_optimal_f1_threshold` and `evaluate_predictions` output valid metrics in $[0, 1]$ without NaN failures.