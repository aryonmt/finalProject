# AMS-SkipGNN

Attentive Multi-Scale Skip Graph Neural Network for molecular interaction prediction (DDI, PPI, DTI, GDI). Reimplements Huang et al. SkipGNN without leakage, then adds a weighted skip graph, gated fusion, and a 4-way decoder.

## Layout

- `src/data` leak-free loaders, skip graphs, samplers
- `src/models` GCN, SkipGNN baseline, AMS-SkipGNN, heuristics
- `src/training` BCE-with-logits trainer, early stopping on val AUPRC
- `src/eval` dual-bank metrics (threshold from val only) and 300 DPI figures
- `scripts` fetch, benchmark, ablation, robustness, plots
- `notebooks/ams_skipgnn_kaggle_runner.ipynb` Kaggle T4 runner (Save & Commit All)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python scripts/fetch_data.py
pytest -q
```

## Run

```bash
python scripts/run_benchmark.py --dataset DTI --models gcn skipgnn ams
python scripts/run_ablation.py --dataset DTI
python scripts/run_robustness.py --dataset DTI
python scripts/make_figures.py
```

Quick CPU smoke:

```bash
python scripts/run_benchmark.py --dataset DDI --quick --device cpu
```

Kaggle: open `notebooks/ams_skipgnn_kaggle_runner.ipynb`, GPU T4, **Save & Commit All**. After the run, commit the notebook with printed outputs back into this repo.

Remote: `https://github.com/aryonmt/finalProject.git`

Train positives only go into `A`. Val/test edges never enter the adjacency. F1 threshold is fit on validation, then frozen for both uniform and hard test banks.
