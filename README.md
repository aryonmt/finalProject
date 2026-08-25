# AMS-SkipGNN

Attentive Multi-Scale Skip Graph Neural Network for molecular interaction prediction on four public graphs: **DTI**, **DDI**, **PPI**, and **GDI**.

The code first reimplements Huang et al. SkipGNN without train/test leakage, then adds a resource-allocation skip graph, gated original/skip fusion, and a four-way decoder (AMS). Three extra encoders sit in the same training loop: SkipGATv2, 3-hop SkipGNN, and a contrastive dual-view SkipGNN.

Paper numbers in this repo come from Kaggle T4 runs (seeds `42`, `123`, `7`). Local machines are for setup, CPU smoke tests, and figure regeneration.

Remote: [https://github.com/aryonmt/finalProject.git](https://github.com/aryonmt/finalProject.git)

## What is in the box

| You want | Go here |
| --- | --- |
| Setup, train, reproduce figures | this README |
| Folder map | [docs/repository-layout.md](docs/repository-layout.md) |
| Leak-free data and skip graphs | [docs/data-and-sampling.md](docs/data-and-sampling.md) |
| Model math and CLI names | [docs/models.md](docs/models.md) |
| Trainer, metrics, dual banks | [docs/training-and-evaluation.md](docs/training-and-evaluation.md) |
| Cached tables and coverage gaps | [docs/results.md](docs/results.md) |
| Figure catalogue | [docs/figures.md](docs/figures.md) |
| Kaggle T4 runner | [docs/kaggle.md](docs/kaggle.md) |
| End-to-end reproduction | [docs/reproducing-results.md](docs/reproducing-results.md) |

## Repository layout

```
configs/          per-dataset YAML (merged on top of configs/default.yaml)
data/raw/         fold-1 CSVs from the upstream SkipGNN repo (fetched, then tracked)
docs/             how the code, data, models, and experiments fit together
figures/          300 DPI PNGs regenerated from results/
notebooks/        Kaggle runner + archived executed stage notebooks
results/          benchmark.csv, summary.json, PR curves, ablation, robustness
scripts/          fetch, train, plot, import a Kaggle zip
src/              loaders, models, trainer, metrics, plotting
tests/            CPU unit tests (no GPU required)
```

On-disk dataset names are `DTI`, `DDI`, `PPI`, `GDI` (singular `--dataset`). Do not invent other folder names.

## Setup

Python 3.10–3.12. A GPU is **not** required for install or `pytest`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"
python scripts/fetch_data.py
pytest -q
```

`fetch_data.py` sparse-clones [kexinhuang12345/SkipGNN](https://github.com/kexinhuang12345/SkipGNN.git) and copies `data/{DDI,PPI,DTI,GDI}/fold1/` into `data/raw/{DDI,PPI,DTI,GDI}/`. The fold-1 paths are the official splits. Do not “fix” them.

Conda alternative: `conda env create -f environment.yml`.

## Reproduce the paper figures (no training)

Cached metrics already live under `results/`. Figures are a function of those tables:

```bash
python scripts/make_figures.py --skip-tsne
```

That command also rewrites `results/model_comparison.csv` (mean over seeds) and tidies each `benchmark.csv`. t-SNE PNGs need `embeddings_*.npz`, which are gitignored; skip them unless you exported embeddings yourself.

## Train (full protocol)

Hyperparameters come from `configs/default.yaml` plus `configs/<dataset>.yaml`. DTI/DDI/PPI: 30 epochs, patience 8. GDI: 20 epochs, patience 6, batch 256. Adam `1e-3`, dropout 0.5, hidden 64.

**Full GPU training belongs on Kaggle T4**, not a laptop. See [docs/kaggle.md](docs/kaggle.md).

Local one-dataset example (CUDA if present):

```bash
python scripts/run_benchmark.py --dataset DTI --models gcn skipgnn ams heuristic gat 3hop contrastive --seeds 42 123 7
python scripts/run_ablation.py --dataset DTI
python scripts/run_robustness.py --dataset DTI
python scripts/make_figures.py --skip-tsne
```

Rows merge into `results/<DATASET>/benchmark.csv` by `(dataset, model, seed)`. Re-running a model overwrites that seed only.

`--quick` is a **2-epoch smoke test** that writes under `results/temp/` so it cannot overwrite paper CSVs. Never report `--quick` numbers.

CPU smoke:

```bash
python scripts/run_benchmark.py --dataset DDI --quick --device cpu
```

## Experimental coverage (this checkout)

| Dataset | GCN / SkipGNN / AMS / heuristic | SkipGATv2 / 3-hop / contrastive |
| --- | --- | --- |
| DTI | 3 seeds, uniform + hard | 3 seeds, uniform + hard |
| DDI | 3 seeds, uniform + hard | 3 seeds, uniform + hard |
| PPI | 3 seeds, uniform + hard | 3 seeds, uniform + hard |
| GDI | 3 seeds, uniform + hard | **re-run pending** (do not quote the encode-once extras; Stage 5 on current `main` uses per-batch encode) |

Ablation and missing-edge robustness are reported on **DTI**.

## Evaluation rules (do not change)

- Train positives only go into adjacency `A`. Val/test edges never leak into the graph.
- Uniform test = the official test CSV.
- Hard test = official test positives + degree-biased negatives that respect the bipartite cut (DTI/GDI).
- F1 threshold `τ*` is fit on **validation only**, then frozen for both test banks.
- Primary ranking metrics are AUPRC and AUROC at the default 0.5 operating point (`uniform_*` / `hard_*` columns).

## Model keys

| CLI key | Class | Skip graph |
| --- | --- | --- |
| `gcn` | `StandardGCN` | none |
| `skipgnn` | `SkipGNNBaseline` | binary 2-hop |
| `ams` | `AMSSkipGNN` | weighted RA skip + gate + 4-way decoder |
| `heuristic` | resource allocation | n/a |
| `gat` | `SkipGATv2` | weighted (chunked sparse attention) |
| `3hop` | `ThreeHopSkipGNN` | weighted + 3-walk bundle |
| `contrastive` | `ContrastiveSkipGNN` | weighted + InfoNCE |

Factory: `src.models.build_model`.

## Citation / upstream

Splits and the original SkipGNN architecture: Huang et al., *SkipGNN: Predicting Molecular Interactions with Skip-Graph Networks* ([code](https://github.com/kexinhuang12345/SkipGNN)).
