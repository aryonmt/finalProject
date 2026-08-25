# Repository layout

Every path below is relative to the repo root. Names on disk are the source of truth: `DTI`, `DDI`, `PPI`, `GDI` (uppercase), CLI `--dataset` (singular).

## Top level

| Path | Role |
| --- | --- |
| `src/` | Installable package (`pip install -e .`). Loaders, models, trainer, metrics, plots. |
| `scripts/` | User-facing CLIs. Thin wrappers; logic lives in `src/`. |
| `configs/` | YAML hyperparameters. `default.yaml` is merged with `<dataset>.yaml`. |
| `data/raw/<DATASET>/` | `train.csv`, `val.csv`, `test.csv`, plus an entity list. |
| `results/` | Cached metrics. Commit CSVs, JSON summaries, AMS PR `.npz`. Not embeddings or checkpoints. |
| `figures/` | 300 DPI PNGs. Regenerated from `results/` by `scripts/make_figures.py`. |
| `notebooks/` | Kaggle runner plus archived executed stage notebooks. |
| `docs/` | How the pieces work, including [training-protocol.md](training-protocol.md). |
| `tests/` | CPU unit tests. `pytest -q`. |
| `pyproject.toml` | Package metadata and runtime dependencies. |
| `environment.yml` | Conda env that pip-installs the package. |

## `src/`

```
src/data/         constants, DatasetBundle, leak-free loader, skip graphs, samplers
src/models/       gcn, skipgnn, ams, gat, 3hop, contrastive, heuristics, factory
src/training/     seed, train loop, dual-bank eval, embedding export
src/eval/         F1 threshold, AUROC/AUPRC, figure generation
```

Public constructors:

- `load_dataset_splits(name, …)` → `DatasetBundle`
- `build_model(name, bundle, …)` → `nn.Module`
- `train_one_model(model, bundle, …)` → metrics dict
- `generate_all_figures(results_dir, output_dir, skip_tsne=…)`

## `scripts/`

| Script | What it does |
| --- | --- |
| `fetch_data.py` | Clone upstream SkipGNN and copy fold-1 splits into `data/raw/`. |
| `run_benchmark.py` | Train one dataset × models × seeds; merge CSV/JSON. |
| `run_ablation.py` | Four AMS steps on one dataset → `results/ablation.csv`. |
| `run_robustness.py` | Drop train edges → `results/robustness.csv`. |
| `make_figures.py` | Rebuild `figures/fig*.png` and `results/model_comparison.csv`. |
| `plot_tsne.py` | t-SNE from `embeddings_*.npz` if those files exist locally. |
| `import_kaggle_bundle.py` | Fold a Kaggle zip into `results/` + `figures/`. |
| `pack_kaggle_bundle.py` | Optional zip helper (the notebook also writes the zip itself). |

## What is gitignored

Embeddings (`results/**/embeddings_*.npz`), checkpoints, zips, `results/temp/`, virtualenvs, and `__pycache__`. Tables and PNGs are tracked so a clone can regenerate figures without a GPU.
