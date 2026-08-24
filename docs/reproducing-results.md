# Reproducing results

There are three honest reproduction levels.

## 1. Figures from committed tables (minutes, CPU)

This is what a reader should do first.

```bash
pip install -e ".[dev]"
python scripts/make_figures.py --skip-tsne
```

Inputs: `results/*/benchmark.csv`, `results/ablation.csv`, `results/robustness.csv`, `results/*/pr_ams_seed42.npz`, `results/*/summary.json`.

Outputs: `figures/fig1_*.png` … `fig10_*.png` and `results/model_comparison.csv`.

t-SNE PNGs already in `figures/` were generated on Kaggle from embeddings that are **not** in git. Rebuilding them requires `--save-embeddings` during train, then `python scripts/plot_tsne.py`.

## 2. CPU smoke (does not match paper numbers; writes under `results/temp/`)

```bash
python scripts/fetch_data.py
pytest -q
python scripts/run_benchmark.py --dataset DDI --quick --device cpu
```

Use this to verify the install. `--quick` defaults to `results/temp/` and will not overwrite `results/DDI/benchmark.csv`.

## 3. Full training (Kaggle T4)

Follow [kaggle.md](kaggle.md). Summary:

1. Upload `notebooks/kaggle_runner.ipynb`.
2. GPU T4, Internet on, **Save & Run All**.
3. Default clone branch is `main`. Override with env `REPO_BRANCH` if needed.
4. Download `/kaggle/working/ams_skipgnn_kaggle_bundle.zip`.
5. On the laptop: `python scripts/import_kaggle_bundle.py --src ams_skipgnn_kaggle_bundle.zip --make-figures`
6. Do not commit the zip.

Expected wall-clock (order of magnitude, T4):

- DTI / DDI GCN–AMS: hours, not days
- Extra models on DDI/PPI at batch 1024: several hours
- GDI GAT: ~15 minutes **per epoch** at batch 256; that is why GDI extra models are missing

To fill the GDI gap only:

```bash
python scripts/run_benchmark.py --dataset GDI --models gat 3hop contrastive \
    --seeds 42 123 7 --batch-size 256 --save-embeddings
```

Run that on a GPU with enough time (or resume from a previous Kaggle output via **Add Input → Notebook Output Files** as documented in the runner notebook).

## What must stay identical

- Fold-1 CSVs from upstream SkipGNN
- Seeds `42 / 123 / 7`
- Train-only adjacency
- Hard negatives from `generate_bipartite_aware_hard_negatives`
- Val-only `τ*`
- `--dataset` spelling (`DTI` not `dti` for the folder; the loader uppercases)

## Importing someone else’s Kaggle zip

```bash
python scripts/import_kaggle_bundle.py --src path/to/bundle.zip \
    --archive-notebook notebooks/executed/kaggle_run.ipynb --make-figures
```

The importer merges CSV rows and copies PR curves / figures. Embeddings stay out unless you pass `--copy-embeddings`.
