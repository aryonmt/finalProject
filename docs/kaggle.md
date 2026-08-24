# Kaggle T4 runner

Full training is designed for a **Kaggle Notebook with a T4 GPU**.

## Upload

File: [`notebooks/kaggle_runner.ipynb`](../notebooks/kaggle_runner.ipynb)

Settings:

- Accelerator: **GPU T4**
- Internet: **on** (the first cell clones this GitHub repo)
- **Save Version → Save & Run All** (not “Quick Save”)

The notebook clones `https://github.com/aryonmt/finalProject.git` branch `main` (override with env `REPO_BRANCH`), `pip install -e .`, then runs `scripts/run_benchmark.py` as subprocesses.

Stage 4 extra models on DDI/PPI are already in `results/`. A fresh Save & Run All **skips** any dataset/model that already has seeds 42/123/7. GDI `gat` / `3hop` / `contrastive` stay off unless you set env **`FILL_GDI_EXTRAS=1`** (GAT is ~15 min/epoch on T4).

## Stages (historical)

Earlier versions of the notebook used a `STAGE` env var:

| STAGE | What it trained |
| --- | --- |
| 0 | DTI `--quick` smoke |
| 1 | DTI + DDI: GCN, SkipGNN, AMS, heuristic |
| 2 | PPI + GDI baselines + DTI ablation/robustness |
| 3 | DTI extra models (`gat`, `3hop`, `contrastive`) + t-SNE |
| 4 | Extra models on DDI + PPI + GDI (GDI GAT did not finish) |

Executed archives of stages 1–3 live under `notebooks/executed/`. They are a log of what ran, not the upload target.

The current runner defaults to `STAGE=4` for DDI/PPI extras (already complete → skip). Set `FILL_GDI_EXTRAS=1` only if you want GDI `gat` / `3hop` / `contrastive`.

## Batch sizes used on T4

- DDI / PPI extra models: `--batch-size 1024`
- GDI extra models: `--batch-size 256`
- GAT skip-graph attention: `GAT_EDGE_CHUNK=131072` (see `src/models/skip_gat.py`)

Models run **one subprocess at a time** so a crash does not lose earlier CSVs.

## Resume

If a version dies mid-GDI:

1. Start a new notebook from the latest `kaggle_runner.ipynb` on GitHub.
2. **Add Input → Notebook Output Files** → the failed version.
3. Save & Run All. The ingest cell copies `results/` from that output and skips finished seeds.

## Bring artifacts home

The last cell writes `/kaggle/working/ams_skipgnn_kaggle_bundle.zip`. Download **only that zip**. On the repo machine:

```bash
python scripts/import_kaggle_bundle.py --src ams_skipgnn_kaggle_bundle.zip --make-figures
```

Do not commit zips. `kaggle kernels pull` downloads notebook **source**, not Output files. Use the Output tab or `kaggle kernels output`.
