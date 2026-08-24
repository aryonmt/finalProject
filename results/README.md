# Cached experiment tables

These files are the paper results. Regenerate plots with:

```bash
python scripts/make_figures.py --skip-tsne
```

from the repo root. Do not hand-edit numbers.

| File | Keep in git? |
| --- | --- |
| `*/benchmark.csv` | yes |
| `*/summary.json` | yes (learning curves) |
| `*/pr_ams_seed*.npz` | yes |
| `ablation.csv`, `robustness.csv`, `model_comparison.csv` | yes |
| `*/embeddings_*.npz` | no (gitignored) |
| `*/checkpoints/` | no |
| `temp/`, `*.zip` | no |

**GDI** has no `gat` / `3hop` / `contrastive` rows. That is a coverage gap, not a dropped file. See [docs/results.md](../docs/results.md).
