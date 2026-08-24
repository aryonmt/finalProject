# Results tables

Canonical numbers are the CSVs in git, not screenshots.

## Files

| Path | Contents |
| --- | --- |
| `results/DTI/benchmark.csv` | All models × 3 seeds (including extra architectures) |
| `results/DDI/benchmark.csv` | Same |
| `results/PPI/benchmark.csv` | Same |
| `results/GDI/benchmark.csv` | GCN, SkipGNN, AMS, heuristic only |
| `results/*/summary.json` | Per-run histories (learning curves) plus metrics |
| `results/*/pr_ams_seed{42,123,7}.npz` | AMS precision–recall arrays |
| `results/ablation.csv` | DTI AMS ladder |
| `results/robustness.csv` | DTI missing-edge sweep |
| `results/model_comparison.csv` | Mean over seeds (rewritten by `make_figures.py`) |

Column meanings: [training-and-evaluation.md](training-and-evaluation.md).

## Mean hard AUPRC (3 seeds)

Rounded from `model_comparison.csv` after figure generation. Prefer the CSV for any quote.

| Model | DTI | DDI | PPI | GDI |
| --- | --- | --- | --- | --- |
| GCN | 0.731 | 0.658 | 0.609 | 0.749 |
| SkipGNN | 0.695 | 0.724 | 0.622 | 0.720 |
| AMS | 0.798 | 0.912 | 0.674 | 0.829 |
| Heuristic | 0.776 | 0.672 | 0.578 | **0.842** |
| SkipGATv2 | **0.804** | 0.900 | **0.779** | — |
| 3-hop | 0.803 | 0.895 | 0.688 | — |
| Contrastive | 0.800 | 0.905 | 0.657 | — |

Uniform AUPRC is much closer across neural models (often > 0.91). The hard bank is where skip-graph design shows up: AMS lifts DTI/DDI/GDI over SkipGNN; SkipGATv2 leads PPI hard AUPRC; the bipartite 3-walk heuristic is strongest on GDI hard.

## Known gap

**GDI has no `gat` / `3hop` / `contrastive` rows.** The GDI skip graph is on the order of 20k nodes with millions of skip edges. Chunked GAT fits in T4 memory but costs ~15 minutes per epoch at batch 256. Stage 4 finished DDI and PPI extra models; GDI GAT was cancelled to keep those CSVs. Do not invent numbers for the missing cells.

## How to quote a number

1. Open `results/<DATASET>/benchmark.csv`.
2. Average the three seeds, or cite one seed explicitly.
3. Say **uniform** or **hard**, and **AUPRC** or **AUROC**.
4. If you retrain, merge rather than overwrite unrelated rows.
