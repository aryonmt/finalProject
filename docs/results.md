# Results tables

Canonical numbers are the CSVs in git, not screenshots. Protocol and validity: [training-protocol.md](training-protocol.md).

## Files

| Path | Contents |
| --- | --- |
| `results/DTI/benchmark.csv` | All 7 models × 3 seeds |
| `results/DDI/benchmark.csv` | All 7 models × 3 seeds |
| `results/PPI/benchmark.csv` | All 7 models × 3 seeds |
| `results/GDI/benchmark.csv` | All 7 models × 3 seeds |
| `results/*/summary.json` | Per-run histories (learning curves) plus metrics |
| `results/*/pr_ams_seed{42,123,7}.npz` | AMS precision–recall arrays (seed 42 used in fig4) |
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
| SkipGATv2 | **0.804** | 0.900 | **0.778** | 0.837 |
| 3-hop | 0.803 | 0.895 | 0.688 | 0.832 |
| Contrastive | 0.800 | 0.905 | 0.657 | 0.833 |

Uniform AUPRC is much closer across neural models (often > 0.91). The hard bank is where skip-graph design shows up: AMS lifts DTI/DDI over SkipGNN; SkipGATv2 leads PPI hard AUPRC; on GDI the bipartite 3-walk heuristic is still strongest, with SkipGATv2 / 3-hop / contrastive in the AMS band after the per-batch re-run.

GDI extras in git are the Stage 5 **per-batch encode** job (batch 1024). Do not mix them with the discarded encode-once zip.

## How to quote a number

1. Open `results/<DATASET>/benchmark.csv`.
2. Average the three seeds, or cite one seed explicitly.
3. Say **uniform** or **hard**, and **AUPRC** or **AUROC**.
4. If you retrain, merge rather than overwrite unrelated rows.
