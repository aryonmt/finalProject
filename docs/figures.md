# Figures

All comparison plots are built from seed-level CSV rows with a **fixed dataset order** (DTI, DDI, PPI, GDI) and **fixed model order** (GCN, SkipGNN, AMS, Heuristic, SkipGATv2, 3-hop, Contrastive). Error bars are **standard deviation across seeds** (`seaborn.barplot(..., errorbar="sd")`). Pairwise AMS-vs-SkipGNN charts were removed; ranking figures show all seven models.

Regenerate:

```bash
python scripts/make_figures.py --skip-tsne
```

| File | What it shows |
| --- | --- |
| `fig1_uniform_vs_hard_auprc.png` | All models, uniform and hard AUPRC (two panels) |
| `fig2_ablation_ladder.png` | DTI AMS steps 0–3, hard AUPRC |
| `fig3_missing_edge_robustness.png` | DTI test AUPRC vs dropped train-edge fraction |
| `fig4_precision_recall_curves.png` | Uniform-test PR overlay per dataset (seed 42). Drawn for every `pr_<model>_seed42.npz` on disk; current checkout stores AMS arrays. Rankings for all seven models are in fig1. |
| `fig5_hard_auprc_heatmap.png` | Mean hard AUPRC, dataset × model |
| `fig6_delta_hard_auprc.png` | Hard AUPRC of each model minus AMS (AMS = 0) |
| `fig7_hard_auprc_seeds.png` | Box plot of hard AUPRC by seed, all models |
| `fig8_uniform_vs_hard_auroc.png` | All models, uniform and hard AUROC |
| `fig9_learning_curves.png` | Val AUPRC vs epoch (mean ± SD), all neural models |
| `fig10_f1_at_tau.png` | Uniform-test F1 at validation-chosen τ* |
| `fig_tsne_<DATASET>_<model>_seed<s>.png` | Node embeddings (optional; needs npz) |

Styling lives in `src/eval/plotting.py` (`MODEL_PALETTE`, 300 DPI, serif, no top/right spines).

t-SNE uses sklearn `TSNE` on at most 4000 sampled nodes, colored by source vs target type on bipartite graphs.
