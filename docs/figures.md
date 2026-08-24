# Figures

All comparison plots are built from seed-level CSV rows with a **fixed dataset order** (DTI, DDI, PPI, GDI) and **fixed model order** (GCN, SkipGNN, AMS, Heuristic, SkipGATv2, 3-hop, Contrastive). Error bars are **standard deviation across seeds** (`seaborn.barplot(..., errorbar="sd")`). Missing GDI extra models appear as absent bars, not zeros.

Regenerate:

```bash
python scripts/make_figures.py --skip-tsne
```

| File | What it shows |
| --- | --- |
| `fig1_uniform_vs_hard_auprc.png` | SkipGNN vs AMS, uniform and hard AUPRC |
| `fig2_ablation_ladder.png` | DTI AMS steps 0–3, hard AUPRC |
| `fig3_missing_edge_robustness.png` | DTI test AUPRC vs dropped train-edge fraction |
| `fig4_precision_recall_curves.png` | AMS PR curves (seed 42, four datasets) |
| `fig5_all_models_uniform_hard.png` | Every trained model, both banks |
| `fig6_delta_hard_auprc.png` | AMS − SkipGNN hard AUPRC |
| `fig7_hard_auprc_seeds.png` | Box plot of hard AUPRC by seed |
| `fig8_uniform_vs_hard_auroc.png` | SkipGNN vs AMS AUROC |
| `fig9_learning_curves.png` | Val AUPRC vs epoch (mean ± SD) |
| `fig10_extra_architectures_hard_auprc.png` | AMS vs SkipGATv2 / 3-hop / contrastive |
| `fig_tsne_<DATASET>_<model>_seed<s>.png` | Node embeddings (optional; needs npz) |

Styling lives in `src/eval/plotting.py` (`PALETTE`, `MODEL_PALETTE`, 300 DPI, serif, no top/right spines).

t-SNE uses sklearn `TSNE` on at most 4000 sampled nodes, colored by source vs target type on bipartite graphs.
