# Training protocol and comparison validity

This is the protocol actually used for the numbers in `results/`. Quote a row only if this page says the comparison is valid.

Shared knobs for every neural model: hidden 64, decoder 64, dropout 0.5, Adam `lr=1e-3`, weight decay `5e-4`, grad clip 5, seeds **42 / 123 / 7**, one-hot features, `BCEWithLogitsLoss`. Contrastive adds batch InfoNCE with `λ=0.1`. Heuristic is resource-allocation on the train graph (no Adam). SkipGATv2 uses 4 attention heads.

The trainer encodes the **full graph on every decoder minibatch** and takes one Adam step per batch. `--encode-once` is a speed flag (one Adam step per epoch). It underfits GDI-scale models and must stay off for paper rows.

Eval (`predict_probs`) always encodes once. That is scoring, not training.

## What ran

Train sizes (pairs): DTI 21,194 · DDI 67,919 · PPI 32,651 · GDI 114,445. Step counts use `drop_last=True`.

| Dataset | Model | Epochs / patience | Batch | Encode | Adam steps / epoch | Vs AMS on that dataset |
| --- | --- | --- | --- | --- | --- | --- |
| DTI | GCN, SkipGNN, AMS | 30 / 8 | 128 | every batch | 165 | matched (AMS is the reference) |
| DTI | SkipGATv2, 3-hop, contrastive | 30 / 8 | 128 | every batch | 165 | **valid — same batch as AMS** |
| DTI | Heuristic | — | — | — | 0 | valid baseline |
| DDI | GCN, SkipGNN, AMS | 30 / 8 | 128 | every batch | 530 | matched |
| DDI | SkipGATv2, 3-hop, contrastive | 30 / 8 | 1024 | every batch | 66 | **valid — fewer steps than AMS** |
| DDI | Heuristic | — | — | — | 0 | valid baseline |
| PPI | GCN, SkipGNN, AMS | 30 / 8 | 128 | every batch | 255 | matched |
| PPI | SkipGATv2, 3-hop, contrastive | 30 / 8 | 1024 | every batch | 31 | **valid — fewer steps than AMS** |
| PPI | Heuristic | — | — | — | 0 | valid baseline |
| GDI | GCN, SkipGNN, AMS | 20 / 6 | 256 | every batch | 447 | matched |
| GDI | SkipGATv2, 3-hop, contrastive | 20 / 6 | 1024 | every batch | 111 | **valid — fewer steps than AMS** |
| GDI | Heuristic | — | — | — | 0 | valid baseline |

GDI YAML is `configs/gdi.yaml` (20 epochs, patience 6, batch 256). Extra-model Kaggle jobs pass `--batch-size 1024` and do **not** pass `--encode-once`.

## Discarded run (do not quote)

An earlier GDI extras job used `encode_once=True` and batch 2048: **one Adam step per epoch** (20 steps total). GAT hard AUPRC was ~0.66 vs AMS ~0.83. Those rows were never merged into git. The GDI `gat` / `3hop` / `contrastive` rows in `results/GDI/benchmark.csv` are the per-batch re-run (GAT hard AUPRC ~0.84).

Batch 2048 vs 1024 was not the failure mode. Encode-once vs encode-every-batch was.

## How to read a gap

| Situation | Meaning |
| --- | --- |
| Matched batch | Same encode loop, batch, epochs, patience, lr, seeds, splits. Quote freely. |
| Extra-model batch 1024 | Same encode loop; extras saw fewer Adam steps than AMS. Same class of caveat already used for DDI/PPI extras. |
| Encode-once | Broken optimizer schedule. Never quote against AMS / GCN / SkipGNN. |

Evaluation is identical for every neural row: val AUPRC early stop, best checkpoint restored, uniform official test, hard bank from test positives + train-graph sampling, `τ*` fit on validation only.
