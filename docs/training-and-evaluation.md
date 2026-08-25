# Training and evaluation

## Loop

`train_one_model` in `src/training/trainer.py`:

1. `set_seed(seed)`.
2. Shuffle train pairs with `drop_last=True` so the last incomplete batch is not a silent regularizer.
3. Each batch: encoder on the **full** graph, decoder on the batch pairs, `BCEWithLogitsLoss`. Contrastive models add `lambda_cl * InfoNCE`.
4. Gradient clip (`grad_clip=5`).
5. After each epoch, score **validation AUPRC**. Keep the best state. Stop after `patience` epochs without improvement (default 8).
6. Restore the best state. Build the hard negative bank from **test positives** + train-graph sampling.
7. Score val, uniform test, and hard test. Fit `τ*` on val F1 only.

The encoder is invoked **once per decoder batch** by default (same protocol as AMS). `--encode-once` is a speed flag and underfits. What actually ran, and which comparisons are valid: [training-protocol.md](training-protocol.md). Eval still encodes once (`predict_probs`).

## Hyperparameters

Defaults live in `configs/default.yaml`. Dataset YAML files override keys.

Reported Kaggle runs used hidden 64, dropout 0.5, Adam `1e-3`, weight decay `5e-4`, and seeds **42, 123, 7**. Epoch cap and patience come from YAML:

| Dataset | epochs | patience | batch_size (baselines) | Extra-model batch (Kaggle) |
| --- | --- | --- | --- | --- |
| DTI / DDI / PPI | 30 | 8 | 128 | 1024 (DDI/PPI extras) |
| GDI | **20** (`configs/gdi.yaml`) | **6** | **256** | 1024 (extras, per-batch encode) |

`--quick` is a wiring check (2 epochs, seed 42) and writes under `results/temp/` so it cannot clobber paper CSVs.

## Dual-bank metrics

`evaluate_dual_bank` (`src/eval/metrics.py`):

- `tau_star` = argmax F1 on a 91-point grid in `[0.05, 0.95]`, **validation labels only**.
- For val / uniform test / hard test, report AUROC, AUPRC, F1, Brier at `0.5` and at `τ*`.

`scripts/run_benchmark.py` writes the ranking metrics used in figures:

- `uniform_auprc`, `uniform_auroc` ← official test at 0.5
- `hard_auprc`, `hard_auroc` ← hard bank at 0.5
- `f1_tau`, `tau_star`, `best_val_auprc`

AUPRC is the headline ranking metric on these imbalanced graphs. Hard AUPRC is the stress test (degree-biased negatives).

## Merging runs

`run_benchmark.py` **merges** by `(dataset, model, seed)`. A later Stage 4 run can add `gat` rows without wiping Stage 1 `ams` rows. `summary.json` keeps per-epoch `history` for learning-curve plots.

## CLI flags worth knowing

```text
python scripts/run_benchmark.py --dataset DTI --models gcn skipgnn ams \
    --seeds 42 123 7 --device auto --save-embeddings --save-checkpoints
```

`--save-embeddings` writes gitignored `embeddings_<model>_seed<seed>.npz` for t-SNE. `--save-checkpoints` writes `results/<DS>/checkpoints/` (also gitignored). `--encode-once` encodes the graph once per epoch (one Adam step); keep it off for paper GDI extras.
