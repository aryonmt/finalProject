# System architecture

The runtime is a single-process PyTorch trainer around a leak-free `DatasetBundle`. There is no server, no database, and no multi-GPU wrapper.

```
data/raw/<DS>/*.csv
        │
        ▼
 load_dataset_splits  ──► DatasetBundle
        │                   adj_train, f_orig, f_skip_*, splits
        ▼
 build_model(name, bundle)
        │
        ▼
 train_one_model  ──► metrics dict (uniform + hard)
        │
        ▼
 results/<DS>/benchmark.csv  +  summary.json
        │
        ▼
 generate_all_figures  ──► figures/*.png
```

## Design constraints

- **No leakage.** Val/test positives never enter `A` or the skip graphs.
- **One factory.** Adding a model is: implement `skip_kind` + `forward`, register the name in `src/models/factory.py`, pass the CLI key to `run_benchmark.py`.
- **Merge, don’t clobber.** Benchmark CSV rows are keyed by dataset/model/seed so staged Kaggle runs can accumulate.
- **CPU tests, GPU papers.** `tests/test_smoke.py` uses tiny random graphs. Paper tables come from T4 notebooks.

## Module map

| Module | Responsibility |
| --- | --- |
| `src.data.loader` | CSV → ids → train-only graph → Laplacian tensors |
| `src.data.normalization` | Binary / RA / 3-hop skip operators |
| `src.data.samplers` | Uniform and degree-biased negatives |
| `src.models.*` | Encoders; `factory.build_model` |
| `src.training.trainer` | BCE loop, early stop, dual-bank eval |
| `src.eval.metrics` | Val-only τ*, AUROC/AUPRC/F1 |
| `src.eval.plotting` | Publication figures from CSVs |

## Kaggle vs laptop

The Kaggle notebook is **not** a second codebase. It clones this repo and calls the same scripts. The laptop is for `pytest`, figure regen, and importing the zip. See [kaggle.md](kaggle.md). Which runs are comparable: [training-protocol.md](training-protocol.md).
