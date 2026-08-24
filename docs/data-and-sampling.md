# Data and sampling

## Upstream splits

`scripts/fetch_data.py` clones [kexinhuang12345/SkipGNN](https://github.com/kexinhuang12345/SkipGNN.git) into `data/_upstream/` (gitignored) and copies **fold 1** only:

| Dataset | Upstream folder | Expected nodes | Bipartite? |
| --- | --- | --- | --- |
| DDI | `data/DDI/fold1` | 1514 | no |
| PPI | `data/PPI/fold1` | 5604 | no |
| DTI | `data/DTI/fold1` | 7343 (drugs then genes) | yes |
| GDI | `data/GDI/fold1` | 19783 (genes then diseases) | yes |

Do not rewrite fold-1 files to “fix” counts. The loader checks `n_nodes` / `n_source` against `src/data/constants.py` and raises if the typing heuristic drifted by more than one entity.

## Leak-free adjacency

`load_dataset_splits` (`src/data/loader.py`):

1. Maps entity strings to contiguous integer ids.
2. For DTI/GDI, orients every pair so column 0 is a source and column 1 is a target (`_orient_bipartite`).
3. Builds an **undirected** binary adjacency from **train positives only**.
4. Laplacian-normalizes that graph (`f_orig`, self-loops on).
5. Builds skip graphs from the same train adjacency (never from val/test).

Val and test positives are recorded in `known_positives` so hard-negative sampling cannot accidentally draw a true edge, but they are **not** written into `A`.

## Skip graphs

All skip operators zero the diagonal. Isolated nodes stay zero under Laplacian normalization (no `1e-5` degree clip).

| Tensor on `DatasetBundle` | Builder | Used by |
| --- | --- | --- |
| `f_orig` | `D^{-1/2}(A+I)D^{-1/2}` | every GNN |
| `f_skip_bin` | `sign(A A^T)` then Laplacian, no self-loops | SkipGNN baseline |
| `f_skip_weighted` | resource allocation `A D^{-1} A^T` | AMS, GAT, contrastive, 3-hop |
| `f_3hop` | `W2 D^{-1} A` (3-walk return) | 3-hop model |

`model.skip_kind` tells the trainer which tensor (or pair) to pass as `adj_skip`:

- `"none"` → `None` (GCN)
- `"binary"` → `f_skip_bin`
- `"weighted"` → `f_skip_weighted`
- `"three_hop_bundle"` → `(f_skip_weighted, f_3hop)`

## Features

Default `input_type=one_hot` is a sparse identity (`n_nodes × n_nodes`). Optional node2vec files (`*.emb`) exist upstream but are not required for the reported runs.

## Uniform vs hard negatives

**Uniform test bank** = the official `test.csv` labels.

**Hard test bank** = all test positives plus one degree-biased negative per positive (`generate_bipartite_aware_hard_negatives` in `src/data/samplers.py`):

- Homogeneous graphs (DDI, PPI): corrupt one endpoint by drawing uniformly from the same **degree quartile** as the true partner (four bins at the 25/50/75 percentiles of train degrees), never a known positive or self-loop.
- Bipartite graphs (DTI, GDI): the same quartile rule, but only the **target** endpoint is corrupted. Sources never pair with sources.

The F1 threshold is **not** fit on this bank. See [training-and-evaluation.md](training-and-evaluation.md).
