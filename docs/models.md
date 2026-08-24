# Models

Every encoder implements `encode` / `decode` / `forward(x, adj, pairs, adj_skip)` and returns `(logits, embeddings)` at eval time. Contrastive SkipGNN returns a third value (InfoNCE) **only while `model.training` is true**.

Build via `src.models.build_model(name, bundle, hidden1=64, hidden2=64, decoder_hidden=64, dropout=0.5)`.

## Shared pieces

- `SparseGraphConvolution` (`src/models/layers.py`): `X W` then sparse `A @ support`.
- Concat decoder (GCN, SkipGNN): `[h_u ; h_v]`.
- Four-way decoder (AMS, GAT, 3-hop, contrastive): `[h_u ; h_v ; h_u ⊙ h_v ; |h_u − h_v|]` plus BatchNorm.

## `gcn` — `StandardGCN`

Two Laplacian GCN layers on `f_orig` only. `skip_kind = "none"`. Control for “message passing without a skip graph”.

## `skipgnn` — `SkipGNNBaseline`

Bug-free replica of Huang et al.: binary skip, **additive** original/skip fusion, concat decoder. `skip_kind = "binary"`.

## `ams` — `AMSSkipGNN`

Three changes on top of SkipGNN:

1. Weighted resource-allocation skip (`f_skip_weighted`).
2. Sigmoid gate mixing original vs skip channels at each depth (`use_gate=True`).
3. Four-way decoder (`four_way_decoder=True`).

`skip_kind = "weighted"`. Ablations (`make_ablation` / `scripts/run_ablation.py`):

| Step | What changes |
| --- | --- |
| `0_skipgnn` | Binary skip, additive fusion, concat decoder |
| `1_weighted` | Same architecture, weighted skip |
| `2_gated` | Gate on, concat decoder |
| `3_full` | Gate + four-way decoder (paper AMS) |

## `heuristic` — resource allocation

Not a neural net. Homogeneous graphs: one-hop RA. Bipartite graphs: 3-walk RA (`_bipartite_three_walk_scores`) because 1-hop common neighbors are empty.

## `gat` — `SkipGATv2`

Same gated original/skip skeleton as AMS, but each graph convolution is sparse GATv2. Attention is **chunked** (`_ChunkedSparseGATv2`) so peak memory is O(chunk) rather than O(E). Chunk size: env `GAT_EDGE_CHUNK` (default `131072`). Needed because the GDI skip graph has millions of edges.

`skip_kind = "weighted"`.

## `3hop` — `ThreeHopSkipGNN`

Three parallel GCN streams (1-hop original, 2-hop skip, 3-hop return) fused with a learned path-attention softmax. `skip_kind = "three_hop_bundle"`.

## `contrastive` — `ContrastiveSkipGNN`

Two towers (original vs skip), gated mix for the decoder, plus a batch InfoNCE between projected views. The trainer adds `lambda_cl * cl_loss` to BCE when `forward` returns three tensors.

`skip_kind = "weighted"`.
