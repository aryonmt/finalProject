# Configs

`run_benchmark.py` loads `default.yaml`, then overlays `<dataset>.yaml` (lowercase filename: `dti.yaml`, …).

Keys: `hidden_dim`, `decoder_hidden`, `dropout`, `lr`, `weight_decay`, `batch_size`, `epochs`, `patience`, `grad_clip`, `input_type`, `seeds_stage1`.

Kaggle extra-model runs pass `--batch-size` on the CLI (1024 for DDI, PPI, and GDI extras). GDI *baselines* stay at YAML batch 256. See [docs/training-protocol.md](../docs/training-protocol.md).
