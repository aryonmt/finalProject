# Data

`python scripts/fetch_data.py` fills `raw/{DDI,PPI,DTI,GDI}` from Huang et al. SkipGNN **fold 1**.

Each dataset folder contains `train.csv`, `val.csv`, `test.csv`, and an entity list. Node2vec `*.emb` files are not required for the reported one-hot runs and are gitignored if present.

`_upstream/` and `processed/` are gitignored caches. Do not commit zips of the graphs.
