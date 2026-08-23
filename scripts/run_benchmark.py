from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset_splits
from src.eval.plotting import pr_arrays, write_summary
from src.models.factory import build_model
from src.models.heuristics import compute_heuristic_scores
from src.training.trainer import set_seed, train_one_model


def load_cfg(dataset: str) -> dict:
    default = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    specific = yaml.safe_load((ROOT / "configs" / f"{dataset.lower()}.yaml").read_text(encoding="utf-8"))
    default.update(specific)
    return default


def device_of(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items() if k not in {"hard_pairs", "hard_labels", "hard_probs", "test_probs", "history"}}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def run_heuristics(bundle, seed: int) -> dict:
    set_seed(seed)
    scores = compute_heuristic_scores(bundle.adj_train, bundle.test.pairs, method="resource_allocation")
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(bundle.test.labels, scores)),
        "auprc": float(average_precision_score(bundle.test.labels, scores)),
        "method": "resource_allocation",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="DTI", choices=["DTI", "DDI", "PPI", "GDI"])
    p.add_argument("--models", nargs="+", default=["gcn", "skipgnn", "ams"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--input-type", default="one_hot")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = load_cfg(args.dataset)
    epochs = args.epochs or (2 if args.quick else int(cfg.get("epochs", 30)))
    seeds = args.seeds or ([42] if args.quick else list(cfg.get("seeds_stage1", [42, 123, 7])))
    device = device_of(args.device)
    print(f"dataset={args.dataset} device={device} epochs={epochs} seeds={seeds} models={args.models}", flush=True)

    bundle = load_dataset_splits(args.dataset, input_type=args.input_type, device=device)
    print(
        f"loaded {bundle.name}: n={bundle.n_nodes} src={bundle.n_source} tgt={bundle.n_target} "
        f"train={len(bundle.train.labels)} val={len(bundle.val.labels)} test={len(bundle.test.labels)}",
        flush=True,
    )

    out_dir = ROOT / args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {"dataset": args.dataset, "device": str(device), "runs": []}

    for model_name in args.models:
        if model_name == "heuristic":
            for seed in seeds:
                h = run_heuristics(bundle, seed)
                h.update({"model": "heuristic", "seed": seed})
                payload["runs"].append(h)
                rows.append(h)
                print(f"heuristic seed={seed} auprc={h['auprc']:.4f}", flush=True)
            continue
        for seed in seeds:
            print(f"=== {model_name} seed={seed} ===", flush=True)
            model = build_model(
                model_name,
                bundle,
                hidden1=int(cfg.get("hidden_dim", 64)),
                hidden2=int(cfg.get("hidden_dim", 64)),
                decoder_hidden=int(cfg.get("decoder_hidden", 64)),
                dropout=float(cfg.get("dropout", 0.5)),
            )
            metrics = train_one_model(
                model,
                bundle,
                epochs=epochs,
                batch_size=int(cfg.get("batch_size", 128)),
                lr=float(cfg.get("lr", 1e-3)),
                weight_decay=float(cfg.get("weight_decay", 5e-4)),
                patience=int(cfg.get("patience", 8)),
                grad_clip=float(cfg.get("grad_clip", 5.0)),
                seed=seed,
            )
            rec = {
                "model": model_name,
                "seed": seed,
                "uniform_auprc": metrics["test_uniform@0.5"]["auprc"],
                "hard_auprc": metrics["test_hard@0.5"]["auprc"],
                "uniform_auroc": metrics["test_uniform@0.5"]["auroc"],
                "hard_auroc": metrics["test_hard@0.5"]["auroc"],
                "f1_tau": metrics["test_uniform@tau"]["f1"],
                "tau_star": metrics["tau_star"],
                "best_val_auprc": metrics["best_val_auprc"],
            }
            rows.append(rec)
            payload["runs"].append({"model": model_name, "seed": seed, **_jsonify(metrics)})
            print(
                f"{model_name} seed={seed} uniform_auprc={rec['uniform_auprc']:.4f} "
                f"hard_auprc={rec['hard_auprc']:.4f}",
                flush=True,
            )
            if model_name == "ams":
                prec, rec_arr = pr_arrays(bundle.test.labels, metrics["test_probs"])
                np.savez(out_dir / f"pr_ams_seed{seed}.npz", prec=prec, rec=rec_arr)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    write_summary(payload, out_dir / "summary.json")
    print(df.to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
