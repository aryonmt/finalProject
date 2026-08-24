"""Train named models on one dataset and merge metrics into results/."""

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
from src.training.trainer import extract_node_embeddings, set_seed, train_one_model


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
    drop = {"hard_pairs", "hard_labels", "hard_probs", "test_probs"}
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items() if k not in drop}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _merge_benchmark(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "seed"]
    if path.exists():
        old = pd.read_csv(path)
        if all(k in old.columns for k in keys) and all(k in new_df.columns for k in keys):
            old_ix = old.set_index(keys).index
            new_ix = new_df.set_index(keys).index
            keep = old[~old_ix.isin(new_ix)]
            return pd.concat([keep, new_df], ignore_index=True)
    return new_df


def _merge_summary(path: Path, payload: dict) -> dict:
    if not path.exists():
        return payload
    old = json.loads(path.read_text(encoding="utf-8"))
    old_runs = old.get("runs", [])
    new_runs = payload.get("runs", [])

    def _key(run: dict) -> tuple:
        return (str(run.get("model", "")), int(run.get("seed", -1)))

    new_keys = {_key(r) for r in new_runs}
    kept = [r for r in old_runs if _key(r) not in new_keys]
    merged = dict(old)
    merged.update(payload)
    merged["runs"] = kept + new_runs
    return merged


def run_heuristics(bundle, seed: int) -> dict:
    set_seed(seed)
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.data.samplers import generate_bipartite_aware_hard_negatives

    scores_uniform = compute_heuristic_scores(
        bundle.adj_train,
        bundle.test.pairs,
        method="resource_allocation",
        bipartite=bundle.bipartite,
    )
    uniform_auroc = float(roc_auc_score(bundle.test.labels, scores_uniform))
    uniform_auprc = float(average_precision_score(bundle.test.labels, scores_uniform))

    pos_test = bundle.test.pairs[bundle.test.labels >= 0.5]
    hard_neg = generate_bipartite_aware_hard_negatives(
        pos_test,
        bundle.adj_train,
        bundle.known_positives,
        bundle.bipartite,
        bundle.n_source,
        bundle.n_target,
        seed=seed,
    )
    hard_pairs = np.concatenate([pos_test, hard_neg], axis=0)
    hard_labels = np.concatenate(
        [
            np.ones(len(pos_test), dtype=np.int32),
            np.zeros(len(hard_neg), dtype=np.int32),
        ]
    )
    scores_hard = compute_heuristic_scores(
        bundle.adj_train,
        hard_pairs,
        method="resource_allocation",
        bipartite=bundle.bipartite,
    )
    return {
        "uniform_auroc": uniform_auroc,
        "uniform_auprc": uniform_auprc,
        "hard_auroc": float(roc_auc_score(hard_labels, scores_hard)),
        "hard_auprc": float(average_precision_score(hard_labels, scores_hard)),
        "method": "resource_allocation",
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train named models on one dataset and merge rows into results/<DS>/benchmark.csv."
    )
    p.add_argument("--dataset", default="DTI", choices=["DTI", "DDI", "PPI", "GDI"])
    p.add_argument("--models", nargs="+", default=["gcn", "skipgnn", "ams", "heuristic"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="CPU smoke: 2 epochs, seed 42. Never treat these numbers as paper results.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--input-type", default="one_hot")
    p.add_argument(
        "--out",
        default=None,
        help="Results root. Default: results/ (paper tables) or results/temp/ with --quick.",
    )
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    args = p.parse_args()
    if args.out is None:
        args.out = "results/temp" if args.quick else "results"

    cfg = load_cfg(args.dataset)
    epochs = args.epochs or (2 if args.quick else int(cfg.get("epochs", 30)))
    batch_size = args.batch_size or int(cfg.get("batch_size", 128))
    seeds = args.seeds or ([42] if args.quick else list(cfg.get("seeds_stage1", [42, 123, 7])))
    device = device_of(args.device)
    if args.quick:
        print(
            f"SMOKE RUN (--quick): writing under {args.out}. Do not quote these metrics.",
            flush=True,
        )
    print(
        f"dataset={args.dataset} device={device} epochs={epochs} "
        f"batch_size={batch_size} seeds={seeds} models={args.models}",
        flush=True,
    )

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
                h.update({"model": "heuristic", "seed": seed, "dataset": args.dataset})
                payload["runs"].append(h)
                rows.append(h)
                print(
                    f"heuristic seed={seed} uniform_auprc={h['uniform_auprc']:.4f} "
                    f"hard_auprc={h['hard_auprc']:.4f}",
                    flush=True,
                )
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
                batch_size=batch_size,
                lr=float(cfg.get("lr", 1e-3)),
                weight_decay=float(cfg.get("weight_decay", 5e-4)),
                patience=int(cfg.get("patience", 8)),
                grad_clip=float(cfg.get("grad_clip", 5.0)),
                seed=seed,
            )
            rec = {
                "dataset": args.dataset,
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
            if args.save_embeddings:
                emb = extract_node_embeddings(model, bundle)
                np.savez_compressed(
                    out_dir / f"embeddings_{model_name}_seed{seed}.npz",
                    embeddings=emb.astype(np.float32),
                    n_source=np.int32(bundle.n_source),
                    bipartite=np.bool_(bundle.bipartite),
                    dataset=np.asarray(args.dataset),
                    model=np.asarray(model_name),
                    seed=np.int32(seed),
                )
                print(f"saved embeddings {emb.shape}", flush=True)
            if args.save_checkpoints:
                ckpt_dir = out_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"{model_name}_seed{seed}.pt")

    df = _merge_benchmark(out_dir / "benchmark.csv", pd.DataFrame(rows))
    df.to_csv(out_dir / "benchmark.csv", index=False)
    write_summary(_merge_summary(out_dir / "summary.json", payload), out_dir / "summary.json")
    print(df.to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
