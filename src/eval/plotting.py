from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PALETTE = {
    "SkipGNN Uniform": "#7293CB",
    "AMS Uniform": "#2E5B88",
    "SkipGNN Hard": "#E1974C",
    "AMS Hard": "#AB4E19",
}

MODEL_PALETTE = {
    "gcn": "#7A7A7A",
    "skipgnn": "#7293CB",
    "ams": "#2E5B88",
    "heuristic": "#8C6D31",
    "gat": "#4C9A62",
    "3hop": "#C75D5D",
    "contrastive": "#7B6BBF",
}

MODEL_LABELS = {
    "gcn": "GCN",
    "skipgnn": "SkipGNN",
    "ams": "AMS",
    "heuristic": "Heuristic",
    "gat": "SkipGATv2",
    "3hop": "3-hop",
    "contrastive": "Contrastive",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_uniform_vs_hard(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="dataset", y="auprc", hue="series", ax=ax, palette=PALETTE)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUPRC")
    ax.set_xlabel("Dataset")
    ax.legend(frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation_ladder(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="step", y="hard_auprc", hue="dataset", ax=ax)
    ax.set_ylabel("Hard AUPRC")
    ax.set_xlabel("Ablation step")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(out_path)
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, sub in df.groupby("model"):
        ax.plot(sub["missing_frac"], sub["auprc_mean"], marker="o", label=model)
        ax.fill_between(
            sub["missing_frac"],
            sub["auprc_mean"] - sub["auprc_std"],
            sub["auprc_mean"] + sub["auprc_std"],
            alpha=0.15,
        )
    ax.set_xlabel("Missing-edge fraction")
    ax.set_ylabel("Test AUPRC")
    ax.legend(frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_grid(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    names = list(curves.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, name in zip(axes.ravel(), names):
        prec, rec = curves[name]
        ax.plot(rec, prec)
        ax.set_title(name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_mean(group: pd.DataFrame, primary: str, fallback: str | None = None) -> float | None:
    if primary in group.columns and group[primary].notna().any():
        return float(group[primary].mean())
    if fallback and fallback in group.columns and group[fallback].notna().any():
        return float(group[fallback].mean())
    return None


def _load_benchmarks(results_dir: Path) -> pd.DataFrame | None:
    bench_files = list(results_dir.rglob("benchmark.csv"))
    if not bench_files:
        return None
    frames = []
    for path in bench_files:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_all_models_uniform_hard(df: pd.DataFrame, out_path: Path) -> None:
    records: list[dict[str, Any]] = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        key = str(model).lower()
        label = MODEL_LABELS.get(key, str(model))
        uni = _metric_mean(group, "uniform_auprc", "auprc")
        hard = _metric_mean(group, "hard_auprc")
        if uni is not None:
            records.append({"dataset": dataset, "model": label, "split": "Uniform", "auprc": uni})
        if hard is not None:
            records.append({"dataset": dataset, "model": label, "split": "Hard", "auprc": hard})
    if not records:
        return
    plot_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, split in zip(axes, ("Uniform", "Hard")):
        sub = plot_df[plot_df["split"] == split]
        sns.barplot(data=sub, x="dataset", y="auprc", hue="model", ax=ax)
        ax.set_title(f"{split} AUPRC")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("AUPRC")
        ax.set_xlabel("Dataset")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_hard_auprc(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for dataset, group in df.groupby("dataset"):
        skip = group[group["model"].str.lower() == "skipgnn"]
        ams = group[group["model"].str.lower() == "ams"]
        skip_h = _metric_mean(skip, "hard_auprc")
        ams_h = _metric_mean(ams, "hard_auprc")
        if skip_h is None or ams_h is None:
            continue
        rows.append({"dataset": dataset, "delta": ams_h - skip_h})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = ["#2E5B88" if v >= 0 else "#AB4E19" for v in plot_df["delta"]]
    ax.bar(plot_df["dataset"], plot_df["delta"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Hard AUPRC (AMS − SkipGNN)")
    ax.set_xlabel("Dataset")
    fig.savefig(out_path)
    plt.close(fig)


def plot_hard_auprc_box(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.dropna(subset=["hard_auprc"]).copy()
    if sub.empty:
        return
    sub["label"] = sub["model"].str.lower().map(lambda k: MODEL_LABELS.get(k, k))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=sub, x="dataset", y="hard_auprc", hue="label", ax=ax)
    ax.set_ylabel("Hard AUPRC")
    ax.set_xlabel("Dataset")
    ax.legend(frameon=False, title="")
    fig.savefig(out_path)
    plt.close(fig)


def plot_uniform_vs_hard_auroc(df: pd.DataFrame, out_path: Path) -> None:
    records: list[dict[str, Any]] = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        key = str(model).lower()
        if key not in {"skipgnn", "ams"}:
            continue
        label = MODEL_LABELS[key]
        uni = _metric_mean(group, "uniform_auroc", "auroc")
        hard = _metric_mean(group, "hard_auroc")
        if uni is not None:
            records.append({"dataset": dataset, "auprc": uni, "series": f"{label} Uniform"})
        if hard is not None:
            records.append({"dataset": dataset, "auprc": hard, "series": f"{label} Hard"})
    if not records:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=pd.DataFrame(records), x="dataset", y="auprc", hue="series", ax=ax, palette=PALETTE)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Dataset")
    ax.legend(frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_learning_curves(results_dir: Path, out_path: Path) -> None:
    series: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in sorted(results_dir.rglob("summary.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        dataset = blob.get("dataset", path.parent.name)
        for run in blob.get("runs", []):
            history = run.get("history")
            model = str(run.get("model", "")).lower()
            if not history or model in {"heuristic", ""}:
                continue
            hist = pd.DataFrame(history)
            if "epoch" not in hist.columns or "val_auprc" not in hist.columns:
                continue
            series.setdefault((dataset, model), []).append(hist[["epoch", "val_auprc"]])
    if not series:
        return
    datasets = sorted({k[0] for k in series})
    n = len(datasets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.6 * nrows), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        for model in ("gcn", "skipgnn", "ams", "gat", "3hop", "contrastive"):
            runs = series.get((dataset, model))
            if not runs:
                continue
            merged = pd.concat(runs)
            mean = merged.groupby("epoch")["val_auprc"].mean()
            std = merged.groupby("epoch")["val_auprc"].std().fillna(0.0)
            ax.plot(mean.index, mean.values, label=MODEL_LABELS.get(model, model), color=MODEL_PALETTE.get(model))
            ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.12, color=MODEL_PALETTE.get(model))
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val AUPRC")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        rows.append(
            {
                "dataset": dataset,
                "model": str(model),
                "uniform_auprc": _metric_mean(group, "uniform_auprc", "auprc"),
                "hard_auprc": _metric_mean(group, "hard_auprc"),
                "uniform_auroc": _metric_mean(group, "uniform_auroc", "auroc"),
                "hard_auroc": _metric_mean(group, "hard_auroc"),
                "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else len(group),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "model"]).to_csv(out_path, index=False)


def generate_all_figures(results_dir: str | Path, output_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    output_dir = _ensure_dir(Path(output_dir))

    merged = _load_benchmarks(results_dir)
    if merged is not None:
        records: list[dict[str, Any]] = []
        for (dataset, model), group in merged.groupby(["dataset", "model"]):
            key = str(model).lower()
            if key in {"skipgnn", "skipgnn"}:
                label = "SkipGNN"
            elif key == "ams":
                label = "AMS"
            else:
                continue
            if "uniform_auprc" not in group.columns:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "auprc": float(group["uniform_auprc"].mean()),
                    "series": f"{label} Uniform",
                }
            )
            records.append(
                {
                    "dataset": dataset,
                    "auprc": float(group["hard_auprc"].mean()),
                    "series": f"{label} Hard",
                }
            )
        if records:
            plot_uniform_vs_hard(pd.DataFrame(records), output_dir / "fig1_uniform_vs_hard_auprc.png")
        plot_all_models_uniform_hard(merged, output_dir / "fig5_all_models_uniform_hard.png")
        plot_delta_hard_auprc(merged, output_dir / "fig6_delta_hard_auprc.png")
        plot_hard_auprc_box(merged, output_dir / "fig7_hard_auprc_seeds.png")
        plot_uniform_vs_hard_auroc(merged, output_dir / "fig8_uniform_vs_hard_auroc.png")
        write_comparison_table(merged, results_dir / "model_comparison.csv")
    plot_learning_curves(results_dir, output_dir / "fig9_learning_curves.png")

    abl_files = list(results_dir.rglob("ablation.csv"))
    if abl_files:
        plot_ablation_ladder(
            pd.concat([pd.read_csv(f) for f in abl_files], ignore_index=True),
            output_dir / "fig2_ablation_ladder.png",
        )
    rob_files = list(results_dir.rglob("robustness.csv"))
    if rob_files:
        plot_robustness(
            pd.concat([pd.read_csv(f) for f in rob_files], ignore_index=True),
            output_dir / "fig3_missing_edge_robustness.png",
        )

    pr_files = list(results_dir.rglob("pr_ams*.npz"))
    if pr_files:
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for path in sorted(pr_files):
            dataset = path.parent.name
            if dataset in curves:
                continue
            blob = np.load(path)
            prec = blob["prec"] if "prec" in blob.files else blob["precision"]
            rec = blob["rec"] if "rec" in blob.files else blob["recall"]
            curves[dataset] = (prec, rec)
        if curves:
            plot_pr_grid(curves, output_dir / "fig4_precision_recall_curves.png")


def pr_arrays(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prec, rec, _ = precision_recall_curve(labels, probs)
    return prec, rec


