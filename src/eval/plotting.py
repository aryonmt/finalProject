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


def generate_all_figures(results_dir: str | Path, output_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    output_dir = _ensure_dir(Path(output_dir))

    bench_files = list(results_dir.rglob("benchmark.csv"))
    if bench_files:
        frames = []
        for path in bench_files:
            df = pd.read_csv(path)
            if "dataset" not in df.columns:
                df["dataset"] = path.parent.name
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
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


