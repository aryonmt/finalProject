from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.plotting import generate_all_figures, select_pr_curve_files


def test_generate_all_figures_from_tiny_tables(tmp_path: Path) -> None:
    results = tmp_path / "results"
    dti = results / "DTI"
    dti.mkdir(parents=True)
    rows = []
    for model in ("gcn", "skipgnn", "ams"):
        for seed, hard in ((42, 0.70), (123, 0.71), (7, 0.72)):
            rows.append(
                {
                    "dataset": "DTI",
                    "model": model,
                    "seed": seed,
                    "uniform_auprc": 0.90,
                    "hard_auprc": hard + (0.05 if model == "ams" else 0.0),
                    "uniform_auroc": 0.91,
                    "hard_auroc": 0.68,
                }
            )
    pd.DataFrame(rows).to_csv(dti / "benchmark.csv", index=False)
    out = tmp_path / "figures"
    generate_all_figures(results, out, skip_tsne=True)
    assert (out / "fig1_uniform_vs_hard_auprc.png").exists()
    assert (out / "fig5_all_models_uniform_hard.png").exists()
    assert (out / "fig10_extra_architectures_hard_auprc.png").exists()
    assert (results / "model_comparison.csv").exists()


def test_select_pr_curve_prefers_seed42(tmp_path: Path) -> None:
    ddi = tmp_path / "DDI"
    ddi.mkdir()
    seed123 = ddi / "pr_ams_seed123.npz"
    seed42 = ddi / "pr_ams_seed42.npz"
    seed123.write_bytes(b"")
    seed42.write_bytes(b"")
    chosen = select_pr_curve_files([seed123, seed42])
    assert chosen["DDI"] == seed42

