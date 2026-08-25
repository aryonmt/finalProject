from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.plotting import collect_pr_curves_seed42, generate_all_figures, select_pr_curve_files


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
                    "f1_tau": 0.80,
                }
            )
    pd.DataFrame(rows).to_csv(dti / "benchmark.csv", index=False)
    out = tmp_path / "figures"
    generate_all_figures(results, out, skip_tsne=True)
    assert (out / "fig1_uniform_vs_hard_auprc.png").exists()
    assert (out / "fig5_hard_auprc_heatmap.png").exists()
    assert (out / "fig6_delta_hard_auprc.png").exists()
    assert (out / "fig8_uniform_vs_hard_auroc.png").exists()
    assert (out / "fig10_f1_at_tau.png").exists()
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


def test_collect_pr_curves_groups_models(tmp_path: Path) -> None:
    dti = tmp_path / "DTI"
    dti.mkdir()
    ams42 = dti / "pr_ams_seed42.npz"
    gat123 = dti / "pr_gat_seed123.npz"
    gat42 = dti / "pr_gat_seed42.npz"
    for path in (ams42, gat123, gat42):
        path.write_bytes(b"")
    grouped = collect_pr_curves_seed42([ams42, gat123, gat42])
    assert grouped["DTI"]["ams"] == ams42
    assert grouped["DTI"]["gat"] == gat42

