from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EXTRAS = ("gat", "3hop", "contrastive")
BASELINES = ("gcn", "skipgnn", "ams", "heuristic")
SEEDS = {7, 42, 123}


def _models(path: Path) -> set[str]:
    import pandas as pd

    df = pd.read_csv(path)
    return {str(m).lower() for m in df["model"].unique()}


def _seeds_for(path: Path, model: str) -> set[int]:
    import pandas as pd

    df = pd.read_csv(path)
    sub = df[df["model"].astype(str).str.lower() == model]
    return {int(s) for s in sub["seed"].dropna()}


def test_paper_tables_have_expected_coverage() -> None:
    for ds in ("DTI", "DDI", "PPI"):
        path = RESULTS / ds / "benchmark.csv"
        assert path.exists(), path
        models = _models(path)
        for name in BASELINES + EXTRAS:
            assert name in models, f"{ds} missing {name}"
            assert SEEDS <= _seeds_for(path, name), f"{ds} {name} seeds"
        assert (RESULTS / ds / "pr_ams_seed42.npz").exists(), ds

    gdi = RESULTS / "GDI" / "benchmark.csv"
    models = _models(gdi)
    for name in BASELINES + EXTRAS:
        assert name in models, f"GDI missing {name}"
        assert SEEDS <= _seeds_for(gdi, name), f"GDI {name} seeds"
    assert (RESULTS / "GDI" / "pr_ams_seed42.npz").exists()
