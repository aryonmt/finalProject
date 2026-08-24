"""Fold a Kaggle zip (or unpacked folder) into the canonical results/figures layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("DTI", "DDI", "PPI", "GDI")
KEYS = ["dataset", "model", "seed"]
SKIP_COPY_SUFFIXES = {".pt", ".pth", ".ckpt"}
SKIP_COPY_NAMES = {".gitkeep", ".ds_store", "thumbs.db"}


def _find_payload(src: Path) -> Path:
    if (src / "results").is_dir() or (src / "figures").is_dir():
        return src
    nested = [p for p in src.iterdir() if p.is_dir()]
    for child in nested:
        if (child / "results").is_dir() or (child / "figures").is_dir():
            return child
    return src


def _unpack(src: Path, staging: Path) -> Path:
    if src.is_dir():
        return _find_payload(src)
    if src.suffix.lower() != ".zip":
        raise SystemExit(f"expected a zip or directory, got {src}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(staging)
    return _find_payload(staging)


def _merge_csv(dest: Path, incoming: Path) -> None:
    new = pd.read_csv(incoming)
    if dest.exists():
        old = pd.read_csv(dest)
        if all(k in old.columns for k in KEYS) and all(k in new.columns for k in KEYS):
            old_ix = old.set_index(KEYS).index
            new_ix = new.set_index(KEYS).index
            keep = old[~old_ix.isin(new_ix)]
            new = pd.concat([keep, new], ignore_index=True)
    sort_cols = [c for c in KEYS if c in new.columns]
    if sort_cols:
        new = new.sort_values(sort_cols).reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(dest, index=False)


def _merge_summary(dest: Path, incoming: Path) -> None:
    payload = json.loads(incoming.read_text(encoding="utf-8"))
    if dest.exists():
        old = json.loads(dest.read_text(encoding="utf-8"))
        old_runs = old.get("runs", [])
        new_runs = payload.get("runs", [])

        def key(run: dict) -> tuple:
            return (str(run.get("model", "")), int(run.get("seed", -1)))

        new_keys = {key(r) for r in new_runs}
        merged = dict(old)
        merged.update(payload)
        merged["runs"] = [r for r in old_runs if key(r) not in new_keys] + new_runs
        payload = merged
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def import_payload(
    payload: Path,
    repo: Path,
    archive_notebook: str | None,
    copy_embeddings: bool,
) -> dict[str, int]:
    counts = {"csv": 0, "json": 0, "npz": 0, "figures": 0, "notebooks": 0}
    src_results = payload / "results"
    if src_results.is_dir():
        for dataset in DATASETS:
            src_ds = src_results / dataset
            dest_ds = repo / "results" / dataset
            if not src_ds.is_dir():
                continue
            dest_ds.mkdir(parents=True, exist_ok=True)
            bench = src_ds / "benchmark.csv"
            if bench.exists():
                _merge_csv(dest_ds / "benchmark.csv", bench)
                counts["csv"] += 1
            summary = src_ds / "summary.json"
            if summary.exists():
                _merge_summary(dest_ds / "summary.json", summary)
                counts["json"] += 1
            for path in src_ds.rglob("*"):
                if not path.is_file():
                    continue
                name = path.name.lower()
                if name in SKIP_COPY_NAMES or path.suffix.lower() in SKIP_COPY_SUFFIXES:
                    continue
                if name in {"benchmark.csv", "summary.json"}:
                    continue
                if name.startswith("embeddings_") and not copy_embeddings:
                    continue
                rel = path.relative_to(src_ds)
                _copy_file(path, dest_ds / rel)
                if path.suffix.lower() == ".npz":
                    counts["npz"] += 1
        for name in ("ablation.csv", "robustness.csv", "model_comparison.csv"):
            src = src_results / name
            if src.exists():
                _copy_file(src, repo / "results" / name)
                counts["csv"] += 1

    src_figures = payload / "figures"
    dest_figures = repo / "figures"
    if src_figures.is_dir():
        dest_figures.mkdir(parents=True, exist_ok=True)
        for path in src_figures.rglob("*.png"):
            _copy_file(path, dest_figures / path.relative_to(src_figures))
            counts["figures"] += 1

    src_notebooks = payload / "notebooks"
    if src_notebooks.is_dir():
        nbs = sorted(src_notebooks.glob("*.ipynb"))
        if nbs:
            dest_name = archive_notebook or f"kaggle_{nbs[0].stem}_executed.ipynb"
            dest = repo / "notebooks" / Path(dest_name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nbs[0], dest)
            counts["notebooks"] += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Kaggle zip or unpacked folder")
    p.add_argument("--archive-notebook", default="", help="Destination executed notebook name")
    p.add_argument("--copy-embeddings", action="store_true")
    p.add_argument("--cleanup-src", action="store_true")
    p.add_argument("--make-figures", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    staging = ROOT / "results" / "_kaggle_bundle_staging"
    payload = _unpack(src, staging)
    counts = import_payload(
        payload,
        ROOT,
        args.archive_notebook or None,
        copy_embeddings=args.copy_embeddings,
    )
    if staging.exists() and src.is_file():
        shutil.rmtree(staging, ignore_errors=True)
    if args.cleanup_src and src.is_dir() and src != ROOT / "results":
        shutil.rmtree(src, ignore_errors=True)

    if args.make_figures:
        from src.eval.plotting import generate_all_figures

        generate_all_figures(ROOT / "results", ROOT / "figures", skip_tsne=True)

    print("imported", counts, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
