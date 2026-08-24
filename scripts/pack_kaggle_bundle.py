"""Pack Kaggle artifacts into one zip the user can drop into the repo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUNDLE_NAME = "ams_skipgnn_kaggle_bundle"
NOTEBOOK_NAME = "kaggle_runner.ipynb"

IMPORT_TXT = """Drop this zip at the repo root.

After extract / import:
  results/     -> repo results/
  figures/     -> repo figures/
  notebooks/kaggle_runner.ipynb -> notebooks/executed/kaggle_run.ipynb
  MANIFEST.json -> git sha, STAGE, GPU, file list

Do not commit the zip. Keep results/, figures/, and the executed archive.
Prefer: python scripts/import_kaggle_bundle.py --src ams_skipgnn_kaggle_bundle.zip --make-figures
"""


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _copy_tree(src: Path, dest: Path) -> int:
    if not src.exists():
        return 0
    n = 0
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {".gitkeep", ".DS_Store"}:
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    return n


def _find_executed_notebook(repo: Path) -> Path | None:
    candidates = [
        Path("/kaggle/working/__notebook__.ipynb"),
        Path("/kaggle/working/__notebook_source__.ipynb"),
        Path("/kaggle/working") / NOTEBOOK_NAME,
        repo / "notebooks" / NOTEBOOK_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def pack_bundle(repo: Path, zip_stem: Path) -> Path:
    staging = zip_stem.parent / f".{BUNDLE_NAME}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_results = _copy_tree(repo / "results", staging / "results")
    n_figures = _copy_tree(repo / "figures", staging / "figures")

    nb_src = _find_executed_notebook(repo)
    nb_copied = False
    if nb_src is not None:
        dest_nb = staging / "notebooks" / NOTEBOOK_NAME
        dest_nb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb_src, dest_nb)
        nb_copied = True

    files = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(repo),
        "stage": os.environ.get("STAGE", "1"),
        "repo_cwd": str(repo),
        "python": sys.version.split()[0],
        "n_result_files": n_results,
        "n_figure_files": n_figures,
        "notebook_source": str(nb_src) if nb_src else None,
        "notebook_copied": nb_copied,
        "files": files,
        "import_map": {
            "results/": "results/",
            "figures/": "figures/",
            f"notebooks/{NOTEBOOK_NAME}": "notebooks/executed/kaggle_run.ipynb",
        },
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        manifest["torch_error"] = str(exc)

    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (staging / "IMPORT.txt").write_text(IMPORT_TXT, encoding="utf-8")

    zip_stem.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(shutil.make_archive(str(zip_stem), "zip", root_dir=staging))
    shutil.rmtree(staging, ignore_errors=True)
    return archive


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument(
        "--out",
        default="",
        help="zip stem (without .zip). Default: /kaggle/working/... if present, else repo root.",
    )
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    if args.out:
        stem = Path(args.out)
    elif Path("/kaggle/working").exists():
        stem = Path("/kaggle/working") / BUNDLE_NAME
    else:
        stem = repo / BUNDLE_NAME
    archive = pack_bundle(repo, stem)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"BUNDLE {archive}", flush=True)
    print(f"size_mb {size_mb:.2f}", flush=True)
    print("download this single zip and drop it at the repo root", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
