"""Sparse-clone SkipGNN fold-1 splits into data/raw/{DDI,PPI,DTI,GDI}."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data" / "_upstream" / "SkipGNN"
RAW = ROOT / "data" / "raw"
REPO = "https://github.com/kexinhuang12345/SkipGNN.git"

COPY_MAP = {
    "DDI": {
        "splits": "data/DDI/fold1",
        "extras": [("data/DDI/ddi_unique_smiles.csv", "ddi_unique_smiles.csv")],
    },
    "PPI": {
        "splits": "data/PPI/fold1",
        "extras": [("data/PPI/protein_list.csv", "protein_list.csv")],
    },
    "DTI": {
        "splits": "data/DTI/fold1",
        "extras": [("data/DTI/entity_list.csv", "entity_list.csv")],
    },
    "GDI": {
        "splits": "data/GDI/fold1",
        "extras": [("data/GDI/entity_list.csv", "entity_list.csv")],
    },
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_upstream() -> None:
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    if (UPSTREAM / "data").exists():
        print(f"upstream already present: {UPSTREAM}")
        return
    if UPSTREAM.exists():
        shutil.rmtree(UPSTREAM)
    _run(["git", "clone", "--depth", "1", REPO, str(UPSTREAM)])


def copy_splits() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, spec in COPY_MAP.items():
        dest = RAW / name
        dest.mkdir(parents=True, exist_ok=True)
        src_splits = UPSTREAM / spec["splits"]
        for split in ("train.csv", "val.csv", "test.csv"):
            shutil.copy2(src_splits / split, dest / split)
            print(f"copied {name}/{split}")
        for rel, out_name in spec["extras"]:
            shutil.copy2(UPSTREAM / rel, dest / out_name)
            print(f"copied {name}/{out_name}")


def main() -> int:
    clone_upstream()
    copy_splits()
    print("DONE: data/raw is ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
