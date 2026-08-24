"""Dump review-relevant project files into one text bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md"}
INCLUDE_NAMES = {".gitignore"}
INCLUDE_NOTEBOOKS = True

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "data",
    "results",
    "figures",
    "checkpoints",
    "runs",
    "wandb",
    "build",
    "dist",
    ".eggs",
}


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def _iter_files(root: Path, exclude: Path | None = None) -> list[Path]:
    skip = exclude.resolve() if exclude is not None else None
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if skip is not None and path.resolve() == skip:
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in SKIP_DIRS:
            continue
        if any(_should_skip_dir(part) for part in rel_parts[:-1] if part != "data"):
            continue
        if path.name in INCLUDE_NAMES or path.suffix.lower() in INCLUDE_SUFFIXES:
            out.append(path)
        elif INCLUDE_NOTEBOOKS and path.suffix.lower() == ".ipynb":
            out.append(path)
    return out


def _notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "code")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        chunks.append(f"# --- notebook cell {i} ({ctype}) ---\n{src.rstrip()}\n")
    return "\n".join(chunks)


def _file_text(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return _notebook_source(path)
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "review_bundle.txt"),
        help="output text file",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    files = _iter_files(ROOT, exclude=out_path)
    parts: list[str] = [
        f"# Project review bundle\n# root: {ROOT}\n# files: {len(files)}\n",
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        body = _file_text(path)
        parts.append(f"\n\n{'=' * 80}\nFILE: {rel}\n{'=' * 80}\n\n{body.rstrip()}\n")
    text = "".join(parts)
    out_path.write_text(text, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.1f} KB, {len(files)} files)")
    for path in files:
        print(f"  {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
