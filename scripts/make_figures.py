"""Rebuild publication figures from cached results/ tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.plotting import generate_all_figures


def main() -> int:
    p = argparse.ArgumentParser(
        description="Regenerate paper figures from results/*.csv and PR .npz files."
    )
    p.add_argument("--results", default="results", help="Root of cached benchmark tables")
    p.add_argument("--out", default="figures", help="PNG output directory")
    p.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE (needs embeddings_*.npz, which are gitignored)",
    )
    args = p.parse_args()
    generate_all_figures(ROOT / args.results, ROOT / args.out, skip_tsne=bool(args.skip_tsne))
    print(f"figures written to {ROOT / args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
