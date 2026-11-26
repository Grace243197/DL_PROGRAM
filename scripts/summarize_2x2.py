"""Summarize the 2x2 design from one experiment run log.

The design is temporal (frame vs GRU) x prior injection (absent vs present):

    frame, frame_prior, gru, gru_prior

Effects are computed per seed and then summarized as mean +/- seed SD.  This is
descriptive, not a significance test; with three seeds the seed SD is the
relevant uncertainty estimate.

Example:

    python scripts/summarize_2x2.py \
      --runs experiments/main/runs.csv \
      --metric test_macro_f1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CELLS = ("frame", "frame_prior", "gru", "gru_prior")


def _fmt(values: pd.Series) -> str:
    return f"{values.mean():+.4f} +/- {values.std(ddof=1):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/main/runs.csv"))
    parser.add_argument("--metric", default="test_macro_f1")
    args = parser.parse_args()

    table = pd.read_csv(args.runs)
    missing = set(CELLS) - set(table["cell"])
    if missing:
        raise SystemExit(f"run log is missing cells: {sorted(missing)}")

    pivot = table.pivot_table(index="seed", columns="cell", values=args.metric)
    pivot = pivot[list(CELLS)]
    seeds = pivot.index.tolist()

    frame = pivot[["frame", "frame_prior"]].mean(axis=1)
    gru = pivot[["gru", "gru_prior"]].mean(axis=1)
    no_prior = pivot[["frame", "gru"]].mean(axis=1)
    prior = pivot[["frame_prior", "gru_prior"]].mean(axis=1)
    temporal = gru - frame
    prior_effect = prior - no_prior
    interaction = (pivot["gru_prior"] - pivot["gru"]) - (
        pivot["frame_prior"] - pivot["frame"]
    )

    print(f"runs: {args.runs}  metric: {args.metric}  seeds: {seeds}")
    print("\nPer-seed values")
    print(pivot.round(4).to_string())
    print("\nCell means")
    print(pivot.mean().round(4).to_string())
    print("\nEffects")
    for name, values in (
        ("temporal (GRU - Frame)", temporal),
        ("prior (Prior - No prior)", prior_effect),
        ("interaction", interaction),
    ):
        print(f"{name:26s} {_fmt(values)}  seeds: " + ", ".join(f"{v:+.4f}" for v in values))
    print("\nMarginal means")
    print(f"frame-level {frame.mean():.4f}  GRU {gru.mean():.4f}")
    print(f"no prior    {no_prior.mean():.4f}  prior {prior.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
