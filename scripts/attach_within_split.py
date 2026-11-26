"""Attach a window-level split column to a frozen-feature cache index.

The training dataset reads ``<cache-dir>/index.csv`` unconditionally, so this
script creates a new cache directory containing copies of the cached arrays and
an ``index.csv`` with the extra split column.  The original cache is never
modified.

Example:

    python scripts/attach_within_split.py \
      --cache-dir features/mobilenet_v3_small_16x576 \
      --split-file splits/within_subject_split.csv \
      --output-dir features/mobilenet_v3_small_16x576_within
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import read_json, sha256_file, write_json_atomic


CACHE_ARRAYS = ("features.npy", "valid_mask.npy", "priors.npy")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "features" / "mobilenet_v3_small_16x576",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "splits" / "within_subject_split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "features" / "mobilenet_v3_small_16x576_within",
    )
    parser.add_argument("--split-column", default="within_split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = _resolve(args.cache_dir)
    split_path = _resolve(args.split_file)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists; pass --overwrite to replace it")

    index = pd.read_csv(cache_dir / "index.csv", dtype={"subject_id": str})
    for column in ("npz_path", "cache_row"):
        if column not in index.columns:
            raise SystemExit(f"cache index is missing {column!r}")
    assignment = pd.read_csv(split_path, dtype={"subject_id": str})
    if args.split_column not in assignment.columns:
        raise SystemExit(f"split file has no {args.split_column!r} column")
    if assignment["npz_path"].duplicated().any():
        raise SystemExit("split file contains duplicate npz_path values")

    merged = index.merge(
        assignment[["npz_path", args.split_column]],
        on="npz_path",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(index):
        raise SystemExit("merge changed the number of cached windows")
    if merged[args.split_column].isna().any():
        missing = merged.loc[merged[args.split_column].isna(), "npz_path"].head(5).tolist()
        raise SystemExit(f"split file does not cover cached windows, first missing: {missing}")
    if set(merged[args.split_column].astype(str)) != {"train", "val", "test"}:
        raise SystemExit(f"unexpected split values: {sorted(merged[args.split_column].unique())}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in CACHE_ARRAYS:
        shutil.copy2(cache_dir / name, output_dir / name)
    meta = read_json(cache_dir / "meta.json")
    meta = dict(meta)
    meta["split_file"] = str(split_path.relative_to(ROOT)) if split_path.is_relative_to(ROOT) else str(split_path)
    meta["split_file_sha256"] = sha256_file(split_path)
    meta["split_column"] = args.split_column
    meta["window_counts"] = {
        name: int((merged[args.split_column].astype(str) == name).sum())
        for name in ("train", "val", "test")
    }
    write_json_atomic(output_dir / "meta.json", meta)
    merged.to_csv(output_dir / "index.csv", index=False, lineterminator="\n")

    print(f"Wrote {output_dir}")
    print(f"windows: {meta['window_counts']}")
    print(f"split sha256: {meta['split_file_sha256']}")
    print(merged[args.split_column].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
