"""Freeze the within-subject window split (H3 protocol).

Protocol frozen on 2026-11-26 by Mao and Cai:

* Source of truth is ``features/windows.csv`` (22,553 accepted windows); never
  glob ``data/windows/**/*.npz``.
* For each ``(subject_id, label)`` group, sort windows by source file and
  window index, shuffle with a fixed seed, then assign 70/15/15 to
  train/val/test.
* Windows are non-overlapping 5-second windows, so this random window split
  does not leak identical frames; residual temporal correlation is a disclosed
  caveat, not a hidden one.
* The original ``split`` column (subject-disjoint 42/9/9) is never modified.
  The new assignment is written to a separate ``within_split`` column.

The output is frozen: re-running with the same configuration must reproduce
exactly the same file.  Use ``--force`` only when intentionally re-freezing.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, load_config, read_json, sha256_file, write_json_atomic


LABELS = [0, 5, 10]
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {"subject_id", "label", "npz_path", "source_file", "window_index"}
OUTPUT_COLUMNS = [
    "npz_path",
    "subject_id",
    "source_file",
    "video_key",
    "label",
    "class_name",
    "window_index",
    "subject_split",
    "within_split",
]


def _allocate(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return (train, val, test) counts for one group, with no empty split."""
    if n < 3:
        raise ValueError(f"group has only {n} windows; need at least 3 for 70/15/15")
    train_fraction, val_fraction, test_fraction = fractions
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, n - 2
    if n_train + n_val + n_test != n:
        raise AssertionError("allocation does not sum to the group size")
    return n_train, n_val, n_test


def assign_within_split(
    table: pd.DataFrame,
    seed: int,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.Series:
    """Assign every window to train/val/test within its subject x class group.

    The returned Series is indexed like ``table`` and is deterministic for a
    fixed seed and a fixed input table.
    """
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    if table.empty:
        raise ValueError("manifest is empty")
    if table["npz_path"].duplicated().any():
        raise ValueError("manifest contains duplicate npz_path values")

    subject = table["subject_id"].astype(str).str.zfill(2)
    label = table["label"].astype(int)
    keys = pd.DataFrame({"subject_id": subject, "label": label}, index=table.index)
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=table.index, dtype=object)

    for (subject_id, label_value), group in keys.groupby(
        ["subject_id", "label"], sort=True
    ):
        rows = table.loc[group.index]
        order = rows.sort_values(
            ["source_file", "window_index"], kind="mergesort"
        ).index.to_numpy()
        order = rng.permutation(order)
        n_train, n_val, _n_test = _allocate(len(order), fractions)
        assignment.loc[order[:n_train]] = "train"
        assignment.loc[order[n_train : n_train + n_val]] = "val"
        assignment.loc[order[n_train + n_val :]] = "test"
        if assignment.loc[order].isna().any():
            raise AssertionError(f"unassigned window in group ({subject_id}, {label_value})")

    if assignment.isna().any():
        missing_rows = assignment[assignment.isna()].index[:5].tolist()
        raise AssertionError(f"unassigned windows, first rows: {missing_rows}")
    if set(assignment.unique()) != set(SPLITS):
        raise AssertionError(f"unexpected split values: {sorted(assignment.unique())}")
    return assignment


def _build_output(table: pd.DataFrame, assignment: pd.Series) -> pd.DataFrame:
    missing = {"split", "npz_path", "subject_id", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    output = pd.DataFrame(
        {
            "npz_path": table["npz_path"].to_numpy(),
            "subject_id": table["subject_id"].astype(str).str.zfill(2).to_numpy(),
            "source_file": table["source_file"].to_numpy(),
            "video_key": table["video_key"].to_numpy(),
            "label": table["label"].astype(int).to_numpy(),
            "class_name": table["class_name"].to_numpy(),
            "window_index": table["window_index"].astype(int).to_numpy(),
            "subject_split": table["split"].to_numpy(),
            "within_split": assignment.to_numpy(),
        }
    )
    return output[OUTPUT_COLUMNS]


def _summary(output: pd.DataFrame) -> dict:
    counts = output.groupby("within_split").size().reindex(SPLITS, fill_value=0)
    per_class = pd.crosstab(output["within_split"], output["label"]).reindex(
        index=SPLITS, columns=LABELS, fill_value=0
    )
    return {
        "window_counts": {name: int(counts[name]) for name in SPLITS},
        "per_class_counts": {
            name: {str(label): int(per_class.loc[name, label]) for label in LABELS}
            for name in SPLITS
        },
        "subject_counts": {
            name: int(output.loc[output["within_split"] == name, "subject_id"].nunique())
            for name in SPLITS
        },
        "num_windows": int(len(output)),
        "num_groups": int(output.groupby(["subject_id", "label"]).ngroups),
    }


def _validate(table: pd.DataFrame, output: pd.DataFrame) -> None:
    if len(output) != len(table):
        raise AssertionError(f"output has {len(output)} rows, manifest has {len(table)}")
    if output["npz_path"].duplicated().any():
        raise AssertionError("output contains duplicate npz_path values")
    if output["npz_path"].tolist() != table["npz_path"].tolist():
        raise AssertionError("output rows do not cover the manifest in the same order")
    if output["within_split"].isna().any():
        raise AssertionError("output contains unassigned windows")
    if set(output["within_split"].unique()) != set(SPLITS):
        raise AssertionError(f"unexpected split values: {sorted(output['within_split'].unique())}")
    if not (output["subject_split"] == table["split"]).all():
        raise AssertionError("original subject-disjoint split column was modified")
    for name in SPLITS:
        subset = output.loc[output["within_split"] == name]
        if subset.empty:
            raise AssertionError(f"split {name!r} is empty")
        if set(subset["label"].astype(int).unique()) != set(LABELS):
            raise AssertionError(f"split {name!r} does not contain all three classes")
    # Within-subject means the same subjects intentionally appear in every
    # split; this is checked explicitly so the protocol is visible.
    subjects = {
        name: set(output.loc[output["within_split"] == name, "subject_id"])
        for name in SPLITS
    }
    if not all(subjects["train"] & subjects[name] for name in ("val", "test")):
        raise AssertionError("within-subject split unexpectedly has disjoint subjects")


def _load_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"subject_id": str})


def _same_assignment(existing: pd.DataFrame, output: pd.DataFrame) -> bool:
    if not {"npz_path", "within_split"}.issubset(existing.columns):
        return False
    if len(existing) != len(output):
        return False
    merged = existing[["npz_path", "within_split"]].merge(
        output[["npz_path", "within_split"]],
        on="npz_path",
        how="outer",
        suffixes=("_existing", "_new"),
        validate="one_to_one",
    )
    if len(merged) != len(output) or merged["within_split_existing"].isna().any():
        return False
    return bool((merged["within_split_existing"] == merged["within_split_new"]).all())


def build_within_split(
    config_path: Path | None = None,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    meta_path: Path | None = None,
    seed: int | None = None,
    force: bool = False,
) -> dict:
    config = load_config(config_path)
    settings = config.get("within_split", {})
    fractions = (
        float(settings.get("train", 0.70)),
        float(settings.get("val", 0.15)),
        float(settings.get("test", 0.15)),
    )
    seed = int(seed if seed is not None else settings.get("seed", config["split"]["seed"]))
    manifest_path = Path(manifest_path) if manifest_path is not None else ROOT / "features" / "windows.csv"
    output_path = Path(output_path) if output_path is not None else ROOT / "splits" / "within_subject_split.csv"
    meta_path = Path(meta_path) if meta_path is not None else output_path.with_suffix(".meta.json")

    table = pd.read_csv(manifest_path, dtype={"subject_id": str})
    table["subject_id"] = table["subject_id"].str.zfill(2)
    assignment = assign_within_split(table, seed=seed, fractions=fractions)
    output = _build_output(table, assignment)
    _validate(table, output)
    summary = _summary(output)

    existing = _load_existing(output_path)
    if existing is not None and not force:
        if _same_assignment(existing, output):
            print(f"Frozen split already matches: {output_path}")
            print(summary)
            return summary
        raise RuntimeError(
            f"Existing frozen split differs from configuration: {output_path}. "
            "Refusing to overwrite; pass --force only to re-freeze intentionally."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    output.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, output_path)

    meta = {
        "protocol": "subject_x_class_random_70_15_15",
        "description": (
            "Within each (subject_id, label) group, windows are sorted by "
            "source_file and window_index, shuffled with the fixed seed, and "
            "assigned 70/15/15. Windows are non-overlapping 5 s windows."
        ),
        "seed": seed,
        "fractions": {"train": fractions[0], "val": fractions[1], "test": fractions[2]},
        "source_manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        "output_sha256": sha256_file(output_path),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "original_split_untouched": True,
        **summary,
        "notes": [
            "Same 60 subjects appear in train/val/test by design; do not apply the subject-disjoint leakage check.",
            "Residual temporal correlation between adjacent non-overlapping windows remains a disclosed caveat.",
            "Use features/windows.csv as the source of truth; do not glob data/windows.",
        ],
    }
    write_json_atomic(meta_path, meta)
    print(f"Wrote {output_path}")
    print(f"Wrote {meta_path}")
    print(summary)
    print(f"sha256 {meta['output_sha256']}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the within-subject 70/15/15 window split (H3 protocol)"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen split; only for an intentional protocol change",
    )
    args = parser.parse_args()
    build_within_split(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        meta_path=args.meta,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
