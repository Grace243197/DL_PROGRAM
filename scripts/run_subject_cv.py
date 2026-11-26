"""Run subject-level cross-validation for frozen-feature hyperparameter choice.

Only the 42 training subjects from the frozen 42/9/9 split are used.  Each
fold holds out whole subjects; the 9-subject test set is never touched.  The
goal is a stability check for the hyperparameters that were already explored,
not a replacement for the final test evaluation.

Example::

    python scripts/run_subject_cv.py

The output is ``experiments/cv/summary.csv`` with mean/std validation
Macro-F1 across folds for each cell/config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


GRID = [
    {"name": "base", "dropout": 0.2, "hidden_dim": 128, "lr": 1e-3, "wd": 1e-4},
    {"name": "drop5", "dropout": 0.5, "hidden_dim": 128, "lr": 1e-3, "wd": 1e-4},
    {"name": "small", "dropout": 0.5, "hidden_dim": 64, "lr": 1e-3, "wd": 1e-4},
    {"name": "small_lr4", "dropout": 0.5, "hidden_dim": 64, "lr": 1e-4, "wd": 1e-4},
    {"name": "small_lr4_wd2", "dropout": 0.5, "hidden_dim": 64, "lr": 1e-4, "wd": 1e-2},
    {"name": "drop5_lr4_wd2", "dropout": 0.5, "hidden_dim": 128, "lr": 1e-4, "wd": 1e-2},
    {"name": "small_wd2", "dropout": 0.5, "hidden_dim": 64, "lr": 1e-3, "wd": 1e-2},
    {"name": "drop2_lr4", "dropout": 0.2, "hidden_dim": 128, "lr": 1e-4, "wd": 1e-4},
]
CELLS = ("frame_prior", "gru_prior")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--owner", default="cai")
    parser.add_argument(
        "--cells", nargs="+", choices=CELLS, default=list(CELLS)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "cv",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_folds(args: argparse.Namespace) -> Path:
    index = pd.read_csv(
        ROOT / "features" / "mobilenet_v3_small_16x576" / "index.csv",
        dtype={"subject_id": str},
    )
    index["subject_id"] = index["subject_id"].str.zfill(2)
    subjects = sorted(index.loc[index["split"] == "train", "subject_id"].unique())
    if len(subjects) != 42:
        raise SystemExit(f"expected 42 train subjects, found {len(subjects)}")
    rng = np.random.RandomState(args.seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    fold_of = {subject: position % args.folds for position, subject in enumerate(shuffled)}
    table = pd.DataFrame(
        {"subject_id": shuffled, "fold": [fold_of[s] for s in shuffled]}
    )
    path = args.output_root / "folds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    counts = table["fold"].value_counts().sort_index().to_dict()
    print(f"folds: {path}")
    print(f"subjects per fold: {counts}")
    return path


def _command(
    cell: str,
    config: dict,
    fold: int,
    fold_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.model.train",
        "--cell",
        cell,
        "--seeds",
        str(args.train_seed),
        "--dropout",
        str(config["dropout"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--learning-rate",
        str(config["lr"]),
        "--weight-decay",
        str(config["wd"]),
        "--fold-file",
        str(fold_file),
        "--fold",
        str(fold),
        "--skip-test",
        "--num-workers",
        str(args.num_workers),
        "--owner",
        args.owner,
        "--notes",
        f"cv grid {config['name']}",
        "--experiment-log",
        str(args.output_root / "runs.csv"),
        "--checkpoint-dir",
        str(args.output_root / "checkpoints"),
        "--log-dir",
        str(args.output_root / "logs"),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    return command


def _run(command: list[str], console_path: Path) -> None:
    console_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    console_path.write_text(
        result.stdout
        + ("\n--- stderr ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise RuntimeError(f"command failed ({result.returncode}):\n{tail}")


def _completed_keys(runs_path: Path) -> set[tuple[str, int, int, str]]:
    if not runs_path.exists():
        return set()
    frame = pd.read_csv(runs_path)
    if not {"cell", "fold", "seed", "notes"}.issubset(frame.columns):
        return set()
    return set(
        zip(
            frame["cell"].astype(str),
            frame["fold"].astype(int),
            frame["seed"].astype(int),
            frame["notes"].astype(str),
        )
    )


def main() -> int:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    fold_file = _write_folds(args)
    runs_path = args.output_root / "runs.csv"
    completed = set() if args.no_resume else _completed_keys(runs_path)
    config_by_name = {config["name"]: config for config in GRID}
    total = len(args.cells) * len(GRID) * args.folds
    index = 0
    for cell in args.cells:
        for fold in range(args.folds):
            for config in GRID:
                index += 1
                note = f"cv grid {config['name']}"
                key = (cell, fold, args.train_seed, note)
                label = f"{cell}/{config['name']}/fold{fold}"
                if key in completed:
                    print(f"[{index}/{total}] {label} (already done)", flush=True)
                    continue
                print(f"[{index}/{total}] {label}", flush=True)
                command = _command(cell, config, fold, fold_file, args)
                if args.dry_run:
                    print("  " + " ".join(command))
                    continue
                _run(
                    command,
                    args.output_root
                    / "console"
                    / f"{cell}_{config['name']}_fold{fold}.txt",
                )

    if args.dry_run:
        return 0

    frame = pd.read_csv(runs_path)
    frame = frame[frame["notes"].astype(str).str.startswith("cv grid ")].copy()
    frame["config"] = frame["notes"].map(
        lambda note: (
            config_by_name[note.removeprefix("cv grid ")]["name"]
            if note.removeprefix("cv grid ") in config_by_name
            else None
        )
    )
    frame = frame.dropna(subset=["config"])
    frame["fold"] = frame["fold"].astype(int)
    summary = (
        frame.groupby(["cell", "config"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"),
            val_qwk_mean=("val_qwk", "mean"),
        )
        .sort_values(["cell", "val_macro_f1_mean"], ascending=[True, False])
    )
    summary_path = args.output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    frame.to_csv(args.output_root / "summary_folds.csv", index=False)
    print("\nCV summary (validation only):")
    print(summary.to_string(index=False))
    for cell in args.cells:
        cell_summary = summary[summary["cell"] == cell]
        if cell_summary.empty:
            continue
        base = cell_summary[cell_summary["config"] == "base"]
        best = cell_summary.iloc[0]
        if not base.empty:
            delta = best["val_macro_f1_mean"] - base.iloc[0]["val_macro_f1_mean"]
            print(
                f"\n{cell}: best={best['config']} "
                f"mean={best['val_macro_f1_mean']:.4f} "
                f"(base={base.iloc[0]['val_macro_f1_mean']:.4f}, delta={delta:+.4f})"
            )
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
