"""Run a val-only hyperparameter grid for the frozen-feature heads.

The grid uses the existing feature cache, so each run trains only the small
head.  Every run passes ``--skip-test``: the test split is untouched until a
single configuration is selected by validation Macro-F1.

Example::

    python scripts/tune_frozen_heads.py

After it finishes, run the printed command once to evaluate the selected
configuration on test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--owner", default="cai")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "tuning",
    )
    parser.add_argument(
        "--cells", nargs="+", choices=CELLS, default=list(CELLS)
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command(
    cell: str,
    config: dict,
    seed: int,
    args: argparse.Namespace,
    skip_test: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.model.train",
        "--cell",
        cell,
        "--seeds",
        str(seed),
        "--dropout",
        str(config["dropout"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--learning-rate",
        str(config["lr"]),
        "--weight-decay",
        str(config["wd"]),
        "--num-workers",
        str(args.num_workers),
        "--owner",
        args.owner,
        "--notes",
        f"frozen-head grid {config['name']}",
        "--experiment-log",
        str(args.output_root / "runs.csv"),
        "--checkpoint-dir",
        str(args.output_root / "checkpoints"),
        "--log-dir",
        str(args.output_root / "logs"),
    ]
    if skip_test:
        command.append("--skip-test")
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
        result.stdout + ("\n--- stderr ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
        raise RuntimeError(f"command failed ({result.returncode}):\n{tail}")


def main() -> int:
    args = _parse_args()
    runs_path = args.output_root / "runs.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.cells) * len(GRID) * len(args.seeds)
    index = 0
    for cell in args.cells:
        for seed in args.seeds:
            for config in GRID:
                index += 1
                command = _command(cell, config, seed, args, skip_test=True)
                label = f"{cell}/{config['name']}/seed{seed}"
                print(f"[{index}/{total}] {label}", flush=True)
                if args.dry_run:
                    print("  " + " ".join(command))
                    continue
                _run(
                    command,
                    args.output_root
                    / "console"
                    / f"{cell}_{config['name']}_seed{seed}.txt",
                )
                if not runs_path.exists():
                    raise RuntimeError(f"training did not write {runs_path}")
                frame = pd.read_csv(runs_path)
                row = frame.iloc[-1].to_dict()
                rows.append(
                    {
                        "cell": cell,
                        "config": config["name"],
                        "seed": seed,
                        "dropout": config["dropout"],
                        "hidden_dim": config["hidden_dim"],
                        "lr": config["lr"],
                        "wd": config["wd"],
                        "best_epoch": row["best_epoch"],
                        "val_macro_f1": row["val_macro_f1"],
                        "val_balanced_accuracy": row["val_balanced_accuracy"],
                        "val_qwk": row["val_qwk"],
                        "run_id": row["run_id"],
                        "log_path": row["log_path"],
                    }
                )
                print(
                    f"  val_f1={row['val_macro_f1']:.4f} "
                    f"val_bal={row['val_balanced_accuracy']:.4f} "
                    f"best_epoch={row['best_epoch']}",
                    flush=True,
                )

    if args.dry_run:
        return 0
    summary = pd.DataFrame(rows)
    summary_path = args.output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nTop configurations by validation Macro-F1:")
    print(
        summary.sort_values("val_macro_f1", ascending=False)
        .head(8)
        .to_string(index=False)
    )
    best = summary.sort_values("val_macro_f1", ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best.to_string())
    print("\nRun once on test with:")
    print(
        " ".join(
            _command(
                best["cell"],
                {
                    "name": best["config"],
                    "dropout": best["dropout"],
                    "hidden_dim": int(best["hidden_dim"]),
                    "lr": best["lr"],
                    "wd": best["wd"],
                },
                int(best["seed"]),
                args,
                skip_test=False,
            )
        )
    )
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
