"""Train one cell of the 2x2 design from cached frozen-backbone features.

Examples
--------
Train CNN-Frame on the subject-disjoint split with one seed::

    python -m src.model.train --cell frame --seeds 42

Train all four cells with the three configured seeds::

    python -m src.model.train --cell all

The test split is evaluated once, after model selection on validation
Macro-F1.  Each run appends a row to ``experiments/runs.csv`` and writes a
JSON log plus a checkpoint under ``experiments/`` (both gitignored).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.common import load_config, sha256_file, write_json_atomic
from src.model.dataset import CachedFeatureDataset, PriorNormalizer
from src.model.factory import build_model
from src.model.metrics import (
    DEFAULT_CLASS_NAMES,
    aggregate_videos,
    bootstrap_subject_ci,
    video_metrics,
    window_metrics,
)


CELLS = {
    "frame": (False, False),
    "frame_prior": (False, True),
    "gru": (True, False),
    "gru_prior": (True, True),
}
RUN_COLUMNS = [
    "run_id", "timestamp_utc", "owner", "git_commit", "config_sha256",
    "cache_dir", "cache_num_windows", "split_column", "split_sha256",
    "fold", "fold_file",
    "cell", "temporal", "use_prior", "fusion", "seed",
    "epochs_requested", "epochs_run", "best_epoch", "early_stopped",
    "batch_size", "learning_rate", "weight_decay", "amp", "optimizer",
    "deterministic", "train_windows", "val_windows", "test_windows",
    "val_loss", "val_accuracy", "val_macro_f1", "val_balanced_accuracy",
    "val_qwk", "test_loss", "test_accuracy", "test_macro_f1",
    "test_balanced_accuracy", "test_qwk",
    "test_video_macro_f1_mean_prob", "test_video_macro_f1_majority",
    "test_video_num", "test_macro_f1_ci_low", "test_macro_f1_ci_high",
    "test_video_macro_f1_mean_prob_ci_low",
    "test_video_macro_f1_mean_prob_ci_high",
    "test_video_macro_f1_majority_ci_low",
    "test_video_macro_f1_majority_ci_high",
    "val_per_class_recall", "test_per_class_recall",
    "checkpoint", "log_path", "notes",
]


@dataclass
class RunContext:
    args: argparse.Namespace
    config: dict
    model_config: dict
    cache_meta: dict
    normalizer: PriorNormalizer
    train_ds: Dataset
    val_ds: Dataset
    test_ds: Dataset
    test_table: pd.DataFrame | None
    train_windows: int
    val_windows: int
    test_windows: int
    fold: int | None
    fold_file: str | None
    bootstrap_n: int
    bootstrap_seed: int
    split_hash: str
    config_sha256: str
    git_commit: str | None
    device: torch.device
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    amp: bool
    deterministic: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "features" / "mobilenet_v3_small_16x576",
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=None,
        help="CSV with subject_id,fold for subject-level cross-validation",
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--cell", choices=[*CELLS, "all"], default="frame",
        help="model cell to train; 'all' runs the four 2x2 cells",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fusion", choices=["film", "concat"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="tune on validation only; do not evaluate the test split",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--amp", dest="amp", action="store_true")
    group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--deterministic", dest="deterministic", action="store_true")
    group.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=20261105)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--owner", default="cai")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=ROOT / "experiments" / "runs.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "experiments" / "checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "experiments" / "logs",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _split_hash(table: pd.DataFrame, split_column: str) -> str:
    if split_column not in table.columns:
        raise ValueError(f"cache index has no split column {split_column!r}")
    pairs = sorted(
        zip(
            table["npz_path"].astype(str).tolist(),
            table[split_column].astype(str).tolist(),
        )
    )
    payload = "\n".join(f"{path}\t{split}" for path, split in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_fold_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"subject_id": str})
    table["subject_id"] = table["subject_id"].str.zfill(2)
    missing = {"subject_id", "fold"} - set(table.columns)
    if missing:
        raise ValueError(f"fold file is missing columns: {sorted(missing)}")
    table["fold"] = table["fold"].astype(int)
    return table


def _fold_hash(table: pd.DataFrame, roles: np.ndarray) -> str:
    pairs = sorted(
        zip(
            table["npz_path"].astype(str).tolist(),
            np.asarray(roles).astype(str).tolist(),
        )
    )
    payload = "\n".join(f"{path}\t{role}" for path, role in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _subset_indices(size: int, limit: int, seed: int) -> np.ndarray | None:
    if limit <= 0 or limit >= size:
        return None
    return np.sort(np.random.RandomState(seed).choice(size, limit, replace=False))


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator if shuffle else None,
    )


def _forward_loss(
    model: torch.nn.Module,
    batch,
    use_prior: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, valid_mask, priors, labels, _index = batch
    inputs = inputs.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    priors = priors.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    logits = model(
        inputs, valid_mask, priors if use_prior else None
    ).logits.float()
    loss = F.cross_entropy(logits, labels)
    return logits, labels, loss


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    use_prior: bool,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, labels, loss = _forward_loss(model, batch, use_prior, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, labels, loss = _forward_loss(model, batch, use_prior, device)
            loss.backward()
            optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += count
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    use_prior: bool,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets = []
    for batch in loader:
        logits, labels, loss = _forward_loss(model, batch, use_prior, device)
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += count
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    metrics = window_metrics(y_true, y_pred, DEFAULT_CLASS_NAMES)
    metrics["loss"] = total_loss / total
    metrics["accuracy"] = correct / total
    return metrics


@torch.inference_mode()
def _evaluate_detailed(
    model: torch.nn.Module,
    loader: DataLoader,
    use_prior: bool,
    device: torch.device,
    table: pd.DataFrame | None,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> dict:
    """Evaluate once and keep window, video, and bootstrap metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    probabilities = []
    predictions = []
    targets = []
    for batch in loader:
        logits, labels, loss = _forward_loss(model, batch, use_prior, device)
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += count
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    probs = torch.cat(probabilities).numpy()
    window = window_metrics(y_true, y_pred, DEFAULT_CLASS_NAMES)
    window["loss"] = total_loss / total
    window["accuracy"] = correct / total
    result = {"window": window}
    if table is None:
        return result
    table = table.reset_index(drop=True)
    if len(table) != len(y_true):
        raise ValueError(
            f"test table has {len(table)} rows, predictions have {len(y_true)}"
        )
    subjects = table["subject_id"].astype(str).to_numpy()
    videos = table["video_key"].astype(str).to_numpy()
    aggregate = aggregate_videos(y_true, y_pred, probs, videos, subjects)
    result["video"] = video_metrics(aggregate, DEFAULT_CLASS_NAMES)
    result["bootstrap"] = {
        "window_macro_f1": bootstrap_subject_ci(
            y_true, y_pred, subjects, "macro_f1", bootstrap_n, bootstrap_seed
        ),
        "video_macro_f1_mean_prob": bootstrap_subject_ci(
            aggregate["y_true"],
            aggregate["y_pred_mean"],
            aggregate["subjects"],
            "macro_f1",
            bootstrap_n,
            bootstrap_seed,
        ),
        "video_macro_f1_majority": bootstrap_subject_ci(
            aggregate["y_true"],
            aggregate["y_pred_majority"],
            aggregate["subjects"],
            "macro_f1",
            bootstrap_n,
            bootstrap_seed,
        ),
    }
    return result


def _append_run_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame.reindex(columns=RUN_COLUMNS)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row], columns=RUN_COLUMNS)
    frame.to_csv(path, index=False)


def _run_one(context: RunContext, cell: str, seed: int) -> dict:
    temporal, use_prior = CELLS[cell]
    _set_seed(seed, context.deterministic)

    model_config = copy.deepcopy(context.model_config)
    if context.args.fusion is not None:
        model_config["model"]["fusion"] = context.args.fusion
    if context.args.dropout is not None:
        model_config["model"]["dropout"] = context.args.dropout
    if context.args.hidden_dim is not None:
        model_config["model"]["hidden_dim"] = context.args.hidden_dim
    model = build_model(
        model_config, temporal=temporal, use_prior=use_prior
    ).to(context.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.learning_rate,
        weight_decay=context.weight_decay,
    )
    use_amp = context.amp and context.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_loader = _make_loader(
        context.train_ds, context.batch_size, True, seed,
        context.args.num_workers, context.device,
    )
    val_loader = _make_loader(
        context.val_ds, context.batch_size, False, seed,
        context.args.num_workers, context.device,
    )
    test_loader = _make_loader(
        context.test_ds, context.batch_size, False, seed,
        context.args.num_workers, context.device,
    )

    best_val = float("-inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    epochs_run = 0
    for epoch in range(1, context.epochs + 1):
        started = time.time()
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, use_prior,
            context.device, use_amp,
        )
        val_metrics = _evaluate(model, val_loader, use_prior, context.device)
        epochs_run = epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "elapsed_s": time.time() - started,
            }
        )
        print(
            f"[{cell} seed={seed}] epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val + 1e-9:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if context.patience > 0 and stale >= context.patience:
                print(f"[{cell} seed={seed}] early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    val_metrics = _evaluate(model, val_loader, use_prior, context.device)
    test_metrics = (
        None
        if context.args.skip_test
        else _evaluate_detailed(
            model,
            test_loader,
            use_prior,
            context.device,
            context.test_table,
            context.bootstrap_n,
            context.bootstrap_seed,
        )
    )

    timestamp = datetime.now(timezone.utc)
    fold_suffix = f"_fold{context.fold}" if context.fold is not None else ""
    run_id = (
        f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{cell}_seed{seed}{fold_suffix}"
    )
    checkpoint_path = context.args.checkpoint_dir / f"{run_id}.pt"
    log_path = context.args.log_dir / f"{run_id}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "run_id": run_id,
            "cell": cell,
            "temporal": temporal,
            "use_prior": use_prior,
            "seed": seed,
            "fold": context.fold,
            "fold_file": context.fold_file,
            "model_config": model_config,
            "model_state": best_state,
            "normalizer": context.normalizer.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    write_json_atomic(
        log_path,
        {
            "run_id": run_id,
            "timestamp_utc": timestamp.isoformat(),
            "cell": cell,
            "temporal": temporal,
            "use_prior": use_prior,
            "seed": seed,
            "fold": context.fold,
            "fold_file": context.fold_file,
            "model_config": model_config,
            "train_config": {
                "epochs_requested": context.epochs,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "early_stopped": epochs_run < context.epochs,
                "batch_size": context.batch_size,
                "learning_rate": context.learning_rate,
                "weight_decay": context.weight_decay,
                "amp": use_amp,
                "optimizer": "AdamW",
                "deterministic": context.deterministic,
            },
            "data": {
                "cache_dir": str(context.args.cache_dir),
                "cache_meta": context.cache_meta,
                "split_column": context.args.split_column,
                "split_sha256": context.split_hash,
                "fold": context.fold,
                "fold_file": context.fold_file,
                "train_windows": context.train_windows,
                "val_windows": context.val_windows,
                "test_windows": context.test_windows,
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(context.device),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if context.device.type == "cuda"
                    else None
                ),
            },
            "history": history,
            "val": val_metrics,
            "test": test_metrics,
            "normalizer": context.normalizer.state_dict(),
            "notes": context.args.notes,
        },
    )

    test_window = None if test_metrics is None else test_metrics["window"]
    test_video = None if test_metrics is None else test_metrics.get("video")
    test_boot = None if test_metrics is None else test_metrics.get("bootstrap")
    row = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "owner": context.args.owner,
        "git_commit": context.git_commit,
        "config_sha256": context.config_sha256,
        "cache_dir": str(context.args.cache_dir),
        "cache_num_windows": int(context.cache_meta["num_windows"]),
        "split_column": context.args.split_column,
        "split_sha256": context.split_hash,
        "fold": context.fold,
        "fold_file": context.fold_file,
        "cell": cell,
        "temporal": temporal,
        "use_prior": use_prior,
        "fusion": model_config["model"]["fusion"],
        "seed": seed,
        "epochs_requested": context.epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "early_stopped": epochs_run < context.epochs,
        "batch_size": context.batch_size,
        "learning_rate": context.learning_rate,
        "weight_decay": context.weight_decay,
        "amp": use_amp,
        "optimizer": "AdamW",
        "deterministic": context.deterministic,
        "train_windows": context.train_windows,
        "val_windows": context.val_windows,
        "test_windows": context.test_windows,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_qwk": val_metrics["quadratic_weighted_kappa"],
        "test_loss": None if test_window is None else test_window["loss"],
        "test_accuracy": None if test_window is None else test_window["accuracy"],
        "test_macro_f1": None if test_window is None else test_window["macro_f1"],
        "test_balanced_accuracy": (
            None if test_window is None else test_window["balanced_accuracy"]
        ),
        "test_qwk": (
            None if test_window is None else test_window["quadratic_weighted_kappa"]
        ),
        "test_video_macro_f1_mean_prob": (
            None if test_video is None else test_video["mean_prob"]["macro_f1"]
        ),
        "test_video_macro_f1_majority": (
            None if test_video is None else test_video["majority"]["macro_f1"]
        ),
        "test_video_num": None if test_video is None else test_video["num_videos"],
        "test_macro_f1_ci_low": (
            None if test_boot is None else test_boot["window_macro_f1"]["low"]
        ),
        "test_macro_f1_ci_high": (
            None if test_boot is None else test_boot["window_macro_f1"]["high"]
        ),
        "test_video_macro_f1_mean_prob_ci_low": (
            None
            if test_boot is None
            else test_boot["video_macro_f1_mean_prob"]["low"]
        ),
        "test_video_macro_f1_mean_prob_ci_high": (
            None
            if test_boot is None
            else test_boot["video_macro_f1_mean_prob"]["high"]
        ),
        "test_video_macro_f1_majority_ci_low": (
            None
            if test_boot is None
            else test_boot["video_macro_f1_majority"]["low"]
        ),
        "test_video_macro_f1_majority_ci_high": (
            None
            if test_boot is None
            else test_boot["video_macro_f1_majority"]["high"]
        ),
        "val_per_class_recall": json.dumps(val_metrics["per_class_recall"]),
        "test_per_class_recall": (
            None if test_window is None
            else json.dumps(test_window["per_class_recall"])
        ),
        "checkpoint": str(checkpoint_path),
        "log_path": str(log_path),
        "notes": context.args.notes,
    }
    _append_run_log(context.args.experiment_log, row)
    test_text = (
        "skipped" if test_window is None else f"{test_window['macro_f1']:.4f}"
    )
    print(
        f"[{cell} seed={seed}] best_epoch={best_epoch} "
        f"val_f1={val_metrics['macro_f1']:.4f} "
        f"test_f1={test_text} "
        f"checkpoint={checkpoint_path.name}"
    )
    return row


def main() -> int:
    args = _parse_args()
    config = load_config()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    epochs = args.epochs or int(config["train"]["epochs"])
    batch_size = args.batch_size or int(config["train"]["batch_size"])
    learning_rate = args.learning_rate or float(config["train"]["learning_rate"])
    weight_decay = args.weight_decay or float(config["train"]["weight_decay"])
    patience = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else int(config["train"]["early_stop_patience"])
    )
    amp = bool(config["train"].get("amp", True)) if args.amp is None else args.amp
    seeds = args.seeds or list(config["train"]["seeds"])

    if (args.fold_file is None) != (args.fold is None):
        raise SystemExit("--fold-file and --fold must be used together")
    if args.fold_file is not None and not args.skip_test:
        raise SystemExit("--fold-file is for cross-validation; pass --skip-test")

    full = CachedFeatureDataset(args.cache_dir, split_column=args.split_column)
    if args.fold_file is not None:
        args.split_column = "fold"
        fold_table = _load_fold_table(args.fold_file)
        if args.fold not in set(fold_table["fold"]):
            raise SystemExit(f"fold {args.fold} is not in {args.fold_file}")
        fold_subjects = set(fold_table["subject_id"])
        in_fold = full.table["subject_id"].isin(fold_subjects).to_numpy()
        positions = np.flatnonzero(in_fold)
        if len(positions) == 0:
            raise SystemExit("fold file subjects do not appear in the cache index")
        fold_map = dict(zip(fold_table["subject_id"], fold_table["fold"]))
        assigned = full.table["subject_id"].map(fold_map)
        is_val = assigned.iloc[positions].to_numpy() == args.fold
        roles = np.where(is_val, "val", "train")
        train_positions = positions[~is_val]
        val_positions = positions[is_val]
        if len(train_positions) == 0 or len(val_positions) == 0:
            raise SystemExit("fold produced an empty train or validation split")
        train_priors = full.prior_matrix()[train_positions]
        normalizer = PriorNormalizer.fit(train_priors, full.prior_columns)
        full.prior_normalizer = normalizer
        train_ds: Dataset = Subset(full, train_positions.tolist())
        val_ds: Dataset = Subset(full, val_positions.tolist())
        test_ds: Dataset = val_ds
        test_table = None
        test_windows = 0
        split_hash = _fold_hash(full.table.iloc[positions], roles)
        fold_id = args.fold
        fold_file = str(args.fold_file)
    else:
        train_full = CachedFeatureDataset(
            args.cache_dir, split="train", split_column=args.split_column
        )
        val_full = CachedFeatureDataset(
            args.cache_dir, split="val", split_column=args.split_column
        )
        test_full = CachedFeatureDataset(
            args.cache_dir, split="test", split_column=args.split_column
        )
        train_indices = _subset_indices(
            len(train_full), args.limit_train, args.subset_seed
        )
        val_indices = _subset_indices(
            len(val_full), args.limit_val, args.subset_seed
        )
        train_priors = train_full.prior_matrix()
        if train_indices is not None:
            train_priors = train_priors[train_indices]
        normalizer = PriorNormalizer.fit(train_priors, train_full.prior_columns)
        for dataset in (train_full, val_full, test_full):
            dataset.prior_normalizer = normalizer
        train_ds = (
            train_full
            if train_indices is None
            else Subset(train_full, train_indices.tolist())
        )
        val_ds = (
            val_full
            if val_indices is None
            else Subset(val_full, val_indices.tolist())
        )
        test_ds = test_full
        test_table = test_full.table
        test_windows = len(test_full)
        split_hash = _split_hash(full.table, args.split_column)
        fold_id = None
        fold_file = None

    model_config = copy.deepcopy(config)
    model_config["model"]["feature_dim"] = int(full.feature_dim)
    model_config["model"]["prior_dim"] = int(len(full.prior_columns))

    context = RunContext(
        args=args,
        config=config,
        model_config=model_config,
        cache_meta=full.meta,
        normalizer=normalizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        test_table=test_table,
        train_windows=len(train_ds),
        val_windows=len(val_ds),
        test_windows=test_windows,
        fold=fold_id,
        fold_file=fold_file,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
        split_hash=split_hash,
        config_sha256=sha256_file(ROOT / "configs" / "default.yaml"),
        git_commit=_git_commit(),
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        amp=amp,
        deterministic=args.deterministic,
    )

    fold_text = f"  fold={fold_id}" if fold_id is not None else ""
    print(
        f"device={device}  windows={len(full)}  "
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{test_windows}  "
        f"features={full.feature_dim}d  priors={len(full.prior_columns)}{fold_text}"
    )
    cells = list(CELLS) if args.cell == "all" else [args.cell]
    rows = []
    for cell in cells:
        for seed in seeds:
            rows.append(_run_one(context, cell, seed))
    print(f"finished {len(rows)} run(s); log={args.experiment_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
