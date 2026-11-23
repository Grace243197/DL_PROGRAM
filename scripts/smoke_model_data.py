"""Smoke-test the real window dataset and a small cached-feature overfit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.common import load_config
from src.model.dataset import (
    CachedFeatureDataset,
    PriorNormalizer,
    WindowDataset,
    load_manifest,
)
from src.model.heads import FrameClassifier


def _select_windows(manifest, limit: int):
    groups = manifest.groupby(["split", "label"], sort=False, group_keys=False)
    per_group = max(1, limit // max(1, groups.ngroups))
    return groups.head(per_group).copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "features" / "cache_smoke",
    )
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--use-prior", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    frames = int(config["preprocessing"]["frames_per_window"])
    face_size = int(config["preprocessing"]["face_size"])
    minimum = int(config["preprocessing"]["min_valid_frames_per_window"])

    manifest = load_manifest()
    subset = _select_windows(manifest, args.limit)
    dataset = WindowDataset(
        subset, frames_per_window=frames, face_size=face_size
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    inputs, mask, priors, labels, index = next(iter(loader))

    print(f"raw windows:   {len(dataset)}")
    print(f"inputs:        {tuple(inputs.shape)} {inputs.dtype}")
    print(f"valid_mask:    {tuple(mask.shape)} {mask.dtype}")
    print(f"priors:        {tuple(priors.shape)} {priors.dtype}")
    print(f"labels:        {labels.tolist()} (mapped 0/1/2)")
    print(f"index:         {index.tolist()[:5]}")
    if tuple(inputs.shape[1:]) != (frames, 3, face_size, face_size):
        raise AssertionError(f"unexpected input shape {tuple(inputs.shape)}")
    if tuple(mask.shape[1:]) != (frames,):
        raise AssertionError(f"unexpected mask shape {tuple(mask.shape)}")
    if priors.shape[1] != int(config["model"]["prior_dim"]):
        raise AssertionError(f"unexpected prior shape {tuple(priors.shape)}")
    if labels.min().item() < 0 or labels.max().item() > 2:
        raise AssertionError("labels are not mapped to 0/1/2")
    if (mask.sum(dim=1) < minimum).any():
        raise AssertionError("raw batch contains a window below the QC threshold")
    if not torch.isfinite(inputs).all() or not torch.isfinite(priors).all():
        raise AssertionError("raw batch contains non-finite values")
    print("raw dataset contract: PASS")

    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "meta.json").exists():
        print(f"cache not found at {cache_dir}; skipping overfit check")
        return 0

    cached = CachedFeatureDataset(cache_dir)
    normalizer = PriorNormalizer.fit(cached.prior_matrix(), cached.prior_columns)
    cached.prior_normalizer = normalizer
    model = FrameClassifier(
        feature_dim=cached.feature_dim,
        prior_dim=len(cached.prior_columns),
        use_prior=args.use_prior,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(
        cached, batch_size=args.cache_batch_size, shuffle=True, num_workers=0
    )
    model.train()
    initial_loss = None
    final_loss = None
    step = 0
    while step < args.overfit_steps:
        for batch_inputs, batch_mask, batch_priors, batch_labels, _ in train_loader:
            batch_priors_arg = batch_priors if args.use_prior else None
            logits = model(batch_inputs, batch_mask, batch_priors_arg).logits
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())
            step += 1
            if step >= args.overfit_steps:
                break

    model.eval()
    correct = 0
    total = 0
    eval_loader = DataLoader(
        cached, batch_size=args.cache_batch_size, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for batch_inputs, batch_mask, batch_priors, batch_labels, _ in eval_loader:
            batch_priors_arg = batch_priors if args.use_prior else None
            logits = model(batch_inputs, batch_mask, batch_priors_arg).logits
            correct += int((logits.argmax(dim=1) == batch_labels).sum())
            total += int(batch_labels.numel())

    print(f"cached windows: {len(cached)}  feature_dim={cached.feature_dim}")
    print(f"overfit steps:  {step}")
    print(f"loss:           {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"train accuracy: {correct / total:.3f}")
    if final_loss >= initial_loss:
        raise AssertionError("overfit loss did not decrease")
    print("cached overfit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
