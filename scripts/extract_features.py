"""Cache frozen MobileNetV3-Small features for every accepted window.

The cache is aligned with ``features/windows.csv`` (or a stratified smoke
subset) and stores raw priors; normalization is fitted later on the train
split only.  The four model cells and both split protocols reuse this cache.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.common import load_config, sha256_file, write_json_atomic
from src.model.backbone import MobileNetV3FeatureExtractor
from src.model.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WindowDataset,
    load_manifest,
    prior_columns,
)


CACHE_FILES = (
    "features.npy",
    "valid_mask.npy",
    "priors.npy",
    "index.csv",
    "meta.json",
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _select_windows(manifest, limit: int):
    """Take a deterministic, class-stratified subset for smoke caches."""
    if limit <= 0:
        return manifest
    groups = manifest.groupby(["split", "label"], sort=False, group_keys=False)
    per_group = max(1, limit // max(1, groups.ngroups))
    return groups.head(per_group)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "features" / "windows.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "features" / "mobilenet_v3_small_16x576",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stratified window limit; 0 caches every accepted window",
    )
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument("--frame-batch", type=int, default=256)
    parser.add_argument(
        "--num-workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", dest="pretrained", action="store_true")
    group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    manifest = load_manifest(args.manifest)
    selected = _select_windows(manifest, args.limit).copy()
    if selected.empty:
        raise SystemExit("no windows selected")

    output_dir = Path(args.output_dir)
    existing = [output_dir / name for name in CACHE_FILES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"cache exists in {output_dir}; pass --overwrite to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    pretrained = config["model"]["pretrained"] if args.pretrained is None else args.pretrained
    backbone = MobileNetV3FeatureExtractor(pretrained=pretrained, freeze=True)
    backbone.to(device).eval()

    frames_per_window = int(config["preprocessing"]["frames_per_window"])
    face_size = int(config["preprocessing"]["face_size"])
    columns = prior_columns(selected)
    expected_priors = int(config["model"]["prior_dim"])
    if len(columns) != expected_priors:
        raise SystemExit(
            f"manifest has {len(columns)} prior columns, config expects {expected_priors}"
        )

    dataset = WindowDataset(
        selected,
        normalize_images=True,
        frames_per_window=frames_per_window,
        face_size=face_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_windows,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    total = len(dataset)
    features = np.empty((total, frames_per_window, backbone.feature_dim), dtype=np.float16)
    masks = np.empty((total, frames_per_window), dtype=bool)
    priors = np.empty((total, len(columns)), dtype=np.float32)
    written = 0
    progress = tqdm(loader, desc=f"windows ({total})", unit="win")
    for inputs, valid_mask, prior, _label, _index in progress:
        batch_size = int(inputs.shape[0])
        frames = inputs.reshape(batch_size * frames_per_window, 3, face_size, face_size)
        chunks = []
        for start in range(0, frames.shape[0], args.frame_batch):
            chunk = frames[start:start + args.frame_batch].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded = backbone(chunk)
            chunks.append(encoded.to(torch.float16).cpu())
        encoded = torch.cat(chunks, dim=0).reshape(
            batch_size, frames_per_window, backbone.feature_dim
        )
        features[written:written + batch_size] = encoded.numpy()
        masks[written:written + batch_size] = np.asarray(valid_mask, dtype=bool)
        priors[written:written + batch_size] = np.asarray(prior, dtype=np.float32)
        written += batch_size
        progress.set_postfix(done=f"{written}/{total}")
    progress.close()
    if written != total:
        raise RuntimeError(f"wrote {written} windows, expected {total}")

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "valid_mask.npy", masks)
    np.save(output_dir / "priors.npy", priors)
    index = selected.copy()
    index.insert(0, "cache_row", np.arange(len(index), dtype=np.int64))
    index.insert(1, "manifest_row", selected.index.to_numpy(dtype=np.int64))
    index.to_csv(output_dir / "index.csv", index=False)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "num_windows": int(total),
        "frames_per_window": frames_per_window,
        "face_size": face_size,
        "feature_dim": int(backbone.feature_dim),
        "dtype": "float16",
        "backbone": config["model"]["backbone"],
        "pretrained": bool(pretrained),
        "freeze_backbone": True,
        "prior_columns": list(columns),
        "image_normalization": {
            "mean": IMAGENET_MEAN.flatten().tolist(),
            "std": IMAGENET_STD.flatten().tolist(),
            "source": "ImageNet",
        },
        "config_sha256": sha256_file(ROOT / "configs" / "default.yaml"),
        "manifest_sha256": sha256_file(args.manifest),
        "limit": int(args.limit),
        "batch_windows": int(args.batch_windows),
        "frame_batch": int(args.frame_batch),
    }
    write_json_atomic(output_dir / "meta.json", meta)

    print(f"cache:        {output_dir}")
    print(f"windows:      {total}")
    print(f"features:     {features.shape} {features.dtype}")
    print(f"valid_mask:   {masks.shape} {masks.dtype}")
    print(f"priors:       {priors.shape} {priors.dtype}")
    print(f"device:       {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
