"""Datasets and prior normalization for UTA-RLDD window files.

The four model cells consume the same 5-second windows.  Two data paths share
one contract:

* :class:`WindowDataset` reads ``frames`` from the 160x160 BGR ``.npz`` files,
  converts them to RGB, and applies ImageNet normalization.
* :class:`CachedFeatureDataset` reads precomputed frozen-backbone features.

Both return :class:`WindowSample` with ``inputs`` ``[T, ...]``, ``valid_mask``
``[T]``, ``priors`` ``[D]``, ``label`` ``()``, and ``index`` ``()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.common import ROOT, read_json
from ..data.image_ops import bgr_to_rgb


LABEL_TO_INDEX = {0: 0, 5: 1, 10: 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
METADATA_COLUMNS = {
    "source_file", "video_key", "subject_id", "split", "label", "class_name",
    "window_index", "start_s", "end_s", "npz_path",
    "within_split", "split_within", "cache_row", "manifest_row",
}
DEFAULT_MANIFEST = ROOT / "features" / "windows.csv"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class WindowSample(NamedTuple):
    inputs: torch.Tensor
    valid_mask: torch.Tensor
    priors: torch.Tensor
    label: torch.Tensor
    index: torch.Tensor


@dataclass(frozen=True)
class PriorNormalizer:
    """Standardize the 34 window-level priors using train-split statistics."""

    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, columns: Sequence[str]) -> "PriorNormalizer":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(columns):
            raise ValueError(
                f"expected priors ({len(columns)} columns), got {matrix.shape}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(
            tuple(columns),
            mean.astype(np.float32),
            std.astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape[-1] != len(self.columns):
            raise ValueError(
                f"expected {len(self.columns)} priors, got shape {matrix.shape}"
            )
        if np.isnan(matrix).any():
            matrix = np.where(np.isnan(matrix), self.mean, matrix)
        return (matrix - self.mean) / self.std

    def state_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    """Read the accepted-window manifest, preserving subject IDs like ``01``."""
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    table = pd.read_csv(manifest_path, dtype={"subject_id": str})
    if "subject_id" in table:
        table["subject_id"] = table["subject_id"].str.zfill(2)
    if "label" in table:
        table["label"] = table["label"].astype(int)
    missing = {"npz_path", "split", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return table


def prior_columns(table: pd.DataFrame) -> list[str]:
    """Return the 34 numeric prior columns, independent of split column names."""
    columns = [column for column in table.columns if column not in METADATA_COLUMNS]
    if not columns:
        raise ValueError("manifest has no prior columns")
    return columns


def _resolve_npz_path(row: pd.Series) -> Path:
    path = Path(str(row["npz_path"]))
    return path if path.is_absolute() else ROOT / path


def _check_column_match(normalizer: PriorNormalizer, columns: Sequence[str]) -> None:
    if tuple(normalizer.columns) != tuple(columns):
        raise ValueError(
            "prior normalizer columns do not match this dataset: "
            f"{normalizer.columns} != {tuple(columns)}"
        )


class WindowDataset(Dataset):
    """Read frames from ``.npz`` files and return normalized RGB windows."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None = None,
        split_column: str = "split",
        prior_normalizer: PriorNormalizer | None = None,
        normalize_images: bool = True,
        frames_per_window: int = 16,
        face_size: int = 160,
    ) -> None:
        table = manifest
        if split is not None:
            if split_column not in table.columns:
                raise ValueError(f"manifest has no split column {split_column!r}")
            table = table.loc[table[split_column].astype(str) == str(split)]
            if table.empty:
                raise ValueError(f"split {split!r} is empty in column {split_column!r}")
        columns = prior_columns(table)
        if prior_normalizer is not None:
            _check_column_match(prior_normalizer, columns)
        self.table = table
        self.row_indices = np.asarray(table.index, dtype=np.int64)
        self.prior_columns = tuple(columns)
        self.prior_normalizer = prior_normalizer
        self.normalize_images = normalize_images
        self.frames_per_window = int(frames_per_window)
        self.face_size = int(face_size)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, position: int) -> WindowSample:
        row = self.table.iloc[position]
        path = _resolve_npz_path(row)
        with np.load(path, allow_pickle=False) as data:
            frames = np.asarray(data["frames"])
            valid_mask = np.asarray(data["valid_mask"])
        expected_shape = (
            self.frames_per_window,
            self.face_size,
            self.face_size,
            3,
        )
        if frames.shape != expected_shape:
            raise ValueError(f"{path}: expected frames {expected_shape}, got {frames.shape}")
        if valid_mask.shape != (self.frame_count,):
            raise ValueError(
                f"{path}: expected valid_mask ({self.frame_count},), "
                f"got {valid_mask.shape}"
            )
        rgb = bgr_to_rgb(frames)
        inputs = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div_(255.0)
        )
        if self.normalize_images:
            inputs = (inputs - IMAGENET_MEAN) / IMAGENET_STD
        priors = row[list(self.prior_columns)].to_numpy(dtype=np.float32)
        if self.prior_normalizer is not None:
            priors = self.prior_normalizer.transform(priors)
        label = _label_index(row["label"])
        return WindowSample(
            inputs=inputs,
            valid_mask=torch.from_numpy(valid_mask.astype(bool, copy=False)),
            priors=torch.from_numpy(np.asarray(priors, dtype=np.float32)),
            label=torch.tensor(label, dtype=torch.long),
            index=torch.tensor(int(self.row_indices[position]), dtype=torch.long),
        )

    @property
    def frame_count(self) -> int:
        return self.frames_per_window


class CachedFeatureDataset(Dataset):
    """Read precomputed frozen-backbone features, masks, and raw priors."""

    def __init__(
        self,
        cache_dir: Path | str,
        split: str | None = None,
        split_column: str = "split",
        prior_normalizer: PriorNormalizer | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.meta = read_json(self.cache_dir / "meta.json")
        index = pd.read_csv(
            self.cache_dir / "index.csv", dtype={"subject_id": str}
        )
        if "subject_id" in index:
            index["subject_id"] = index["subject_id"].str.zfill(2)
        self.features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        self.valid_masks = np.load(self.cache_dir / "valid_mask.npy", mmap_mode="r")
        self.priors = np.load(self.cache_dir / "priors.npy", mmap_mode="r")
        expected_rows = len(index)
        for name, array in (
            ("features", self.features),
            ("valid_mask", self.valid_masks),
            ("priors", self.priors),
        ):
            if array.shape[0] != expected_rows:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows, index has {expected_rows}"
                )
        columns = tuple(self.meta["prior_columns"])
        if self.priors.shape[1] != len(columns):
            raise ValueError(
                f"prior cache has {self.priors.shape[1]} columns, meta says {len(columns)}"
            )
        if prior_normalizer is not None:
            _check_column_match(prior_normalizer, columns)
        table = index
        if split is not None:
            if split_column not in table.columns:
                raise ValueError(f"cache index has no split column {split_column!r}")
            table = table.loc[table[split_column].astype(str) == str(split)]
            if table.empty:
                raise ValueError(f"split {split!r} is empty in column {split_column!r}")
        self.table = table
        self.cache_rows = table["cache_row"].to_numpy(dtype=np.int64)
        self.prior_columns = columns
        self.prior_normalizer = prior_normalizer

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, position: int) -> WindowSample:
        row = self.table.iloc[position]
        cache_row = int(row["cache_row"])
        inputs = torch.from_numpy(
            np.asarray(self.features[cache_row], dtype=np.float32)
        )
        valid_mask = torch.from_numpy(
            np.array(self.valid_masks[cache_row], dtype=bool, copy=True)
        )
        priors = np.array(self.priors[cache_row], dtype=np.float32, copy=True)
        if self.prior_normalizer is not None:
            priors = self.prior_normalizer.transform(priors)
        label = _label_index(row["label"])
        manifest_row = row.get("manifest_row", cache_row)
        return WindowSample(
            inputs=inputs,
            valid_mask=valid_mask,
            priors=torch.from_numpy(np.asarray(priors, dtype=np.float32)),
            label=torch.tensor(label, dtype=torch.long),
            index=torch.tensor(int(manifest_row), dtype=torch.long),
        )

    def prior_matrix(self) -> np.ndarray:
        """Raw prior rows for fitting a normalizer on this split only."""
        return np.asarray(self.priors[self.cache_rows], dtype=np.float32)

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])


def _label_index(label: object) -> int:
    value = int(label)
    if value not in LABEL_TO_INDEX:
        raise ValueError(f"unexpected label {value}; expected one of {sorted(LABEL_TO_INDEX)}")
    return LABEL_TO_INDEX[value]
