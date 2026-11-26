from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.pipeline import make_pipeline

from .common import ROOT, load_config, read_json, sha256_file


METADATA_COLUMNS = {
    "source_file", "video_key", "subject_id", "split", "label", "class_name",
    "window_index", "start_s", "end_s", "npz_path",
    "within_split", "subject_split", "cache_row", "manifest_row",
}
LABELS = [0, 5, 10]
SPLITS = ("train", "val", "test")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
        "per_class_recall": dict(zip(map(str, LABELS),
                                     map(float, recall_score(y_true, y_pred, labels=LABELS,
                                                            average=None, zero_division=0)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _merge_split_file(table: pd.DataFrame, split_column: str, split_path: Path) -> pd.DataFrame:
    """Attach an external window-level split column, requiring exact coverage."""
    assignment = pd.read_csv(split_path, dtype={"subject_id": str})
    missing = {"npz_path", split_column} - set(assignment.columns)
    if missing:
        raise ValueError(f"split file {split_path} is missing columns: {sorted(missing)}")
    if split_column in table.columns:
        raise ValueError(
            f"split column {split_column!r} already exists in features/windows.csv; "
            "use a new column name such as 'within_split'"
        )
    if assignment["npz_path"].duplicated().any():
        raise ValueError(f"split file {split_path} contains duplicate npz_path values")
    merged = table.merge(
        assignment[["npz_path", split_column]],
        on="npz_path",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(table):
        raise RuntimeError("split file merge changed the number of windows")
    if merged[split_column].isna().any():
        missing_rows = merged.loc[merged[split_column].isna(), "npz_path"].head(5).tolist()
        raise RuntimeError(f"split file does not cover all windows, first missing: {missing_rows}")
    if set(assignment["npz_path"]) != set(table["npz_path"]):
        extra = len(set(assignment["npz_path"]) - set(table["npz_path"]))
        raise RuntimeError(f"split file has {extra} windows not present in features/windows.csv")
    return merged


def _validate_splits(
    table: pd.DataFrame,
    split_column: str,
    subject_split: dict,
    allow_subject_overlap: bool,
) -> None:
    observed = set(table[split_column].astype(str))
    if observed != set(SPLITS):
        raise RuntimeError(f"Unexpected split values in {split_column!r}: {sorted(observed)}")
    for name in SPLITS:
        subset = table.loc[table[split_column].astype(str) == name]
        if subset.empty:
            raise RuntimeError(f"Split {name!r} is empty in column {split_column!r}")
        labels = set(subset["label"].astype(int))
        if labels != set(LABELS):
            raise RuntimeError(f"Split {name!r} lacks classes: {set(LABELS) - labels}")
        if not allow_subject_overlap:
            subjects = set(subset["subject_id"].astype(str))
            if not subjects.issubset(set(subject_split[name])):
                raise RuntimeError(
                    f"Subject leakage or wrong split in {name!r}: {sorted(subjects)}"
                )


def _append_run_log(log_path: Path, row: dict) -> None:
    """Append one run, extending the header if new columns were introduced."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fields: list[str] = []
    if log_path.exists() and log_path.stat().st_size > 0:
        with log_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in fields:
            fields.append(key)
    rows.append({key: row.get(key, "") for key in fields})
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    allow_partial: bool = False,
    split_column: str = "split",
    split_file: Path | None = None,
    allow_subject_overlap: bool = False,
    tag: str = "",
) -> dict:
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    subject_split_path = ROOT / "splits" / "subject_split.json"
    subject_split = read_json(subject_split_path)
    if split_file is not None:
        split_path = _resolve(split_file)
    else:
        split_path = subject_split_path
    manifest = read_json(ROOT / "splits" / "source_manifest.json")
    status_dir = ROOT / "data" / "status"
    completed_keys = set()
    for entry in manifest["files"]:
        marker = status_dir / f"{entry['key']}.json"
        if marker.exists() and not read_json(marker).get("source_retained_for_review", False):
            completed_keys.add(entry["key"])
    complete = len(completed_keys)
    if complete != len(manifest["files"]) and not allow_partial:
        raise RuntimeError(f"Only {complete}/{len(manifest['files'])} videos processed; final RF blocked")
    table = pd.read_csv(ROOT / "features" / "windows.csv", dtype={"subject_id": str})
    if table.empty or table[["split", "subject_id", "label"]].isna().any().any():
        raise RuntimeError("Incomplete feature table")
    table["subject_id"] = table["subject_id"].astype(str).str.zfill(2)
    if set(table["video_key"].astype(str)) != completed_keys:
        raise RuntimeError("Feature table does not match completed video status markers")
    if split_file is not None:
        table = _merge_split_file(table, split_column, split_path)
    _validate_splits(table, split_column, subject_split, allow_subject_overlap)
    sampled = config["preprocessing"]["frames_per_window"]
    minimum = config["preprocessing"]["min_valid_frames_per_window"]
    if (table["detection_failure_rate"] > (sampled - minimum) / sampled + 1e-9).any():
        raise RuntimeError("Feature table contains windows failing the frozen QC rule")
    qc_audit = pd.read_csv(ROOT / "features" / "windows_qc_excluded.csv")
    feature_names = [column for column in table if column not in METADATA_COLUMNS]
    if not feature_names or not all(pd.api.types.is_numeric_dtype(table[column]) for column in feature_names):
        raise RuntimeError("Unexpected nonnumeric feature column")
    subsets = {
        name: table.loc[table[split_column].astype(str) == name] for name in SPLITS
    }
    train = subsets["train"]
    results = []
    best_val = float("-inf")
    best_model = None
    for seed in config["rf"]["seeds"]:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=config["rf"]["n_estimators"],
                min_samples_leaf=config["rf"]["min_samples_leaf"],
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
            ),
        )
        model.fit(train[feature_names], train["label"])
        item = {"seed": seed}
        for name in ("val", "test"):
            subset = subsets[name]
            item[name] = _metrics(subset["label"].to_numpy(), model.predict(subset[feature_names]))
        results.append(item)
        if item["val"]["macro_f1"] > best_val:
            best_val = item["val"]["macro_f1"]
            best_model = model
    split_hash = sha256_file(split_path)
    output = {
        "status": "pilot_partial" if complete != len(manifest["files"]) else "full",
        "processed_videos": complete,
        "total_videos": len(manifest["files"]),
        "split_column": split_column,
        "split_file": str(split_path.relative_to(ROOT)) if split_path.is_relative_to(ROOT) else str(split_path),
        "split_sha256": split_hash,
        "subject_overlap_allowed": allow_subject_overlap,
        "config_sha256": sha256_file(config_path),
        "feature_names": feature_names,
        "window_counts": {name: len(subsets[name]) for name in SPLITS},
        "window_qc": {"min_valid_frames_per_window": minimum,
                      "excluded_windows": len(qc_audit)},
        "detection_failure_rate_by_split": {
            name: float(subsets[name]["detection_failure_rate"].mean()) for name in SPLITS
        },
        "seeds": results,
        "macro_f1_mean": {name: float(np.mean([item[name]["macro_f1"] for item in results]))
                          for name in ("val", "test")},
        "macro_f1_std": {name: float(np.std([item[name]["macro_f1"] for item in results], ddof=1))
                         for name in ("val", "test")},
    }
    output["test_detection_failure_over_10_percent"] = (
        output["detection_failure_rate_by_split"]["test"] > 0.10
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = ["rf_prior", output["status"]]
    if tag:
        parts.append(tag)
    target = ROOT / "results" / ("_".join(parts + [timestamp]) + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    joblib.dump({"model": best_model, "features": feature_names,
                 "split_sha256": split_hash}, target.with_suffix(".joblib"))
    log = ROOT / "experiments" / "rf_prior_runs.csv"
    row = {
        "timestamp_utc": timestamp,
        "status": output["status"],
        "tag": tag,
        "split_column": split_column,
        "split_file": output["split_file"],
        "split_sha256": split_hash,
        "subject_overlap_allowed": allow_subject_overlap,
        "config": str(config_path.relative_to(ROOT)),
        "seeds": ",".join(map(str, config["rf"]["seeds"])),
        "val_macro_f1_mean": output["macro_f1_mean"]["val"],
        "test_macro_f1_mean": output["macro_f1_mean"]["test"],
        "test_qwk_seed_best": results[
            int(np.argmax([item["val"]["macro_f1"] for item in results]))
        ]["test"]["quadratic_weighted_kappa"],
        "result": str(target.relative_to(ROOT)),
    }
    _append_run_log(log, row)
    return {"result": str(target), "summary": output["macro_f1_mean"],
            "status": output["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF-Prior on a frozen window split")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Label output as a pilot; never use it as the final 42/9/9 result")
    parser.add_argument("--split-column", default="split",
                        help="Column in features/windows.csv or in --split-file (default: split)")
    parser.add_argument("--split-file", type=Path,
                        help="Optional CSV with npz_path and the requested split column")
    parser.add_argument("--allow-subject-overlap", action="store_true",
                        help="Skip the subject-disjoint leakage check (required for within-subject splits)")
    parser.add_argument("--tag", default="",
                        help="Optional tag inserted into the result filename and run log")
    args = parser.parse_args()
    if (args.split_column != "split" or args.split_file is not None) and not args.allow_subject_overlap:
        parser.error(
            "a custom split shares subjects across sets by construction; "
            "pass --allow-subject-overlap to confirm this is intentional"
        )
    print(evaluate(
        allow_partial=args.allow_partial,
        split_column=args.split_column,
        split_file=args.split_file,
        allow_subject_overlap=args.allow_subject_overlap,
        tag=args.tag,
    ))


if __name__ == "__main__":
    main()
