from __future__ import annotations

import argparse
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
}
LABELS = [0, 5, 10]


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


def evaluate(allow_partial: bool = False) -> dict:
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    split_path = ROOT / "splits" / "subject_split.json"
    split = read_json(split_path)
    manifest = read_json(ROOT / "splits" / "source_manifest.json")
    status_dir = ROOT / "data" / "status"
    complete = sum((status_dir / f"{entry['key']}.json").exists() for entry in manifest["files"])
    if complete != len(manifest["files"]) and not allow_partial:
        raise RuntimeError(f"Only {complete}/{len(manifest['files'])} videos processed; final RF blocked")
    table = pd.read_csv(ROOT / "features" / "windows.csv")
    if table.empty or table[["split", "subject_id", "label"]].isna().any().any():
        raise RuntimeError("Incomplete feature table")
    table["subject_id"] = table["subject_id"].astype(str).str.zfill(2)
    for name in ("train", "val", "test"):
        observed = set(table.loc[table["split"] == name, "subject_id"])
        if not observed or not observed.issubset(set(split[name])):
            raise RuntimeError(f"Subject leakage or empty split in {name}: {observed}")
        labels = set(table.loc[table["split"] == name, "label"])
        if labels != set(LABELS):
            raise RuntimeError(f"Split {name} lacks classes: {set(LABELS) - labels}")
    feature_names = [column for column in table if column not in METADATA_COLUMNS]
    if not feature_names or not all(pd.api.types.is_numeric_dtype(table[column]) for column in feature_names):
        raise RuntimeError("Unexpected nonnumeric feature column")
    subsets = {name: table.loc[table["split"] == name] for name in ("train", "val", "test")}
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
        "split_sha256": split_hash,
        "config_sha256": sha256_file(config_path),
        "feature_names": feature_names,
        "window_counts": {name: len(subsets[name]) for name in subsets},
        "detection_failure_rate_by_split": {
            name: float(subsets[name]["detection_failure_rate"].mean()) for name in subsets
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
    target = ROOT / "results" / f"rf_prior_{output['status']}_{timestamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    joblib.dump({"model": best_model, "features": feature_names,
                 "split_sha256": split_hash}, target.with_suffix(".joblib"))
    log = ROOT / "experiments" / "rf_prior_runs.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": timestamp, "status": output["status"],
           "config": str(config_path.relative_to(ROOT)), "split_sha256": split_hash,
           "seeds": ",".join(map(str, config["rf"]["seeds"])),
           "val_macro_f1_mean": output["macro_f1_mean"]["val"],
           "test_macro_f1_mean": output["macro_f1_mean"]["test"],
           "result": str(target.relative_to(ROOT))}
    with log.open("a", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=row)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return {"result": str(target), "summary": output["macro_f1_mean"],
            "status": output["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF-Prior on frozen subject split")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Label output as a pilot; never use it as the final 42/9/9 result")
    args = parser.parse_args()
    print(evaluate(args.allow_partial))


if __name__ == "__main__":
    main()
