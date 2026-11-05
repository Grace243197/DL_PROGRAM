from __future__ import annotations

import argparse
import random
from pathlib import Path

from .common import ROOT, load_config, read_json, write_json_atomic


def build_split(config: dict) -> dict:
    settings = config["split"]
    subjects = [f"{index:02d}" for index in range(1, config["dataset"]["expected_subjects"] + 1)]
    assert sum(settings[name] for name in ("train", "val", "test")) == len(subjects)
    random.Random(settings["seed"]).shuffle(subjects)
    train_end = settings["train"]
    val_end = train_end + settings["val"]
    return {
        "seed": settings["seed"],
        "dataset": f"{config['dataset']['owner']}/{config['dataset']['slug']}",
        "train": sorted(subjects[:train_end]),
        "val": sorted(subjects[train_end:val_end]),
        "test": sorted(subjects[val_end:]),
    }


def freeze_split(config_path: Path | None = None, output: Path | None = None) -> dict:
    expected = build_split(load_config(config_path))
    target = output or ROOT / "splits" / "subject_split.json"
    if target.exists():
        actual = read_json(target)
        if actual != expected:
            raise RuntimeError(f"Existing frozen split differs from configuration: {target}")
        return actual
    write_json_atomic(target, expected)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the subject-disjoint 42/9/9 split")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    split = freeze_split(args.config)
    print({name: len(split[name]) for name in ("train", "val", "test")})
    print(f"seed={split['seed']}")


if __name__ == "__main__":
    main()
