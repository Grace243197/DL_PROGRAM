from __future__ import annotations

import pandas as pd
import pytest

from src.data.within_split import assign_within_split, build_within_split


def _synthetic_manifest(subjects: int = 4, per_group: int = 20) -> pd.DataFrame:
    rows = []
    for subject in range(1, subjects + 1):
        subject_id = f"{subject:02d}"
        for label, class_name in ((0, "alert"), (5, "low_vigilant"), (10, "drowsy")):
            source = f"Fold1/{subject_id}/{label}.mov"
            video_key = f"{subject_id}_{label}"
            for window in range(per_group):
                rows.append(
                    {
                        "npz_path": f"data/windows/{subject_id}/{label}/{video_key}/w{window:06d}.npz",
                        "subject_id": subject_id,
                        "source_file": source,
                        "video_key": video_key,
                        "label": label,
                        "class_name": class_name,
                        "window_index": window,
                        "split": "train" if subject <= 2 else ("val" if subject == 3 else "test"),
                    }
                )
    return pd.DataFrame(rows)


def test_assignment_is_deterministic_and_covers_every_window():
    table = _synthetic_manifest()
    first = assign_within_split(table, seed=20261105)
    second = assign_within_split(table, seed=20261105)
    assert first.equals(second)
    assert len(first) == len(table)
    assert not first.isna().any()
    assert set(first.unique()) == {"train", "val", "test"}


def test_per_group_counts_are_70_15_15():
    table = _synthetic_manifest(per_group=20)
    assignment = assign_within_split(table, seed=20261105)
    grouped = table.assign(within_split=assignment).groupby(["subject_id", "label"])
    for _, group in grouped:
        counts = group["within_split"].value_counts()
        assert counts.to_dict() == {"train": 14, "val": 3, "test": 3}


def test_same_subject_appears_in_every_split():
    table = _synthetic_manifest()
    assignment = assign_within_split(table, seed=20261105)
    table = table.assign(within_split=assignment)
    for name in ("train", "val", "test"):
        assert set(table.loc[table["within_split"] == name, "subject_id"]) == {
            "01", "02", "03", "04"
        }


def test_different_seed_changes_assignment():
    table = _synthetic_manifest()
    first = assign_within_split(table, seed=20261105)
    second = assign_within_split(table, seed=20261106)
    assert not first.equals(second)


def test_tiny_group_and_duplicate_paths_are_rejected():
    table = _synthetic_manifest(subjects=1, per_group=2)
    with pytest.raises(ValueError, match="at least 3"):
        assign_within_split(table, seed=1)
    duplicate = _synthetic_manifest(subjects=1, per_group=4)
    duplicate.loc[1, "npz_path"] = duplicate.loc[0, "npz_path"]
    with pytest.raises(ValueError, match="duplicate npz_path"):
        assign_within_split(duplicate, seed=1)


def test_build_preserves_original_split_and_refuses_to_change(tmp_path):
    table = _synthetic_manifest()
    manifest = tmp_path / "windows.csv"
    output = tmp_path / "within_subject_split.csv"
    meta = tmp_path / "within_subject_split.meta.json"
    table.to_csv(manifest, index=False)

    result = build_within_split(manifest_path=manifest, output_path=output, meta_path=meta)
    frozen = pd.read_csv(output, dtype={"subject_id": str})
    assert (frozen["subject_split"] == table["split"].to_numpy()).all()
    assert result["output_sha256"]
    assert meta.exists()

    # Re-running with the same protocol is accepted and leaves the file alone.
    build_within_split(manifest_path=manifest, output_path=output, meta_path=meta)

    # A different seed must not silently overwrite the frozen file.
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        build_within_split(
            manifest_path=manifest, output_path=output, meta_path=meta, seed=7
        )
