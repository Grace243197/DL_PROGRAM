from __future__ import annotations

import cv2
import numpy as np
import shutil

from src.data import kaggle_pipeline as pipeline
from src.data.common import load_config
from src.data.facial_features import FrameResult, summarize_priors
from src.data.split import build_split


def test_frozen_split_is_disjoint_and_deterministic():
    config = load_config()
    first = build_split(config)
    assert first == build_split(config)
    assert [len(first[name]) for name in ("train", "val", "test")] == [42, 9, 9]
    assert len(set(first["train"] + first["val"] + first["test"])) == 60


def test_manifest_accepts_split_video_and_m4v():
    for name, expected in (("Fold/32/10_1.mp4", "10"), ("Fold/46/5.m4v", "5"),
                           ("Fold/01/0.MOV", "0")):
        entry = pipeline.parse_entry({"name": name, "totalBytes": 123})
        assert entry is not None and entry["label"] == expected
    assert pipeline.parse_entry({"name": "Fold/01/readme.txt"}) is None


def test_window_summary_handles_missing_landmarks():
    settings = load_config()["preprocessing"]
    priors = np.full((16, 5), np.nan, dtype=np.float32)
    summary = summarize_priors(priors, np.zeros(16, dtype=bool), settings)
    assert summary["detection_failure_rate"] == 1.0
    assert np.isnan(summary["ecr_sampled"])


def test_video_processing_writes_two_16_frame_windows(tmp_path, monkeypatch):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 16, (64, 64))
    assert writer.isOpened()
    for index in range(160):
        writer.write(np.full((64, 64, 3), index % 255, dtype=np.uint8))
    writer.release()

    class FakeExtractor:
        def __init__(self, settings):
            self.size = settings["face_size"]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract(self, frame):
            crop = cv2.resize(frame, (self.size, self.size))
            return FrameResult(crop, np.array([0.3, 0.2, 0, 0, 0], dtype=np.float32), True)

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "FaceExtractor", FakeExtractor)
    config = load_config()
    entry = pipeline.parse_entry({"name": "Fold/01/0.mp4", "totalBytes": 123})
    split = {"train": ["01"], "val": ["02"], "test": ["03"]}
    summary = pipeline.process_video(video, entry, split, config)
    assert summary["window_count"] == 2
    assert summary["sampled_frames"] == 32
    assert summary["detection_failure_rate"] == 0
    first_window = tmp_path / summary["output_dir"] / "w000000.npz"
    with np.load(first_window) as bundle:
        assert bundle["frames"].shape == (16, 160, 160, 3)
        assert bundle["valid_mask"].all()


def test_opencv_reads_video_beneath_unicode_directory(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 16, (64, 64))
    assert writer.isOpened()
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    destination_dir = tmp_path / "中文目录"
    destination_dir.mkdir()
    destination = destination_dir / source.name
    shutil.move(source, destination)
    reader = cv2.VideoCapture(str(destination))
    try:
        assert reader.isOpened()
        ok, frame = reader.read()
        assert ok and frame is not None
    finally:
        reader.release()

