"""Window- and video-level metrics plus subject-level bootstrap intervals.

The primary metric remains window-level Macro-F1 so the CNN rows are directly
comparable with the RF baseline.  Video-level metrics aggregate window
predictions by mean probability and by majority vote.  Bootstrap intervals
resample whole subjects, not windows, because windows from one subject are not
independent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


DEFAULT_CLASS_NAMES = ("alert", "low_vigilant", "drowsy")


def window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict:
    """Return the primary window-level metrics used by the RF baseline."""
    labels = np.arange(len(class_names))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                y_true, y_pred, labels=labels, weights="quadratic"
            )
        ),
        "per_class_recall": {
            name: float(value) for name, value in zip(class_names, recalls)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def aggregate_videos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    video_keys: np.ndarray,
    subject_ids: np.ndarray,
) -> dict:
    """Aggregate window predictions to one row per video."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    video_keys = np.asarray(video_keys).astype(str)
    subject_ids = np.asarray(subject_ids).astype(str)
    if not (
        len(y_true)
        == len(y_pred)
        == len(probabilities)
        == len(video_keys)
        == len(subject_ids)
    ):
        raise ValueError("window arrays must have the same length")
    num_classes = probabilities.shape[1]
    true_video = []
    pred_mean = []
    pred_majority = []
    subjects_video = []
    unique_videos = []
    for key in np.unique(video_keys):
        indices = np.flatnonzero(video_keys == key)
        labels = np.unique(y_true[indices])
        if len(labels) != 1:
            raise ValueError(f"video {key} contains multiple labels: {labels}")
        subjects = np.unique(subject_ids[indices])
        if len(subjects) != 1:
            raise ValueError(f"video {key} contains multiple subjects: {subjects}")
        true_video.append(int(labels[0]))
        mean_prob = probabilities[indices].mean(axis=0)
        pred_mean.append(int(np.argmax(mean_prob)))
        counts = np.bincount(y_pred[indices], minlength=num_classes)
        pred_majority.append(int(np.argmax(counts)))
        subjects_video.append(str(subjects[0]))
        unique_videos.append(str(key))
    return {
        "video_keys": np.asarray(unique_videos, dtype=str),
        "y_true": np.asarray(true_video, dtype=np.int64),
        "y_pred_mean": np.asarray(pred_mean, dtype=np.int64),
        "y_pred_majority": np.asarray(pred_majority, dtype=np.int64),
        "subjects": np.asarray(subjects_video, dtype=str),
    }


def video_metrics(
    aggregate: dict,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict:
    """Compute video-level metrics for mean-probability and majority vote."""
    return {
        "mean_prob": window_metrics(
            aggregate["y_true"], aggregate["y_pred_mean"], class_names
        ),
        "majority": window_metrics(
            aggregate["y_true"], aggregate["y_pred_majority"], class_names
        ),
        "num_videos": int(len(aggregate["y_true"])),
    }


def bootstrap_subject_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 20261105,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict:
    """Percentile CI obtained by resampling whole subjects with replacement."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subjects = np.asarray(subjects).astype(str)
    if not (len(y_true) == len(y_pred) == len(subjects)):
        raise ValueError("bootstrap arrays must have the same length")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) == 0:
        raise ValueError("no subjects to bootstrap")
    indices_by_subject = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([indices_by_subject[subject] for subject in sampled])
        values.append(window_metrics(y_true[indices], y_pred[indices], class_names)[metric])
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
        "n_boot": int(n_boot),
        "metric": metric,
    }
