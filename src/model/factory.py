"""Build one of the four model cells from the frozen YAML config."""

from __future__ import annotations

from .backbone import MobileNetV3FeatureExtractor
from .heads import FrameClassifier, GRUClassifier


def build_model(config: dict, temporal: bool = False, use_prior: bool = False):
    settings = config["model"]
    common = {
        "feature_dim": settings["feature_dim"],
        "prior_dim": settings["prior_dim"],
        "hidden_dim": settings["hidden_dim"],
        "num_classes": settings["num_classes"],
        "dropout": settings["dropout"],
        "use_prior": use_prior,
        "fusion": settings["fusion"],
    }
    if temporal:
        return GRUClassifier(
            **common,
            num_layers=settings.get("gru_num_layers", 1),
        )
    return FrameClassifier(**common)


def build_backbone(config: dict) -> MobileNetV3FeatureExtractor:
    settings = config["model"]
    return MobileNetV3FeatureExtractor(
        pretrained=settings["pretrained"],
        freeze=settings["freeze_backbone"],
    )
