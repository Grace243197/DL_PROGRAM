from __future__ import annotations

import torch

from src.model.backbone import MobileNetV3FeatureExtractor
from src.model.factory import build_backbone, build_model
from src.model.heads import FrameClassifier, GRUClassifier


def _window_batch(batch_size: int = 2, frames: int = 16, feature_dim: int = 576):
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(batch_size, frames, feature_dim, generator=generator)
    mask = torch.ones(batch_size, frames, dtype=torch.bool)
    priors = torch.randn(batch_size, 34, generator=generator)
    return features, mask, priors


def test_frame_classifier_without_prior() -> None:
    features, mask, _ = _window_batch()
    model = FrameClassifier(use_prior=False)
    output = model(features, mask)
    assert output.logits.shape == (2, 3)
    assert output.gate is None
    assert torch.isfinite(output.logits).all()


def test_frame_classifier_ignores_masked_frames() -> None:
    features, mask, _ = _window_batch()
    model = FrameClassifier(use_prior=False).eval()
    mask[0, 8:] = False
    masked = features.clone()
    masked[0, 8:] = 1e6  # would explode the mean if the mask were ignored
    with torch.no_grad():
        output = model(masked, mask)
    assert torch.isfinite(output.logits).all()


def test_frame_classifier_film_gate_and_logits() -> None:
    features, mask, priors = _window_batch()
    model = FrameClassifier(use_prior=True, fusion="film")
    output = model(features, mask, priors)
    assert output.logits.shape == (2, 3)
    assert output.gate is not None
    assert output.gate.shape == (2, 576)
    assert torch.all(output.gate >= 0) and torch.all(output.gate <= 1)


def test_frame_classifier_concat_ablation() -> None:
    features, mask, priors = _window_batch()
    model = FrameClassifier(use_prior=True, fusion="concat")
    output = model(features, mask, priors)
    assert output.logits.shape == (2, 3)
    assert output.gate is None


def test_gru_classifier_handles_missing_frames() -> None:
    features, mask, priors = _window_batch()
    mask[0, [0, 3, 7, 11]] = False
    mask[1, 13:] = False
    model = GRUClassifier(use_prior=True, fusion="film")
    output = model(features, mask, priors)
    assert output.logits.shape == (2, 3)
    assert output.gate is not None
    assert output.gate.shape == (2, 128)
    assert torch.isfinite(output.logits).all()


def test_mobilenet_backbone_is_frozen_and_returns_576_features() -> None:
    model = MobileNetV3FeatureExtractor(pretrained=False, freeze=True)
    model.eval()
    images = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        features = model(images)
    assert features.shape == (2, 576)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())

    model.train()
    assert model.training is True
    assert model.features.training is False  # BN stays in eval mode


def test_factory_builds_both_cells_from_config() -> None:
    config = {
        "model": {
            "feature_dim": 576,
            "prior_dim": 34,
            "hidden_dim": 128,
            "num_classes": 3,
            "dropout": 0.2,
            "fusion": "film",
            "pretrained": False,
            "freeze_backbone": True,
            "gru_num_layers": 1,
        }
    }
    frame = build_model(config, temporal=False, use_prior=False)
    gru = build_model(config, temporal=True, use_prior=True)
    backbone = build_backbone(config)
    assert isinstance(frame, FrameClassifier)
    assert isinstance(gru, GRUClassifier)
    assert isinstance(backbone, MobileNetV3FeatureExtractor)
    assert backbone.freeze is True
