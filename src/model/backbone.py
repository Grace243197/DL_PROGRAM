"""Frozen MobileNetV3-Small frame encoder.

The backbone is intentionally kept separate from the temporal/prior heads so
the same cached frame features can be reused by every variant in the 2x2
factorial design.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3FeatureExtractor(nn.Module):
    """Map an RGB image batch to 576-dimensional ImageNet features."""

    feature_dim = 576

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freeze = freeze
        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        self._sync_feature_mode()

    def _sync_feature_mode(self) -> None:
        # BatchNorm statistics must stay frozen together with the weights.
        self.features.train(not self.freeze)

    def train(self, mode: bool = True) -> "MobileNetV3FeatureExtractor":
        super().train(mode)
        self._sync_feature_mode()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
        features = self.features(images)
        return self.pool(features).flatten(1)
