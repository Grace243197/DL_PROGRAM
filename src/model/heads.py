"""Temporal and prior-fusion heads for the 2x2 factorial design.

The four model cells share the same frame features:

    frame-level pooling + no prior      -> FrameClassifier(use_prior=False)
    frame-level pooling + FiLM prior    -> FrameClassifier(use_prior=True)
    causal GRU + no prior               -> GRUClassifier(use_prior=False)
    causal GRU + FiLM prior             -> GRUClassifier(use_prior=True)

``fusion="concat"`` is the ablation baseline for FiLM.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    gate: torch.Tensor | None = None


class FiLMModulation(nn.Module):
    """Produce a sigmoid gate from the prior vector and modulate features."""

    def __init__(self, prior_dim: int, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, priors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.mlp(priors))
        return gate * features, gate


def _check_window_inputs(frame_features: torch.Tensor, valid_mask: torch.Tensor) -> None:
    if frame_features.ndim != 3:
        raise ValueError(f"expected frame features (B, T, D), got {tuple(frame_features.shape)}")
    if valid_mask.shape != frame_features.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} must match (B, T) of frame features"
        )


def masked_mean(frame_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Average only over frames with a detected face."""

    _check_window_inputs(frame_features, valid_mask)
    weights = valid_mask.unsqueeze(-1).to(frame_features.dtype)
    total = weights.sum(dim=1).clamp_min(1.0)
    return (frame_features * weights).sum(dim=1) / total


def compact_valid_sequences(
    frame_features: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove invalid frames so a packed GRU never sees padding or holes."""

    _check_window_inputs(frame_features, valid_mask)
    lengths = valid_mask.sum(dim=1).long()
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    if max_length == 0:
        raise ValueError("every window in the batch has an empty valid_mask")
    compact = frame_features.new_zeros((frame_features.size(0), max_length, frame_features.size(2)))
    for index in range(frame_features.size(0)):
        selected = frame_features[index][valid_mask[index]]
        compact[index, : selected.size(0)] = selected
    return compact, lengths.clamp_min(1)


def _mlp_head(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class FrameClassifier(nn.Module):
    """Frame-level average pooling, optionally modulated by priors."""

    def __init__(
        self,
        feature_dim: int = 576,
        prior_dim: int = 34,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
        use_prior: bool = False,
        fusion: str = "film",
    ) -> None:
        super().__init__()
        self.use_prior = use_prior
        self.fusion = fusion
        self.film: FiLMModulation | None = None
        if use_prior and fusion == "film":
            self.film = FiLMModulation(prior_dim, feature_dim)
        if not use_prior:
            head_in = feature_dim
        elif fusion == "film":
            head_in = feature_dim * 2 + prior_dim
        elif fusion == "concat":
            head_in = feature_dim + prior_dim
        else:
            raise ValueError(f"unknown fusion mode: {fusion}")
        self.head = _mlp_head(head_in, hidden_dim, num_classes, dropout)

    def forward(
        self,
        frame_features: torch.Tensor,
        valid_mask: torch.Tensor,
        priors: torch.Tensor | None = None,
    ) -> ModelOutput:
        pooled = masked_mean(frame_features, valid_mask)
        gate = None
        if not self.use_prior:
            representation = pooled
        else:
            if priors is None:
                raise ValueError("priors are required when use_prior=True")
            if self.fusion == "film":
                assert self.film is not None
                modulated, gate = self.film(pooled, priors)
                representation = torch.cat([pooled, priors, modulated], dim=-1)
            else:
                representation = torch.cat([pooled, priors], dim=-1)
        return ModelOutput(self.head(representation), gate)


class GRUClassifier(nn.Module):
    """Causal single-layer GRU, optionally modulated by priors."""

    def __init__(
        self,
        feature_dim: int = 576,
        prior_dim: int = 34,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
        use_prior: bool = False,
        fusion: str = "film",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.use_prior = use_prior
        self.fusion = fusion
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.film: FiLMModulation | None = None
        if use_prior and fusion == "film":
            self.film = FiLMModulation(prior_dim, hidden_dim)
        if not use_prior:
            head_in = hidden_dim
        elif fusion == "film":
            head_in = hidden_dim * 2 + prior_dim
        elif fusion == "concat":
            head_in = hidden_dim + prior_dim
        else:
            raise ValueError(f"unknown fusion mode: {fusion}")
        self.head = _mlp_head(head_in, hidden_dim, num_classes, dropout)

    def forward(
        self,
        frame_features: torch.Tensor,
        valid_mask: torch.Tensor,
        priors: torch.Tensor | None = None,
    ) -> ModelOutput:
        compact, lengths = compact_valid_sequences(frame_features, valid_mask)
        packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        representation = hidden[-1]
        gate = None
        if not self.use_prior:
            fused = representation
        else:
            if priors is None:
                raise ValueError("priors are required when use_prior=True")
            if self.fusion == "film":
                assert self.film is not None
                modulated, gate = self.film(representation, priors)
                fused = torch.cat([representation, priors, modulated], dim=-1)
            else:
                fused = torch.cat([representation, priors], dim=-1)
        return ModelOutput(self.head(fused), gate)
