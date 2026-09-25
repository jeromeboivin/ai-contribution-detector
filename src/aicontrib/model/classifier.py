from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        # Per-feature standardization, fitted on the training set (fit_input_scaling) and saved with
        # the weights. Identity until fitted. Hidden-state features vary a lot in scale across dimensions.
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def fit_input_scaling(self, x: torch.Tensor) -> None:
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.input_mean) / self.input_std)
