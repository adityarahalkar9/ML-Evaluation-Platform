from __future__ import annotations

import torch.nn as nn


class MLPClassifier(nn.Module):
    """Configurable tabular MLP: Linear -> (BatchNorm -> ReLU -> Dropout)* -> Linear."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_layers: list[int], dropout: float, use_batchnorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(model_config: dict, input_dim: int, num_classes: int) -> MLPClassifier:
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=model_config.get("hidden_layers", [64, 32]),
        dropout=model_config.get("dropout", 0.1),
        use_batchnorm=model_config.get("use_batchnorm", True),
    )