# models.py
# Model spasial 2D untuk sign recognition berbasis landmark (x, y, dx, dy).
# Input ke model: (B, 4, T, L) -> diperlakukan sebagai "image" C=4, H=T, W=L

from __future__ import annotations

from typing import Any, Optional, List
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


class Basic2DCNN(nn.Module):
    """
    Baseline CNN 2D sederhana untuk input (B, 4, T, L).
    Tujuan: jadi pembanding "CNN biasa" vs ResNet.
    """
    def __init__(self, num_classes: int, in_channels: int = 4, dropout: float = 0.2, **kwargs: Any) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # (B, 4, T, L)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (T/2, L/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (T/4, L/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                # (B, 128)
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z)


def _make_resnet_backbone(kind: str, num_classes: int, in_channels: int) -> nn.Module:
    if kind == "resnet18":
        backbone = resnet18(weights=None)
    elif kind == "resnet34":
        backbone = resnet34(weights=None)
    elif kind == "resnet50":
        backbone = resnet50(weights=None)
    else:
        raise ValueError(f"Unknown resnet kind: {kind}")

    # Override conv1 supaya bisa nerima 4 channel (x,y,dx,dy)
    backbone.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    # Override head classifier
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Linear(in_feats, num_classes)
    return backbone


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 4, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = _make_resnet_backbone("resnet18", num_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 4, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = _make_resnet_backbone("resnet34", num_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 4, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = _make_resnet_backbone("resnet50", num_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CNN2DResidual(nn.Module):
    """
    Residual CNN 2D untuk input (B, 4, T, L).
    Tujuan: jadi pembanding "CNN residual" vs ResNet.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 4,
        layers: Optional[List[int]] = None,
        base_channels: int = 64,
        dropout: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # (B, 4, T, L)
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (T/2, L/2)

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (T/4, L/4)

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # (B, base_channels * 4, 1, 1)
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.head(z)
