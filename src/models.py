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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Shortcut connection (Skip Connection)
        # Jika dimensi berubah (stride > 1), sesuaikan dimensi inputnya agar bisa dijumlahkan
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output = Fungsi(x) + x
        out = self.conv_path(x)
        out += self.shortcut(x) 
        return torch.relu(out)

class CNN2DResidual(nn.Module):
    """
    Residual CNN 2D hasil modifikasi untuk input (B, 4, T, L).
    """
    def __init__(self, num_classes: int, in_channels: int = 4, dropout: float = 0.3, **kwargs: Any):
        super().__init__()
        
        # Initial Layer (Conv 1 Modified)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Blocks (Blok dengan Skip Connection)
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2) # Downsample T dan L
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   # Conv 1 Modified
        x = self.layer1(x)  # Residual Block 1
        x = self.layer2(x)  # Residual Block 2
        x = self.layer3(x)  # Residual Block 3
        x = self.avgpool(x) # Average Pooling
        return self.head(x) # Classification Output