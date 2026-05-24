"""
models.py
---------
Model untuk klasifikasi gesture BISINDO.

Input semua model: (B, 1, 90, 488) — 1-channel pseudo-image
Output            : (B, n_classes)  — logits mentah

Model tersedia:
    - CNN2DBaseline  : CNN sederhana sebagai baseline
    - ResNet18       : ResNet-18 modifikasi (rekomendasi utama TA)
    - ResNet34       : ResNet-34 modifikasi
    - ResNet50       : ResNet-50 modifikasi

Factory function:
    build_model(name, n_classes) -> nn.Module
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


# ===========================================================================
# 1. CNN2D Baseline (pembanding sederhana)
# ===========================================================================

class CNN2DBaseline(nn.Module):
    """CNN 2D sederhana tanpa residual — dipakai sebagai baseline pembanding.

    Input : (B, 1, 90, 488)
    Output: (B, n_classes)
    """

    def __init__(self, n_classes: int = 10, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # (B, 32, 45, 244)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # (B, 64, 22, 122)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ===========================================================================
# 2. ResNet backbone (shared builder)
# ===========================================================================

def _build_resnet_backbone(
    kind: str,
    n_classes: int,
    in_channels: int,
) -> nn.Module:
    """Modifikasi ResNet standar untuk pseudo-image 1-channel BISINDO.

    Perubahan dari default:
        conv1 : in_channels → in_channels (bukan 3)
        conv1 : stride=2 → stride=1  (input 90×488 tidak perlu aggressive downsample)
        fc    : 1000 → n_classes
    """
    if kind == "resnet18":
        model = resnet18(weights=None)
    elif kind == "resnet34":
        model = resnet34(weights=None)
    elif kind == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Tidak dikenal: {kind}. Pilihan: resnet18 | resnet34 | resnet50")

    # Override conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=1,        # stride 1 karena input kecil (90×488)
        padding=3,
        bias=False,
    )

    # Override classifier head
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


# ===========================================================================
# 3. ResNet wrappers
# ===========================================================================

class ResNet18(nn.Module):
    """ResNet-18 untuk pseudo-image 1-channel. Rekomendasi utama TA."""

    def __init__(self, n_classes: int = 10, in_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.backbone = _build_resnet_backbone("resnet18", n_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet34(nn.Module):
    """ResNet-34 untuk pseudo-image 1-channel."""

    def __init__(self, n_classes: int = 10, in_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.backbone = _build_resnet_backbone("resnet34", n_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet50(nn.Module):
    """ResNet-50 untuk pseudo-image 1-channel."""

    def __init__(self, n_classes: int = 10, in_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.backbone = _build_resnet_backbone("resnet50", n_classes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ===========================================================================
# 4. Factory function
# ===========================================================================

_MODEL_REGISTRY = {
    "cnn2d":    CNN2DBaseline,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
}


def build_model(name: str, n_classes: int = 10, in_channels: int = 1, **kwargs: Any) -> nn.Module:
    """Factory function — buat model dari nama string (sesuai train.yaml).

    Parameters
    ----------
    name        : "cnn2d" | "resnet18" | "resnet34" | "resnet50"
    n_classes   : jumlah kelas output
    in_channels : jumlah channel input (1 untuk pseudo-image BISINDO)

    Returns
    -------
    nn.Module siap training
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' tidak dikenal. "
            f"Pilihan: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](n_classes=n_classes, in_channels=in_channels, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Sanity check
# ===========================================================================

if __name__ == "__main__":
    dummy = torch.randn(4, 1, 90, 488)   # (batch, 1-channel, 90 frame, 488 fitur)

    for name in ["cnn2d", "resnet18", "resnet34", "resnet50"]:
        model = build_model(name, n_classes=10)
        out   = model(dummy)
        params = count_parameters(model)
        print(f"{name:10s} | output: {tuple(out.shape)} | params: {params:,}")