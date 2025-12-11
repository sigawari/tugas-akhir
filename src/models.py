# models.py
# ResNet-18 2D untuk sign recognition berbasis landmark (x, y, dx, dy).
# Input ke model: (B, 4, T, L)

from typing import Any
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet2DSign(nn.Module):
    """ResNet-18 2D untuk input (B, 4, T, L) tanpa pretrained, tanpa TSM."""

    def __init__(self, num_classes: int, in_channels: int = 4, **kwargs: Any) -> None:
        super().__init__()

        # ResNet18 tanpa pretrained
        backbone = resnet18(weights=None)  # pure random init

        # DEBUG: cetak conv1 awal
        print("[DEBUG] Sebelum override conv1:", backbone.conv1)

        # Ubah conv1 supaya terima 4 channel (x, y, dx, dy)
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # DEBUG: cek conv1 sesudah diubah
        print("[DEBUG] Sesudah override conv1:", backbone.conv1)

        # Ganti fc terakhir ke num_classes
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Linear(in_feats, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, T, L)
        """
        return self.backbone(x)
