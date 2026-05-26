"""
dataset.py
----------
PyTorch Dataset untuk klasifikasi gesture BISINDO.

Alur per sample (__getitem__):
    (90, 122, 2)
        → Augmentasi on-the-fly (train only)
        → Hitung delta: concat [x, y, dx, dy] → (90, 122, 4)
        → Flatten landmark: reshape(90, 488)
        → Unsqueeze channel: (1, 90, 488)   <- pseudo-image siap ResNet
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# Konstanta landmark slice
POSE_SLICE       = slice(0, 12)
LEFT_HAND_SLICE  = slice(12, 33)
RIGHT_HAND_SLICE = slice(33, 54)
FACE_SLICE       = slice(54, 122)
EPS = 1e-12


class BISINDODataset(Dataset):
    """Dataset gesture BISINDO dari array numpy.

    Parameters
    ----------
    X       : np.ndarray  (N, T, L, 2)   koordinat (x, y) per frame
    y       : np.ndarray  (N,)            label integer
    augment : bool        aktifkan augmentasi on-the-fly (True=train, False=test)
    cfg_aug : dict | None konfigurasi dari train.yaml['augmentation']
    """

    _DEFAULT_AUG = {
        "temporal_resample": {"enabled": True,  "speed_min": 0.75, "speed_max": 1.25},
        "jitter":            {"enabled": True,  "std": 0.01},
        "scale":             {"enabled": True,  "min": 0.9,  "max": 1.1},
        "mask":              {"enabled": True,  "prob": 0.1},
    }

    def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    augment: bool = False,
    cfg_aug: dict | None = None,
    use_delta: bool = True,   
    ):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment
        self.aug     = cfg_aug if cfg_aug is not None else self._DEFAULT_AUG
        self.use_delta = use_delta                        
        self.N, self.T, self.L, _ = X.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = self.X[idx].copy()   # (T, L, 2)

        # 1. Augmentasi (train only)
        if self.augment:
            if self.aug.get("temporal_resample", {}).get("enabled", True):
                clip = self._temporal_resample(clip)
            if self.aug.get("jitter", {}).get("enabled", True):
                clip = self._spatial_jitter(clip)
            if self.aug.get("scale", {}).get("enabled", True):
                clip = self._spatial_scale(clip)
            if self.aug.get("mask", {}).get("enabled", True):
                clip = self._random_mask(clip)

        # 2. Delta & concat
        if self.use_delta:
            delta    = self._compute_delta(clip)        # (T, L, 2)
            combined = np.concatenate([clip, delta], axis=-1)  # (T, L, 4)
        else:
            combined = clip                              # (T, L, 2)

        # 3. Flatten: (T, L, C) -> (T, L*C)
        flat = combined.reshape(self.T, self.L * combined.shape[-1])

        # 4. Pseudo-image: (1, T, L*C)
        pseudo_img = flat[np.newaxis, ...]

        # 5. Pseudo-image: (T, 488) -> (1, T, 488)
        pseudo_img = flat[np.newaxis, ...]

        x_tensor = torch.from_numpy(pseudo_img.astype(np.float32))
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)

        return x_tensor, y_tensor

    # Delta

    def _compute_delta(self, clip: np.ndarray) -> np.ndarray:
        """Delta frame-to-frame. Tangan: hanya jika kedua frame valid (non-zero)."""
        delta = np.zeros_like(clip, dtype=np.float32)

        # pose & face: delta normal
        delta[1:, POSE_SLICE, :] = clip[1:, POSE_SLICE, :] - clip[:-1, POSE_SLICE, :]
        delta[1:, FACE_SLICE, :] = clip[1:, FACE_SLICE, :] - clip[:-1, FACE_SLICE, :]

        # tangan: delta hanya jika keduanya valid (mencegah fake-motion)
        for sl in [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]:
            prev = clip[:-1, sl, :]
            curr = clip[1:,  sl, :]
            prev_valid = np.any(np.abs(prev) > EPS, axis=-1)
            curr_valid = np.any(np.abs(curr) > EPS, axis=-1)
            valid = prev_valid & curr_valid
            d = curr - prev
            d[~valid] = 0.0
            delta[1:, sl, :] = d

        return delta

    # Augmentasi

    def _temporal_resample(self, clip: np.ndarray) -> np.ndarray:
        cfg   = self.aug.get("temporal_resample", {})
        speed = np.random.uniform(cfg.get("speed_min", 0.75), cfg.get("speed_max", 1.25))
        T     = clip.shape[0]
        n_src = max(2, min(int(round(T * speed)), T * 2))
        src_idx = np.linspace(0, T - 1, n_src)
        dst_idx = np.linspace(0, n_src - 1, T)
        out = np.zeros_like(clip)
        for l in range(clip.shape[1]):
            for c in range(clip.shape[2]):
                src_vals = np.interp(src_idx, np.arange(T), clip[:, l, c])
                out[:, l, c] = np.interp(dst_idx, np.arange(n_src), src_vals)
        return out

    def _spatial_jitter(self, clip: np.ndarray) -> np.ndarray:
        std  = self.aug.get("jitter", {}).get("std", 0.01)
        out  = clip + np.random.normal(0, std, clip.shape).astype(np.float32)
        for sl in [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]:
            valid = np.any(np.abs(clip[:, sl, :]) > EPS, axis=-1)
            out[:, sl, :][~valid] = 0.0
        return out

    def _spatial_scale(self, clip: np.ndarray) -> np.ndarray:
        cfg   = self.aug.get("scale", {})
        scale = np.random.uniform(cfg.get("min", 0.9), cfg.get("max", 1.1))
        return clip * scale

    def _random_mask(self, clip: np.ndarray) -> np.ndarray:
        prob = self.aug.get("mask", {}).get("prob", 0.1)
        out  = clip.copy()
        out[:, np.random.rand(self.L) < prob, :] = 0.0
        return out