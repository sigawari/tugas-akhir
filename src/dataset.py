from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =========================
# PATH CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "npy_dataset"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "dataset" / "splits" / "split_80_20.json"

# =========================
# GLOBAL SHAPE CONFIG
# =========================

T_FRAMES = 90
L_LANDMARKS = 122
IN_CHANNELS_XY = 2
IN_CHANNELS_XYDXDY = 4

# Landmark slices sesuai urutan .npy kamu
POSE_SLICE = slice(0, 12)
LEFT_HAND_SLICE = slice(12, 33)
RIGHT_HAND_SLICE = slice(33, 54)
FACE_SLICE = slice(54, 122)

EPS = 1e-12


# =========================
# IO HELPERS
# =========================

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_index(dataset_dir: Path = DATASET_DIR) -> List[Dict[str, Any]]:
    index_path = dataset_dir / "dataset_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"dataset_index.json tidak ditemukan: {index_path}")
    return read_json(index_path)


# =========================
# SPLIT HELPERS
# =========================

def build_split_from_dataset_index(
    dataset_index: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Split file-level stratified per label.
    Aman dari data leakage antar split karena unit split = file/video.
    """
    if abs((train_ratio + val_ratio ) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio harus = 1.0")

    rng = np.random.default_rng(seed)

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset_index:
        label = item["label"]
        by_label.setdefault(label, []).append(item)

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []

    for label in sorted(by_label.keys()):
        items = by_label[label].copy()
        rng.shuffle(items)

        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_items.extend(items[:n_train])
        val_items.extend(items[n_train:n_train + n_val])

        assert len(items[:n_train]) == n_train
        assert len(items[n_train:n_train + n_val]) == n_val

    label_map = {}
    for item in dataset_index:
        label_map[item["label"]] = int(item["label_id"])

    idx2label = {str(v): k for k, v in sorted(label_map.items(), key=lambda x: x[1])}

    return {
        "meta": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "num_samples": len(dataset_index),
            "num_classes": len(label_map),
            "split_type": "file_level_stratified"
        },
        "label2idx": label_map,
        "idx2label": idx2label,
        "splits": {
            "train": train_items,
            "val": val_items,
        }
    }


def save_split_json(split_data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)


def load_or_create_split(
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    dataset_dir: Path = DATASET_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    split_path = Path(split_path)

    if split_path.is_file():
        return read_json(split_path)

    dataset_index = load_dataset_index(dataset_dir)
    split_data = build_split_from_dataset_index(
        dataset_index=dataset_index,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    save_split_json(split_data, split_path)
    return split_data


# =========================
# AUGMENTATION
# =========================

def add_landmark_jitter(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Gaussian noise pada koordinat (x,y).
    Hanya untuk titik valid hand; pose/face selalu boleh dijitter.
    x shape: (T, L, 2)
    """
    out = x.copy()
    noise = np.random.normal(loc=0.0, scale=std, size=out.shape).astype(np.float32)

    # pose + face: jitter langsung
    out[:, POSE_SLICE, :] += noise[:, POSE_SLICE, :]
    out[:, FACE_SLICE, :] += noise[:, FACE_SLICE, :]

    # hand: hanya jitter jika titik valid (bukan hand absent = (0,0))
    for hand_slice in [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]:
        hand = out[:, hand_slice, :]
        hand_noise = noise[:, hand_slice, :]
        valid_mask = np.any(np.abs(hand) > EPS, axis=-1)  # (T, L_hand)
        hand[valid_mask] += hand_noise[valid_mask]
        out[:, hand_slice, :] = hand

    return out


def apply_random_landmark_mask(x: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """
    Random landmark masking pada level (T, L).
    Kalau ter-mask, set (x,y) = 0.
    Aman untuk simulasi detection drop.
    """
    out = x.copy()
    mask = np.random.rand(out.shape[0], out.shape[1]) < prob
    out[mask] = 0.0
    return out


def apply_train_augmentations(
    x: np.ndarray,
    jitter_prob: float = 0.5,
    jitter_std: float = 0.01,
    mask_prob: float = 0.5,
    mask_ratio: float = 0.05,
) -> np.ndarray:
    out = x.copy()

    if np.random.rand() < jitter_prob:
        out = add_landmark_jitter(out, std=jitter_std)

    if np.random.rand() < mask_prob:
        out = apply_random_landmark_mask(out, prob=mask_ratio)

    return out


# =========================
# DELTA FEATURE ENGINEERING
# =========================

def compute_delta_xy(x: np.ndarray) -> np.ndarray:
    """
    Input:
        x: (T, L, 2) -> channel posisi (x, y)
    Output:
        delta: (T, L, 2) -> channel delta (dx, dy)

    Aturan anti-fake-motion:
    - pose: selalu delta normal
    - face: selalu delta normal
    - hands: delta hanya valid jika prev dan curr sama-sama non-zero
    """
    if x.shape != (T_FRAMES, L_LANDMARKS, IN_CHANNELS_XY):
        raise ValueError(f"Expected shape {(T_FRAMES, L_LANDMARKS, IN_CHANNELS_XY)}, got {x.shape}")

    delta = np.zeros_like(x, dtype=np.float32)

    if x.shape[0] <= 1:
        return delta

    # pose dan face: delta biasa
    delta[1:, POSE_SLICE, :] = x[1:, POSE_SLICE, :] - x[:-1, POSE_SLICE, :]
    delta[1:, FACE_SLICE, :] = x[1:, FACE_SLICE, :] - x[:-1, FACE_SLICE, :]

    # hands: delta masked by validity
    for hand_slice in [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]:
        prev_hand = x[:-1, hand_slice, :]   # (T-1, 21, 2)
        curr_hand = x[1:, hand_slice, :]    # (T-1, 21, 2)

        prev_valid = np.any(np.abs(prev_hand) > EPS, axis=-1)  # (T-1, 21)
        curr_valid = np.any(np.abs(curr_hand) > EPS, axis=-1)  # (T-1, 21)
        valid_mask = prev_valid & curr_valid

        d = curr_hand - prev_hand
        d[~valid_mask] = 0.0

        delta[1:, hand_slice, :] = d

    return delta


def build_multichannel_features(xy: np.ndarray, use_delta: bool = True) -> np.ndarray:
    """
    xy: (T, L, 2)
    return:
        if use_delta:
            (T, L, 4) → (x, y, dx, dy)
        else:
            (T, L, 2) → (x, y)
    """

    if not use_delta:
        return xy.astype(np.float32)

    dx = np.diff(xy[:, :, 0], axis=0, prepend=xy[0:1, :, 0])
    dy = np.diff(xy[:, :, 1], axis=0, prepend=xy[0:1, :, 1])

    feat = np.concatenate(
        [
            xy,
            dx[..., None],
            dy[..., None],
        ],
        axis=-1,
    )

    return feat.astype(np.float32)


# =========================
# DATASET
# =========================

class SignLanguageNPYDataset(Dataset):
    """
    Dataset untuk input model:
      output x: (C, T, L)
      output y: label_id
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        augment: bool = False,
        jitter_prob: float = 0.5,
        jitter_std: float = 0.01,
        mask_prob: float = 0.5,
        mask_ratio: float = 0.05,
        use_delta: bool = True,
    ) -> None:
        super().__init__()
        self.items = items
        self.augment = augment
        self.jitter_prob = jitter_prob
        self.jitter_std = jitter_std
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.use_delta = use_delta

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # 🔧 FIX: Resolve path relatif terhadap PROJECT_ROOT saat runtime
        npy_path = PROJECT_ROOT / item["npy_file"]

        # Validasi (opsional tapi sangat disarankan agar error lebih jelas)
        if not npy_path.exists():
            raise FileNotFoundError(f"❌ File .npy tidak ditemukan: {npy_path}")

        x = np.load(npy_path).astype(np.float32)   # (T, L, 2)

        if x.shape != (T_FRAMES, L_LANDMARKS, IN_CHANNELS_XY):
            raise ValueError(f"Shape salah pada {npy_path}: {x.shape}")

        # augmentasi hanya di train
        if self.augment:
            x = apply_train_augmentations(
                x,
                jitter_prob=self.jitter_prob,
                jitter_std=self.jitter_std,
                mask_prob=self.mask_prob,
                mask_ratio=self.mask_ratio,
            )

        # feature engineering on-the-fly
        feat = build_multichannel_features(x, use_delta=self.use_delta)     # (T, L, 4)

        # channel-first untuk model CNN 2D / ResNet
        feat = np.transpose(feat, (2, 0, 1))       # (4, T, L)

        x_tensor = torch.from_numpy(feat).float()
        y_tensor = torch.tensor(int(item["label_id"]), dtype=torch.long)

        return {
            "x": x_tensor,
            "y": y_tensor,
            "video_id": item["video_id"],
            "label": item["label"],
            "label_id": int(item["label_id"]),
            "npy_file": str(item["npy_file"]),
        }


# =========================
# DATALOADER FACTORY
# =========================

def create_datasets(
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    dataset_dir: Path = DATASET_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_augment: bool = True,
    jitter_prob: float = 0.5,
    jitter_std: float = 0.01,
    mask_prob: float = 0.5,
    mask_ratio: float = 0.05,
    use_delta: bool = True,
) -> Tuple[SignLanguageNPYDataset, SignLanguageNPYDataset, SignLanguageNPYDataset, Dict[str, Any]]:
    split_data = load_or_create_split(
        split_path=split_path,
        dataset_dir=dataset_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_ds = SignLanguageNPYDataset(
        items=split_data["splits"]["train"],
        augment=train_augment,
        jitter_prob=jitter_prob,
        jitter_std=jitter_std,
        mask_prob=mask_prob,
        mask_ratio=mask_ratio,
        use_delta=use_delta,
    )

    val_ds = SignLanguageNPYDataset(
        items=split_data["splits"]["val"],
        augment=False,
        use_delta=use_delta,
    )

    return train_ds, val_ds, split_data


def create_dataloaders(
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    dataset_dir: Path = DATASET_DIR,
    batch_size: int = 16,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_augment: bool = True,
    jitter_prob: float = 0.5,
    jitter_std: float = 0.01,
    mask_prob: float = 0.5,
    mask_ratio: float = 0.05,
    use_delta: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    train_ds, val_ds, split_data = create_datasets(
        split_path=split_path,
        dataset_dir=dataset_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        train_augment=train_augment,
        jitter_prob=jitter_prob,
        jitter_std=jitter_std,
        mask_prob=mask_prob,
        mask_ratio=mask_ratio,
        use_delta=use_delta,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, split_data


# =========================
# DEBUG / DEMO
# =========================

def debug_one_batch():
    train_loader, val_loader, split_data = create_dataloaders(
        batch_size=4,
        num_workers=0,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=42,
        train_augment=True,
    )

    print("\n=== SPLIT INFO ===")
    print(split_data["meta"])
    print("train:", len(split_data["splits"]["train"]))
    print("val  :", len(split_data["splits"]["val"]))

    batch = next(iter(train_loader))

    print("\n=== ONE BATCH DEBUG ===")
    print("x shape:", batch["x"].shape)   # (B, 4, T, L)
    print("y shape:", batch["y"].shape)   # (B,)

    print("\nContoh sample pertama:")
    print("video_id :", batch["video_id"][0])
    print("label    :", batch["label"][0])
    print("label_id :", batch["label_id"][0])

    x0 = batch["x"][0]  # (4, T, L)
    print("x0.shape:", x0.shape)

    print("\nChannel meaning:")
    print("  0 = x")
    print("  1 = y")
    print("  2 = dx")
    print("  3 = dy")

    # contoh indexing
    c, t, l = 0, 0, 0
    print(f"\nContoh x0[{c}, {t}, {l}] = {x0[c, t, l].item()}  -> channel x, frame 0, landmark 0")
    c, t, l = 2, 10, 5
    print(f"x0[{c}, {t}, {l}] = {x0[c, t, l].item()}  -> channel dx, frame 10, landmark 5")


if __name__ == "__main__":
    debug_one_batch()