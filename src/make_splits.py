# make_splits.py
# Buat train/val/test split (70/15/15) berbasis index per kata.
#
# Struktur data:
#   data/processed/<word>/<variant>/{X.npy, y.npy}
#
# Split TIDAK tergantung variant, hanya berdasarkan (word, index),
# sehingga bisa dipakai ulang untuk full/hands/noface/pose.
#
# Output: data/splits/split_shared_index_t70_v15_te15_seed3.json

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import (
    DEFAULT_SEED,
    PROCESSED_DIR,
    DATA_DIR,
    ensure_dir,
    write_json,
    seed,
)

# Sumber kebenaran label (harus konsisten dengan build_dataset.py)
from archive.build_dataset import WORD_LABEL_MAP

# Rasio split
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15  # sisa otomatis

# Folder untuk menyimpan deskripsi split (TIDAK di dalam processed)
SPLIT_DIR: Path = DATA_DIR / "splits"
SPLIT_FILENAME: str = (
    f"split_shared_index_t{int(TRAIN_RATIO * 100)}"
    f"_v{int(VAL_RATIO * 100)}"
    f"_te{int(TEST_RATIO * 100)}"
    f"_seed{DEFAULT_SEED}.json"
)
SPLIT_PATH: Path = SPLIT_DIR / SPLIT_FILENAME


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _discover_words(processed_dir: Path) -> List[str]:
    """Temukan daftar kata (subfolder) di data/processed.

    Hanya ambil folder yang:
      - adalah direktori
      - bukan 'splits'
      - punya minimal satu variant (full/hands/noface/pose) dengan X.npy
    """
    words: List[str] = []

    for entry in sorted(processed_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "splits":
            continue

        # Cek apakah di dalamnya ada salah satu variant dengan X.npy
        has_variant = False
        for variant in ("full", "hands", "noface", "pose"):
            x_path = entry / variant / "X.npy"
            if x_path.is_file():
                has_variant = True
                break

        if has_variant:
            words.append(entry.name)

    if not words:
        raise RuntimeError(f"Tidak ditemukan kata apapun di {processed_dir}")

    return words


def _get_num_samples_for_word(word_dir: Path, base_variant: str = "pose") -> int:
    """Ambil jumlah sampel untuk satu kata dari sebuah variant.

    Default pakai variant 'pose' (training awal pakai pose),
    tapi secara konsep N-nya diharapkan sama untuk semua variant.
    """
    x_path = word_dir / base_variant / "X.npy"
    if not x_path.is_file():
        raise FileNotFoundError(
            f"File X.npy untuk variant '{base_variant}' tidak ditemukan: {x_path}"
        )

    # Bentuk tipikal: (N, T, D) atau (N, D, T), yang penting dimensi pertama = N.
    X = np.load(x_path, mmap_mode="r")
    return int(X.shape[0])


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _stratified_split_per_word(
    words: List[str],
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:
    """Buat stratified split 70/15/15 per kata, lalu digabung.

    Returns
    -------
    train_samples : list of dict
    val_samples   : list of dict
    test_samples  : list of dict
    label2idx     : dict (word -> label)
    """
    # Pakai WORD_LABEL_MAP sebagai satu-satunya sumber label
    missing = [w for w in words if w not in WORD_LABEL_MAP]
    if missing:
        raise ValueError(
            "Words belum ada di WORD_LABEL_MAP. Tambahkan dulu di build_dataset.py: "
            + ", ".join(sorted(missing))
        )

    label2idx: Dict[str, int] = {w: int(WORD_LABEL_MAP[w]) for w in sorted(words)}

    # Validasi label harus 0..C-1 tanpa loncat (biar metrik & idx2label aman)
    uniq = sorted(set(label2idx.values()))
    if uniq != list(range(len(uniq))):
        raise ValueError(
            f"Label di WORD_LABEL_MAP harus contiguous 0..C-1. Dapat: {uniq}"
        )

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    # RNG lokal biar reproducible
    rng = np.random.default_rng(DEFAULT_SEED)

    for word in sorted(words):
        word_dir = PROCESSED_DIR / word
        n_samples = _get_num_samples_for_word(word_dir, base_variant="pose")

        # 0..N-1 untuk kata ini
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_train = int(n_samples * TRAIN_RATIO)
        n_val = int(n_samples * VAL_RATIO)
        # sisanya ke test
        # n_test = n_samples - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        def _make(word_indices) -> List[Dict]:
            return [
                {"word": word, "index": int(i), "label": int(label2idx[word])}
                for i in word_indices
            ]

        train_samples.extend(_make(train_idx))
        val_samples.extend(_make(val_idx))
        test_samples.extend(_make(test_idx))

    return train_samples, val_samples, test_samples, label2idx


def build_split() -> Dict:
    """Buat deskripsi split_shared_index_*.json berbasis index per kata.

    Split bersifat stratified per-kelas (per kata):
      - Tiap kata di-split 70/15/15 sendiri-sendiri
      - Lalu digabung ke global train/val/test
    """
    words = _discover_words(PROCESSED_DIR)

    train_samples, val_samples, test_samples, label2idx = _stratified_split_per_word(
        words
    )

    split = {
        "meta": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "seed": DEFAULT_SEED,
            "num_samples": len(train_samples) + len(val_samples) + len(test_samples),
            "num_classes": len(label2idx),
            "words": sorted(words),
            "split_type": "shared_index_per_word",
        },
        "label2idx": label2idx,
        "idx2label": {str(v): k for k, v in label2idx.items()},
        "splits": {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        },
    }

    return split


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Seed global (opsional, tapi bagus buat konsisten)
    seed()

    # Pastikan folder data/splits/ ada
    ensure_dir(SPLIT_DIR)

    split = build_split()

    # Simpan ke JSON
    write_json(SPLIT_PATH, split, indent=2)

    print(f"Saved split to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
