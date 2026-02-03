# data_reader.py
# PyTorch Dataset & DataLoader wrapper for NPY dataset.
# Loads X/y arrays → returns tensor per sample.
#
# Struktur folder:
#   data/processed/<word>/<variant>/X.npy
#   data/processed/<word>/<variant>/y.npy
#
# Contoh:
#   dataset = SignNPYDataset(word="halo", variant="full")
#   loader  = make_dataloader(word="halo", variant="full", batch_size=8, shuffle=True)

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class SignNPYDataset(Dataset):
    """
    Dataset untuk gesture BISINDO berbasis NPY.

    Catatan penting:
    - Dataset ini berbasis (word, variant) sehingga umumnya SINGLE-CLASS per instance (y.npy berisi label yang sama untuk semua sample dalam satu word).
    - Untuk training multi-class lintas word (skema ablasi utama), gunakan pipeline train.py yang memakai split.json berisi item {word,index,label}.

    - word   : nama kata (folder di data/processed/<word>/)
    - variant: full | noface | hands | pose
    - root   : override PROCESSED_ROOT kalau perlu
    - transform: fungsi opsional untuk mengubah x (sample)
    - target_transform: fungsi opsional untuk mengubah y (label)
    """

    def __init__(
        self,
        word: str,
        variant: str = "full",
        root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_into_memory: bool = True,
    ) -> None:
        super().__init__()

        self.word = word
        self.variant = variant
        self.transform = transform
        self.target_transform = target_transform
        self.load_into_memory = load_into_memory

        if root is None:
            root = PROCESSED_ROOT

        self.data_dir = Path(root) / word / variant
        self.x_path = self.data_dir / "X.npy"
        self.y_path = self.data_dir / "y.npy"

        if not self.x_path.exists():
            raise FileNotFoundError(f"X.npy tidak ditemukan: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"y.npy tidak ditemukan: {self.y_path}")

        if load_into_memory:
            # Load seluruh data ke RAM
            X = np.load(self.x_path)
            y = np.load(self.y_path)

            if X.ndim != 3:
                raise ValueError(f"Bentuk X harus (N, T, D), dapat {X.shape}")
            if y.ndim != 1:
                raise ValueError(f"Bentuk y harus (N,), dapat {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Jumlah sampel X dan y tidak cocok: {X.shape[0]} vs {y.shape[0]}"
                )

            # Simpan sebagai tensor
            # X: (N, T, D) float32
            # y: (N,) int64
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.int64))
            self._len = self.X.shape[0]
        else:
            # Lazy loading: hanya simpan path dan info shape
            X = np.load(self.x_path, mmap_mode="r")
            y = np.load(self.y_path, mmap_mode="r")

            if X.ndim != 3:
                raise ValueError(f"Bentuk X harus (N, T, D), dapat {X.shape}")
            if y.ndim != 1:
                raise ValueError(f"Bentuk y harus (N,), dapat {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Jumlah sampel X dan y tidak cocok: {X.shape[0]} vs {y.shape[0]}"
                )

            self.X = X
            self.y = y
            self._len = X.shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if self.load_into_memory:
            x = self.X[idx]  # (T, D) tensor float32
            y = self.y[idx]  # () tensor int64
        else:
            # kalau lazy → masih numpy, convert per-sample
            x = torch.from_numpy(self.X[idx].astype(np.float32))
            y = torch.from_numpy(np.array(self.y[idx]).astype(np.int64))

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


def make_dataloader(
    word: str,
    variant: str = "full",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    root: Optional[Path] = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Helper untuk langsung bikin DataLoader dari word + variant.

    Contoh:
        loader = make_dataloader("halo", "full", batch_size=16)
        for x, y in loader:
            ...
    """
    ds = SignNPYDataset(
        word=word,
        variant=variant,
        root=root,
        **dataset_kwargs,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


if __name__ == "__main__":
    print(" SCAN SEMUA KELAS / DATASET DI data/processed ")
    print(f"Root: {PROCESSED_ROOT}")

    if not PROCESSED_ROOT.exists():
        print("❌ Folder processed belum ada.")
    else:
        # cari semua kata (word) = subfolder di processed
        word_dirs = [d for d in PROCESSED_ROOT.iterdir() if d.is_dir()]

        if not word_dirs:
            print("❌ Belum ada kata di data/processed.")
        else:
            for word_dir in sorted(word_dirs):
                word = word_dir.name
                # tiap variant = subfolder di dalam word_dir
                variant_dirs = [v for v in word_dir.iterdir() if v.is_dir()]

                if not variant_dirs:
                    print(f"\n[ {word} ] → (tidak ada variant)")
                    continue

                print(f"\n[ {word} ]")
                for vdir in sorted(variant_dirs):
                    variant = vdir.name
                    x_path = vdir / "X.npy"
                    y_path = vdir / "y.npy"

                    if not x_path.exists() or not y_path.exists():
                        continue

                    try:
                        ds = SignNPYDataset(word=word, variant=variant, load_into_memory=True)
                        # kelas unik dari y
                        classes = torch.unique(ds.y).tolist()
                        print(
                            f"  - variant: {variant:<7} | N sampel: {len(ds):<3} | kelas (unik y): {classes}"
                        )
                    except Exception as e:
                        print(f"  - variant: {variant:<7} | ❌ error: {e}")
