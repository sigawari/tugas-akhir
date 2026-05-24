"""
data_splitter.py
----------------
Chronological train/test split untuk dataset BISINDO.

Strategi:
- Per kelas: klip 01–12 → Train, klip 13–15 → Test
- TIDAK random shuffle → mencegah temporal leakage
- Asumsi X.npy disusun per label, per file, sorted (sesuai convert_to_npy)

Penggunaan:
from src.data_splitter import chronological_split

X_train, X_test, y_train, y_test = chronological_split(
    X, y, n_total=15, n_test=3
)
"""

import numpy as np


def chronological_split(
X: np.ndarray,
y: np.ndarray,
n_total: int = 15,
n_test: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
"""Split X, y secara kronologis per kelas.

Parameters
----------
X : np.ndarray  shape (N, T, L, 2)
y : np.ndarray  shape (N,)
n_total : int   jumlah klip per kelas (default 15)
n_test  : int   jumlah klip terakhir per kelas untuk test (default 3)

Returns
-------
X_train, X_test, y_train, y_test

Contoh shape dengan 10 kelas, n_total=15, n_test=3:
    X_train : (120, 90, 122, 2)   ← 10 kelas × 12 klip
    X_test  : ( 30, 90, 122, 2)   ← 10 kelas ×  3 klip
"""
n_train = n_total - n_test
classes = np.unique(y)

train_idx = []
test_idx  = []

for cls in classes:
    # Indeks global semua sampel kelas ini (urutan sudah kronologis dari convert_to_npy)
    cls_idx = np.where(y == cls)[0]

    if len(cls_idx) != n_total:
        raise ValueError(
            f"Kelas {cls} punya {len(cls_idx)} sampel, expected {n_total}. "
            f"Pastikan semua kelas punya jumlah klip yang sama."
        )

    train_idx.extend(cls_idx[:n_train].tolist())   # klip 01–12
    test_idx.extend(cls_idx[n_train:].tolist())    # klip 13–15

X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

print("=" * 50)
print("  Chronological Split")
print("=" * 50)
print(f"  Train : {X_train.shape}  ({n_train} klip/kelas)")
print(f"  Test  : {X_test.shape}  ({n_test} klip/kelas)")
print(f"  Kelas : {len(classes)}")
print("=" * 50)

return X_train, X_test, y_train, y_test