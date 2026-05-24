"""
data_splitter.py
----------------
Chronological train/test split untuk dataset BISINDO.
"""

import numpy as np


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    n_total: int = 15,
    n_test: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split X, y secara kronologis per kelas."""
    n_train = n_total - n_test
    classes = np.unique(y)

    train_idx = []
    test_idx  = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]

        if len(cls_idx) != n_total:
            raise ValueError(
                f"Kelas {cls} punya {len(cls_idx)} sampel, expected {n_total}. "
                f"Pastikan semua kelas punya jumlah klip yang sama."
            )

        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

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