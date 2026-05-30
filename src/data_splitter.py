"""
data_splitter.py
----------------
Stratified K-Fold split untuk dataset BISINDO.

Karena semua 15 klip per kelas direkam dalam sesi yang sama
dan kondisi yang setara, K-Fold stratified lebih representatif
dari chronological split — setiap klip pernah jadi val tepat sekali.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def kfold_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stratified K-Fold split per kelas.

    Returns
    -------
    List of (X_train, X_val, y_train, y_val) untuk setiap fold.
    Dengan 15 klip/kelas dan K=5: train=12 klip/kelas, val=3 klip/kelas.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    print("=" * 50)
    print(f"  Stratified {n_splits}-Fold CV")
    print("=" * 50)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        folds.append((X_train, X_val, y_train, y_val))

        _, val_counts = np.unique(y_val, return_counts=True)
        print(
            f"  Fold {fold_idx + 1}: train={len(X_train)}  val={len(X_val)}"
            f"  ({val_counts[0]} klip/kelas)",
            flush=True,
        )

    print("=" * 50)
    return folds


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    n_total: int = 15,
    n_val: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split X, y secara kronologis per kelas (legacy)."""
    n_train = n_total - n_val
    classes = np.unique(y)
    train_idx, val_idx = [], []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) != n_total:
            raise ValueError(
                f"Kelas {cls} punya {len(cls_idx)} sampel, expected {n_total}."
            )
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:].tolist())

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]