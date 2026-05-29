import numpy as np

X = np.load('data/processed/npy/X.npy')
y = np.load('data/processed/npy/y.npy')
labels = np.load('data/processed/npy/labels.npy', allow_pickle=True)

# ── CEK 1: apakah urutan y sesuai kronologis? ──────────────────
print("Urutan y (harus: 0,0,...,0, 1,1,...,1, dst):")
print(y)
# Kalau acak berarti convert_to_npy ada masalah

# ── CEK 2: apakah sampel antar kelas mirip satu sama lain? ─────
# Hitung jarak L2 antar semua pasang kelas
X_flat = X.reshape(len(X), -1)
for i in range(10):
    for j in range(i+1, 10):
        xi = X_flat[y == i].mean(axis=0)
        xj = X_flat[y == j].mean(axis=0)
        dist = np.linalg.norm(xi - xj)
        print(f"  dist({labels[i]:12s}, {labels[j]:12s}) = {dist:.4f}")