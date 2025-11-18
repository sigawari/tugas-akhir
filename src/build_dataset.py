# build_dataset.py
# Convert raw JSON → NPY arrays (X, y) for training.
# Supports channel selection: full, noface, hands, pose.
# Output: data/processed/npy/<variant>/X_*.npy and y_*.npy
