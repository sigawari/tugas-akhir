## Setup Dataset

Dataset tersedia di Kaggle (±2 GB):
https://www.kaggle.com/datasets/sigawari/slr-bisindo-landmarks

### Prasyarat

1. Install dependencies:
   pip install -r requirements.txt

2. Setup Kaggle API credentials:
   - Buka https://www.kaggle.com/settings → bagian API → Create New Token
   - Simpan kaggle.json ke:
     - Linux/Mac : ~/.kaggle/kaggle.json lalu chmod 600 ~/.kaggle/kaggle.json
     - Windows : C:\Users\<username>\.kaggle\kaggle.json

### Download otomatis

python scripts/download_dataset.py

Script akan mengecek apakah dataset sudah ada.
Jika belum, download dan tempatkan ke folder yang benar secara otomatis.

### Jalankan pipeline

# Jika punya video raw (dari awal):

python scripts/run_preprocess.py --stage all

# Jika pakai dataset Kaggle (langsung dari normalized):

python scripts/run_preprocess.py --stage convert

# Training:

python scripts/run_train.py
