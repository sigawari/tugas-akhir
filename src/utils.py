import yaml
import logging
import random
import numpy as np
import torch
from pathlib import Path

def get_root_dir():
    """Mengembalikan path folder utama proyek secara otomatis."""
    return Path(__file__).resolve().parent.parent

def load_config(config_name="experiment.yaml"):
    """Membaca file konfigurasi dari folder configs/."""
    root = get_root_dir()
    config_path = root / "configs" / config_name
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Menambahkan root_dir agar bisa diakses di file lain
    config['root_dir'] = root
    return config

def get_logger(name):
    """Setup logger agar output di terminal lebih profesional."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def set_seed(seed=42):
    """Mengunci random seed untuk memastikan riset bisa diulang (reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_path(path_str):
    """Mengubah string path menjadi objek Path absolut."""
    root = get_root_dir()
    # Jika path diberikan sebagai string relatif, gabungkan dengan root
    return root / path_str