# utils.py
# Helper utilities:
#   - global constants (seed, paths)
#   - seeding (python / numpy / torch)
#   - device selection (CPU / GPU)
#   - filesystem helpers
#   - JSON I/O
#   - optional Weights & Biases helpers

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for this helper
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Global constants & paths
# ---------------------------------------------------------------------------

# Seed global yang dipakai di seluruh project
DEFAULT_SEED: int = 3

# Root project (src/ ada di dalam root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"


def get_project_root() -> Path:
    """Return path root project."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Return path folder data/."""
    return DATA_DIR


def get_processed_dir() -> Path:
    """Return path folder data/processed/."""
    return PROCESSED_DIR


def get_raw_dir() -> Path:
    """Return path folder data/raw/."""
    return RAW_DIR


# ---------------------------------------------------------------------------
# Seeding utilities
# ---------------------------------------------------------------------------

def seed(seed: int = DEFAULT_SEED) -> int:
    """Set random seed untuk python, numpy, dan torch (jika ada).

    Parameters
    ----------
    seed : int
        Nilai seed untuk semua RNG.

    Returns
    -------
    int
        Seed yang dipakai (boleh di-print untuk log).
    """
    # Python & OS
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # Torch (kalau tersedia)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Untuk hasil deterministik (bisa sedikit lambat)
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        except Exception:
            # Kalau backend belum tersedia / beda versi, abaikan saja
            pass

    return seed


# ---------------------------------------------------------------------------
# Device / CPU-GPU utilities
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> Any:
    """Return torch.device untuk komputasi atau string fallback jika torch tidak ada.

    Parameters
    ----------
    prefer_gpu : bool
        Jika True dan CUDA tersedia, pakai CUDA, jika tidak pakai CPU.

    Returns
    -------
    device : torch.device | str
        torch.device("cuda"/"cpu") ketika torch ter-install, selain itu "cpu".
    """

    if torch is None:
        # Torch tidak ter-install – selalu pakai CPU (string)
        return "cpu"

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info() -> None:
    """Print info singkat tentang ketersediaan CPU/GPU."""

    if torch is None:
        print("torch tidak ter-install, menjalankan di CPU saja")
        return

    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        current = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current)
        print(f"Menggunakan GPU: {name} (index {current}) | total GPU: {num}")
    else:
        print("CUDA tidak tersedia, menggunakan CPU")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Buat folder kalau belum ada.

    Parameters
    ----------
    path : str | Path
        Path folder yang ingin dipastikan ada.

    Returns
    -------
    Path
        Objek Path untuk folder tersebut.
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    """Baca file JSON dan kembalikan objek Python-nya."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Tulis objek Python ke file JSON."""

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Weights & Biases helpers (optional)
# ---------------------------------------------------------------------------

try:
    import wandb  # type: ignore
except ImportError:  # wandb is optional
    wandb = None  # type: ignore


def wandb_init(project: str, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
    """Inisialisasi Weights & Biases run jika wandb ter-install.

    Mengembalikan wandb.run atau None kalau wandb tidak tersedia.
    """

    if wandb is None:
        print("wandb tidak ter-install; melewati wandb.init()")
        return None

    return wandb.init(project=project, config=config, **kwargs)


def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrik ke wandb jika tersedia."""

    if wandb is None or wandb.run is None:  # type: ignore[attr-defined]
        return

    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def wandb_finish() -> None:
    """Akhiri wandb run saat ini jika aktif."""

    if wandb is None:
        return

    try:
        wandb.finish()
    except Exception:
        # Abaikan error kalau tidak ada run aktif
        pass
