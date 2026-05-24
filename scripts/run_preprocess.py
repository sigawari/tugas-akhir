# run_preprocess.py
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import argparse
from src.utils import load_config, get_logger
from src.data_handler import Preprocessor

logger = get_logger("Preprocess")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Preprocessing Data BISINDO")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['extract', 'select', 'clean', 'normalize', 'convert', 'all'],
        default='all',
        help=(
            "Tahap yang ingin dijalankan:\n"
            "  extract   – video → JSON landmark mentah (MediaPipe Holistic)\n"
            "  select    – reduksi landmark: pose 12, tangan 21, wajah 68\n"
            "  clean     – crop ke 90 frame + interpolasi tangan\n"
            "  normalize – normalisasi full-body / wajah / lengan / tangan\n"
            "  convert   – normalized JSON → X.npy, y.npy, labels.npy\n"
            "  all       – jalankan semua tahap secara berurutan"
        )
    )
    args = parser.parse_args()

    config = load_config("preprocess.yaml")
    proc = Preprocessor(config)

    logger.info(f"Memulai preprocessing tahap: {args.stage}")

    if args.stage in ['extract', 'all']:
        logger.info("Step 1/5 – Ekstraksi landmark dari video...")
        proc.extract_landmarks()

    if args.stage in ['select', 'all']:
        logger.info("Step 2/5 – Seleksi landmark (pose 12, tangan 21, wajah 68)...")
        proc.select_landmarks()

    if args.stage in ['clean', 'all']:
        logger.info("Step 3/5 – Cleaning: crop frame + interpolasi missing tangan...")
        proc.clean_data()

    if args.stage in ['normalize', 'all']:
        logger.info("Step 4/5 – Normalisasi landmark...")
        proc.normalize_data()

    if args.stage in ['convert', 'all']:
        logger.info("Step 5/5 – Konversi ke .npy...")
        proc.convert_to_npy()

    logger.info("Preprocessing selesai.")

if __name__ == "__main__":
    main()