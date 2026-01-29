# filepath: c:\Users\hp\multimedia\ta-code\src\downscale.py

import os
from pathlib import Path

import cv2
import numpy as np
import random


def resize_to_target_height(img, target_height: int):
    """Resize so height becomes target_height, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension")

    scale = target_height / float(h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Use INTER_AREA for downscaling quality
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def downscaling(
    img_bgr,
    scale: float | None = None,
    to_rgb: bool = False,
    *,
    scale_range: tuple[float, float] = (0.25, 0.6),
    blur_prob: float = 0.7,
    noise_prob: float = 0.5,
    noise_sigma_range: tuple[float, float] = (3.0, 8.0),
    blur_ksizes: tuple[int, ...] = (3, 5),
    seed: int | None = None,
):
    """Degradative resize (lebih realistis dari resize murni).

    Pipeline:
    1) random downscale (atau pakai `scale` kalau diberi)
    2) optional Gaussian blur
    3) optional additive Gaussian noise
    4) upscale balik ke ukuran semula (biar kompatibel dengan pipeline/landmark)

    Catatan:
    - OpenCV imread menghasilkan BGR.
    - cv2.imwrite mengharapkan BGR. Jadi JANGAN simpan hasil yang sudah di-convert ke RGB.
    - Konversi ke RGB hanya untuk ditampilkan (matplotlib).
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Tentukan scale
    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        chosen_scale = float(scale)
    else:
        lo, hi = scale_range
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError("scale_range must be positive and (min<=max)")
        chosen_scale = random.uniform(lo, hi)

    new_w = max(1, int(round(w * chosen_scale)))
    new_h = max(1, int(round(h * chosen_scale)))

    small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Optional blur (simulasi lensa/motion kecil)
    if blur_prob > 0 and random.random() < blur_prob:
        k = int(random.choice(blur_ksizes))
        if k % 2 == 0 or k < 1:
            raise ValueError("blur_ksizes must contain positive odd integers")
        small = cv2.GaussianBlur(small, (k, k), 0)

    # Optional noise (sensor noise)
    if noise_prob > 0 and random.random() < noise_prob:
        sig_lo, sig_hi = noise_sigma_range
        if sig_lo < 0 or sig_hi < 0 or sig_hi < sig_lo:
            raise ValueError("noise_sigma_range must be (min>=0, max>=min)")
        sigma = random.uniform(sig_lo, sig_hi)
        noise = np.random.normal(0.0, sigma, small.shape).astype(np.float32)
        small_f = small.astype(np.float32) + noise
        small = np.clip(small_f, 0, 255).astype(np.uint8)

    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if to_rgb:
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
    return degraded


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Downscale foto1.png -> foto1_720p.png / foto1_480p.png (atau degradative scale-based)")
    parser.add_argument("--input", default=None, help="Path input image (default: <project_root>/photo/foto1.png)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <project_root>/photo)")
    parser.add_argument("--mode", choices=["height", "scale"], default="height", help="Resize mode")
    parser.add_argument("--height", type=int, default=720, help="Target height (mode=height)")
    parser.add_argument("--scale", type=float, default=None, help="Fixed scale ratio (mode=scale). If omitted, uses --scale-min/--scale-max")
    parser.add_argument("--scale-min", type=float, default=0.25, help="Min random scale (mode=scale)")
    parser.add_argument("--scale-max", type=float, default=0.60, help="Max random scale (mode=scale)")
    parser.add_argument("--blur-prob", type=float, default=0.7, help="Probability apply Gaussian blur (mode=scale)")
    parser.add_argument("--noise-prob", type=float, default=0.5, help="Probability apply Gaussian noise (mode=scale)")
    parser.add_argument("--noise-sigma-min", type=float, default=3.0, help="Min sigma for Gaussian noise (mode=scale)")
    parser.add_argument("--noise-sigma-max", type=float, default=8.0, help="Max sigma for Gaussian noise (mode=scale)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (mode=scale)")
    parser.add_argument("--suffix", default=None, help="Custom suffix for output filename (mode=scale)")
    args = parser.parse_args()

    # src/downscale.py -> project root -> photo/foto1.png
    project_root = Path(__file__).resolve().parents[1]
    default_photo_dir = project_root / "photo"

    input_path = Path(args.input) if args.input else (default_photo_dir / "foto1.png")
    out_dir = Path(args.out_dir) if args.out_dir else default_photo_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "height":
        # Default behavior: create 720p and 480p
        hd_720 = resize_to_target_height(img, 720)
        sd_480 = resize_to_target_height(img, 480)

        out_hd = out_dir / "foto1_720p.png"
        out_sd = out_dir / "foto1_480p.png"

        if not cv2.imwrite(str(out_hd), hd_720):
            raise IOError(f"Failed to write output: {out_hd}")
        if not cv2.imwrite(str(out_sd), sd_480):
            raise IOError(f"Failed to write output: {out_sd}")

        print(f"Saved: {out_hd}")
        print(f"Saved: {out_sd}")
        return

    # mode == scale (degradative)
    degraded_bgr = downscaling(
        img,
        scale=args.scale,
        to_rgb=False,
        scale_range=(args.scale_min, args.scale_max),
        blur_prob=args.blur_prob,
        noise_prob=args.noise_prob,
        noise_sigma_range=(args.noise_sigma_min, args.noise_sigma_max),
        seed=args.seed,
    )

    if args.suffix:
        suffix = args.suffix
    else:
        if args.scale is not None:
            suffix = f"_deg_s{args.scale:.3f}".replace(".", "")
        else:
            suffix = f"_deg_r{args.scale_min:.3f}-{args.scale_max:.3f}".replace(".", "")
        if args.seed is not None:
            suffix += f"_seed{args.seed}"

    out_path = out_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
    if not cv2.imwrite(str(out_path), degraded_bgr):
        raise IOError(f"Failed to write output: {out_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
