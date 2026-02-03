# live_demo_pose.py
# Demo realtime: load model pose dan prediksi dari webcam pakai MediaPipe Pose.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from utils import (
    DATA_DIR,
    DEFAULT_SEED,
    seed,
    get_device,
    read_json,
)
from models import Basic2DCNN, ResNet18, ResNet34, ResNet50


SEQUENCE_LENGTH = 30
NUM_LANDMARKS_POSE = 33  # sesuai POSE_LANDMARK_NAMES di build_dataset.py


def build_input_tensor(buffer_xy: List[np.ndarray]) -> torch.Tensor:
    """buffer_xy: list of (L, 2) for each frame, len = T (30).
    Return tensor shape (1, 4, T, L) = (batch=1, channels=4, time, landmarks).
    """

    T = len(buffer_xy)
    L = NUM_LANDMARKS_POSE

    # (T, L, 2) : 2 = (x, y)
    arr = np.stack(buffer_xy, axis=0)  # (T, L, 2)
    x = arr[..., 0]  # (T, L)
    y = arr[..., 1]  # (T, L)

    # hitung dx, dy
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[1:, :] = x[1:, :] - x[:-1, :]
    dy[1:, :] = y[1:, :] - y[:-1, :]

    # (T, L, 4)
    feats = np.stack([x, y, dx, dy], axis=-1)
    # -> (4, T, L)
    feats = np.transpose(feats, (2, 0, 1))

    # (1, 4, T, L)
    tensor = torch.from_numpy(feats).float().unsqueeze(0)
    return tensor


def load_label_mapping(split_path: Path) -> Dict[int, str]:
    """Baca split.json untuk mendapatkan idx2label."""
    data = read_json(split_path)
    idx2label_raw = data["idx2label"]  # keys string
    idx2label = {int(k): v for k, v in idx2label_raw.items()}
    return idx2label


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name in ("cnn2d", "basic2dcnn", "cnn"):
        return Basic2DCNN(num_classes=num_classes, in_channels=4)
    if model_name in ("resnet18", "r18"):
        return ResNet18(num_classes=num_classes, in_channels=4)
    if model_name in ("resnet34", "r34"):
        return ResNet34(num_classes=num_classes, in_channels=4)
    if model_name in ("resnet50", "r50"):
        return ResNet50(num_classes=num_classes, in_channels=4)
    raise ValueError(f"Unknown --model: {model_name}")


def find_best_checkpoint(model: str, variant: str) -> Path:
    ckpt_root = Path("checkpoints") / model / variant
    return max(ckpt_root.glob("*__best.pt"), key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="pose", choices=["pose", "full", "noface", "hands"])
    p.add_argument("--model", type=str, default="resnet50", choices=["cnn2d", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--split_path", type=str, default=str(DATA_DIR / "splits" / "split.json"))
    p.add_argument("--ckpt", type=str, default=None, help="Optional. Kalau kosong, auto ambil *__best.pt terbaru")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--min_majority", type=float, default=0.6)
    p.add_argument("--min_conf", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()

    seed(DEFAULT_SEED)

    # Untuk saat ini demo hanya implement pose (33 landmark)
    if args.variant != "pose":
        raise ValueError(
            "live_demo.py saat ini hanya mendukung --variant pose (33 pose landmarks). "
            "Untuk demo full/hands/noface perlu extractor MediaPipe yang sesuai."
        )

    device = get_device(prefer_gpu=True) if args.device is None else torch.device(args.device)
    print(f"Device: {device}")

    split_path = Path(args.split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"split_path tidak ditemukan: {split_path}")

    idx2label = load_label_mapping(split_path)
    num_classes = len(idx2label)
    print("idx2label:", idx2label)

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_best_checkpoint(args.model, args.variant)

    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(args.model, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam.")
        return

    buffer_xy: List[np.ndarray] = []
    pred_history: deque[Tuple[int, float]] = deque(maxlen=args.history)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari webcam.")
                break

            # MediaPipe butuh RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if len(lm) < NUM_LANDMARKS_POSE:
                    continue

                xy = np.zeros((NUM_LANDMARKS_POSE, 2), dtype=np.float32)
                for i in range(NUM_LANDMARKS_POSE):
                    xy[i, 0] = lm[i].x
                    xy[i, 1] = lm[i].y
            else:
                cv2.putText(frame, "Pose not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Sign Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            buffer_xy.append(xy)
            if len(buffer_xy) > SEQUENCE_LENGTH:
                buffer_xy.pop(0)

            display_text: Optional[str] = None

            if len(buffer_xy) == SEQUENCE_LENGTH:
                inp = build_input_tensor(buffer_xy).to(device)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    pred_label_idx = int(pred.item())
                    pred_conf = float(conf.item())

                pred_history.append((pred_label_idx, pred_conf))

                if len(pred_history) == pred_history.maxlen:
                    labels_only = [p[0] for p in pred_history]
                    confs_only = [p[1] for p in pred_history]

                    counts: Dict[int, int] = {}
                    for lbl in labels_only:
                        counts[lbl] = counts.get(lbl, 0) + 1

                    best_label_idx = max(counts, key=counts.get)
                    majority_ratio = counts[best_label_idx] / len(labels_only)
                    avg_conf = sum(confs_only) / len(confs_only)

                    if majority_ratio >= args.min_majority and avg_conf >= args.min_conf:
                        display_text = f"{idx2label.get(best_label_idx, str(best_label_idx))} ({avg_conf:.2f})"

            if display_text:
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Sign Demo", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


if __name__ == "__main__":
    main()
