# live_demo_pose.py
# Demo realtime: load model pose dan prediksi dari webcam pakai MediaPipe Pose.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List
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
from models import ResNet2DSign


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


def load_label_mapping() -> Dict[int, str]:
    """Baca split.json untuk mendapatkan idx2label."""
    split_path = DATA_DIR / "splits" / "split.json"
    data = read_json(split_path)
    idx2label_raw = data["idx2label"]  # keys string
    idx2label = {int(k): v for k, v in idx2label_raw.items()}
    return idx2label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/resnet18_pose_best.pt",
        help="Path ke checkpoint model pose.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu/cuda). Default: auto.",
    )
    args = parser.parse_args()

    seed(DEFAULT_SEED)

    # Device
    if args.device is None:
        device_raw = get_device(prefer_gpu=True)
    else:
        device_raw = torch.device(args.device)
    device = device_raw
    print(f"Device: {device}")

    # Label mapping
    idx2label = load_label_mapping()
    num_classes = len(idx2label)
    print("idx2label:", idx2label)

    # Load model
    # gunakan os.path untuk membangun path checkpoint
    # Load model
    # Bangun path checkpoint:
    # - kalau absolute → pakai apa adanya
    # - kalau relative → relatif terhadap folder skrip ini (src/)
    ckpt_path_str = args.ckpt
    ckpt_path = Path(ckpt_path_str)

    if not ckpt_path.is_absolute():
        script_dir = Path(__file__).resolve().parent  # .../ta-code/src
        ckpt_path = (script_dir / ckpt_path).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet2DSign(num_classes=num_classes, in_channels=4)
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

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam.")
        return

    buffer_xy: List[np.ndarray] = []  # list of (L, 2)
    pred_history = deque(maxlen=8)  # simpan 8 prediksi terakhir

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari webcam.")
                break

            h, w, _ = frame.shape

            # MediaPipe butuh RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # Ambil 33 pose landmark
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if len(lm) < NUM_LANDMARKS_POSE:
                    # kalau aneh, skip saja
                    continue

                xy = np.zeros((NUM_LANDMARKS_POSE, 2), dtype=np.float32)
                for i in range(NUM_LANDMARKS_POSE):
                    xy[i, 0] = lm[i].x  # normalized [0,1]
                    xy[i, 1] = lm[i].y  # normalized [0,1]
            else:
                # kalau tidak terdeteksi, bisa:
                # - skip frame, atau
                # - isi nol (di sini kita skip untuk menghindari noise)
                cv2.putText(
                    frame,
                    "Pose not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Sign Pose Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                continue

            # Masukkan ke buffer
            buffer_xy.append(xy)
            if len(buffer_xy) > SEQUENCE_LENGTH:
                buffer_xy.pop(0)

            pred_label = None
            pred_prob = None

            if len(buffer_xy) == SEQUENCE_LENGTH:
                # Bangun input tensor dan prediksi
                inp = build_input_tensor(buffer_xy).to(device)  # (1, 4, T, L)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    pred_label_idx = int(pred.item())
                    pred_conf = float(conf.item())

                # Masukkan ke history (selalu simpan)
                pred_history.append((pred_label_idx, pred_conf))

                # Smoothing: hanya tampilkan kalau cukup stabil dan confident
                if len(pred_history) == pred_history.maxlen:
                    labels_only = [p[0] for p in pred_history]
                    confs_only = [p[1] for p in pred_history]

                    # majority vote
                    counts = {}
                    for lbl in labels_only:
                        counts[lbl] = counts.get(lbl, 0) + 1
                    best_label_idx = max(counts, key=counts.get)
                    majority_ratio = counts[best_label_idx] / len(labels_only)

                    avg_conf = sum(confs_only) / len(confs_only)

                    # threshold bisa kamu tuning
                    if majority_ratio >= 0.6 and avg_conf >= 0.7:
                        pred_label = idx2label[best_label_idx]
                        pred_prob = avg_conf
                    else:
                        pred_label = None
                        pred_prob = None

            # Tampilkan hasil di frame
            if pred_label is not None:
                text = f"{pred_label} ({pred_prob:.2f})"
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Sign Pose Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


if __name__ == "__main__":
    main()
