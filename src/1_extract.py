import os
import cv2
import json
import time
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

# =========================
# CONFIG
# =========================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "video")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "extracted")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# HELPER
# =========================
def landmark_to_dict(landmarks, name_list):
    result = {}
    if landmarks is None:
        for name in name_list:
            result[name] = {"x": 0.0, "y": 0.0, "z": 0.0}
    else:
        for idx, lm in enumerate(landmarks.landmark):
            result[name_list[idx]] = {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            }
    return result


# =========================
# LANDMARK NAMES
# =========================
POSE_NAMES = [lm.name.lower() for lm in mp_holistic.PoseLandmark]

HAND_NAMES = [str(i) for i in range(21)]
FACE_NAMES = [str(i) for i in range(468)]


# =========================
# MAIN PROCESS
# =========================
def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output_data = {
        "metadata": {
            "video_id": video_name,
            "fps": fps,
            "duration_sec": total_frames / fps if fps > 0 else 0,
            "total_frames": total_frames,
            "model": "MediaPipe Holistic",
            "action": label
        },
        "frames": []
    }

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frame_data = {
                "frame_index": frame_idx,
                "timestamp_ms": int(frame_idx * (1000 / fps)),
                "landmarks": {
                    "pose": landmark_to_dict(results.pose_landmarks, POSE_NAMES),
                    "left_hand": landmark_to_dict(results.left_hand_landmarks, HAND_NAMES),
                    "right_hand": landmark_to_dict(results.right_hand_landmarks, HAND_NAMES),
                    "face": landmark_to_dict(results.face_landmarks, FACE_NAMES)
                }
            }

            output_data["frames"].append(frame_data)
            frame_idx += 1

    cap.release()
    return output_data


# =========================
# LOOP ALL DATA
# =========================
def main():
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        save_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(save_dir, exist_ok=True)

        videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]

        print(f"\nProcessing label: {label} ({len(videos)} videos)")

        for vid in tqdm(videos):
            video_path = os.path.join(label_path, vid)

            data = process_video(video_path, label)

            save_path = os.path.join(save_dir, vid.replace(".mp4", ".json"))

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()