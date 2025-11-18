import cv2
import numpy as np
import os
import mediapipe as mp
import json
import time

# === Setup Mediapipe ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# === Path dataset ===
DATA_JSON_PATH = os.path.join(os.getcwd(), 'data_json')
DATA_VIDEO_PATH = os.path.join(os.getcwd(), 'data_video')
os.makedirs(DATA_JSON_PATH, exist_ok=True)
os.makedirs(DATA_VIDEO_PATH, exist_ok=True)

# === Konfigurasi dataset ===
no_sequences = 30        # jumlah video per kata
sequence_length = 30     # jumlah frame per video (sekitar 1 detik jika 30 fps)

# === Mapping nama landmark Pose (33 titik) ===
pose_landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# === Mapping tangan kiri dan kanan (21 titik) ===
hand_landmark_names = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# === Fungsi ekstraksi landmark jadi dict JSON ===
def extract_keypoints_dict(results):
    data = {}

    # Pose landmarks
    if results.pose_landmarks:
        data["pose"] = {
            pose_landmark_names[i]: {
                "x": res.x, "y": res.y, "z": res.z
            }
            for i, res in enumerate(results.pose_landmarks.landmark)
            if i < len(pose_landmark_names)
        }
    else:
        data["pose"] = {}

    # Face landmarks — SIMPAN SEMUA titik (468 titik)
    if results.face_landmarks:
        data["face"] = {
            str(i): {           # key = index titik sebagai string, misal "0", "1", ..., "467"
                "x": res.x,
                "y": res.y,
                "z": res.z
            }
            for i, res in enumerate(results.face_landmarks.landmark)
        }
    else:
        data["face"] = {}

    # Left hand landmarks
    if results.left_hand_landmarks:
        data["left_hand"] = {
            hand_landmark_names[i]: {
                "x": res.x, "y": res.y, "z": res.z
            }
            for i, res in enumerate(results.left_hand_landmarks.landmark)
            if i < len(hand_landmark_names)
        }
    else:
        data["left_hand"] = {}

    # Right hand landmarks
    if results.right_hand_landmarks:
        data["right_hand"] = {
            hand_landmark_names[i]: {
                "x": res.x, "y": res.y, "z": res.z
            }
            for i, res in enumerate(results.right_hand_landmarks.landmark)
            if i < len(hand_landmark_names)
        }
    else:
        data["right_hand"] = {}

    return data

# === Fungsi utilitas deteksi Mediapipe ===
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# === Fungsi untuk menggambar landmark (opsional) ===
def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

# === Proses utama pengumpulan data ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    stop_all = False

    while not stop_all:
        # === Input kata secara manual ===
        action = input("\nMasukkan kata (atau ketik 'q' untuk selesai): ").strip()
        if action.lower() == 'q':
            break

        if not action:
            print("Kata tidak boleh kosong.")
            continue

        print(f"\n🔹 Mengumpulkan data untuk kata: {action}")

        # Siapkan folder untuk kata ini
        json_action_dir = os.path.join(DATA_JSON_PATH, action)
        video_action_dir = os.path.join(DATA_VIDEO_PATH, action)
        os.makedirs(json_action_dir, exist_ok=True)
        os.makedirs(video_action_dir, exist_ok=True)

        for sequence in range(1, no_sequences + 1):
            print(f"  ▶ Sequence {sequence}/{no_sequences}")

            sequence_data = {
                "metadata": {
                    "video_id": f"{action}_sequence_{sequence}",
                    "fps": 30,
                    "duration_sec": sequence_length / 30,
                    "total_frames": sequence_length,
                    "model": "MediaPipe Holistic",
                    "action": action
                },
                "frames": []
            }

            start_time_ms = int(round(time.time() * 1000))

            # === Siapkan VideoWriter (nanti di-init setelah dapat frame pertama) ===
            video_filename = os.path.join(video_action_dir, f"sequence_{sequence}.mp4")
            out = None

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari kamera.")
                    stop_all = True
                    break

                # Inisialisasi VideoWriter pakai ukuran frame pertama
                if out is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_filename, fourcc, 30, (w, h))

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                timestamp_ms = start_time_ms + int((frame_num / 30) * 1000)
                landmarks = extract_keypoints_dict(results)

                frame_data = {
                    "frame_index": frame_num,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": landmarks
                }
                sequence_data["frames"].append(frame_data)

                # UI indikator
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (60, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting "{action}" #{sequence}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, f'{action} | Seq {sequence} | Frame {frame_num}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # Tampilkan dan simpan ke video
                cv2.imshow('OpenCV Feed', image)
                out.write(image)

                # Tombol keluar darurat
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop_all = True
                    break

            # Rilis VideoWriter
            if out is not None:
                out.release()

            # Simpan JSON
            json_path = os.path.join(json_action_dir, f'sequence_{sequence}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ JSON    disimpan: {json_path}")
            print(f"  🎥 Video   disimpan: {video_filename}")

            if stop_all:
                break

cap.release()
cv2.destroyAllWindows()
print("\nPengumpulan data selesai!")
