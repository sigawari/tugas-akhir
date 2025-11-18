import cv2
import numpy as np
import os
import mediapipe as mp
import json
import time

# Referensi Siga: https://www.youtube.com/watch?v=u_Vb5cMlc8A

# === Setup Mediapipe ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# === Path dataset ===
DATA_PATH = os.path.join(os.getcwd(), 'data_json')
os.makedirs(DATA_PATH, exist_ok=True)

# === Konfigurasi dataset ===
no_sequences = 30        # jumlah video per kata
sequence_length = 30     # jumlah frame per video
actions = ['halo', 'terima_kasih']

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

# === Mapping deskriptif FaceMesh (468 titik dipilih) ===
FACE_LANDMARK_MAP = {
    234: "pipi_kiri",
    454: "pipi_kanan",
    10: "jidat_tengah",
    297: "jidat_kiri",
    338: "jidat_kanan",
    152: "dagu",
    13: "bibir_atas_tengah",
    14: "bibir_bawah_tengah",
    61: "bibir_kiri",
    291: "bibir_kanan",
    33: "mata_kiri_luar",
    133: "mata_kiri_dalam",
    362: "mata_kanan_dalam",
    263: "mata_kanan_luar"
}

# === Fungsi untuk ambil nama face landmark ===
def face_landmark_name(idx):
    return FACE_LANDMARK_MAP.get(idx, f"face_landmark_{idx}")

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
        }
    else:
        data["pose"] = {}

    # Face landmarks (pilih yang deskriptif)
    if results.face_landmarks:
        data["face"] = {
            face_landmark_name(i): {
                "x": res.x, "y": res.y, "z": res.z
            }
            for i, res in enumerate(results.face_landmarks.landmark)
            if i in FACE_LANDMARK_MAP
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
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

# === Proses utama pengumpulan data ===
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    stop = False

    for action in actions:
        if stop: break
        print(f"\n🔹 Mengumpulkan data untuk action: {action}")

        for sequence in range(1, no_sequences + 1):
            if stop: break
            print(f"  ▶ Sequence {sequence}/{no_sequences}")

            sequence_data = {
                "metadata": {
                    "video_id": f"{action}_sequence_{sequence}",
                    "fps": 30,
                    "duration_sec": sequence_length / 30,
                    "total_frames": sequence_length,
                    "model": "MediaPipe Holistic"
                },
                "frames": []
            }

            start_time_ms = int(round(time.time() * 1000))

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

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
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting {action} #{sequence}', (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'{action} | Seq {sequence} | Frame {frame_num}', (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop = True
                    break

            # === Simpan ke JSON ===
            json_path = os.path.join(DATA_PATH, action, f'sequence_{sequence}.json')
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ Sequence {sequence} disimpan: {json_path}")

cap.release()
cv2.destroyAllWindows()
print("\n Pengumpulan data selesai!")