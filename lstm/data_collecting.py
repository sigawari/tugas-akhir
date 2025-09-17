import cv2
import numpy as np
import os
import mediapipe as mp
import json
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Mapping nama landmark pose (MediaPipe Pose 33 titik, contoh sebagian)
pose_landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

# Mapping landmark face (468 titik) ke nama standar mediaPipe FaceMesh - versi singkat contoh;
# Bisa buat dictionary sesuai indeks jika diperlukan, atau tulis "face_landmark_0", dst.
def face_landmark_name(idx):
    # Untuk contoh bisa return nama generik
    return f"face_landmark_{idx}"

# Mapping tangan kiri dan kanan (21 titik)
hand_landmark_names = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

def extract_keypoints_dict(results):
    data = {}

    # Pose landmarks
    if results.pose_landmarks:
        data["pose"] = {
            pose_landmark_names[i]: {
                "x": res.x, "y": res.y, "z": res.z, "visibility": res.visibility
            }
            for i, res in enumerate(results.pose_landmarks.landmark)
        }
    else:
        data["pose"] = {}

    # Face landmarks
    if results.face_landmarks:
        data["face"] = {
            face_landmark_name(i): {
                "x": res.x, "y": res.y, "z": res.z
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


no_sequences = 30           # Number of sequences per action
sequence_length = 30        # Number of frames per sequence
DATA_PATH = 'data_json'       

cap = cv2.VideoCapture(0)   # Open webcam

# Fungsi untuk melakukan deteksi dengan Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                   # untuk meningkatkan performa
    results = model.process(image)                  # deteksi
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

# Fungsi untuk menggambar landmark dengan style
def draw_styled_landmarks(image, results):
    # Pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    stop = False

    actions = ['halo', 'terima_kasih'] # Kumpulin dua data dulu
    
    for action in actions:
        if stop:
            break
        print(f"Memulai pengumpulan data untuk action: {action}")
        all_sequences = []
        
        for sequence in range(1, no_sequences + 1):
            if stop:
                break
            sequence_data = {
                "metadata": {
                    "video_id": f"{action}_sequence_{sequence}",
                    "fps": 30,
                    "duration_sec": sequence_length / 30,
                    "total_frames": sequence_length,
                    "total_landmarks": 33 + 468 + 21 + 21,  # total titik pose + face + 2 tangan
                    "landmark_model": "MediaPipe Holistic"
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

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop = True
                    break

            # Simpan ke JSON
            json_seq_path = os.path.join(DATA_PATH, action, f'sequence_{sequence}.json')
            os.makedirs(os.path.dirname(json_seq_path), exist_ok=True) 
            with open(json_seq_path, 'w') as f:
                json.dump(sequence_data, f, indent=2)
            print(f"Sequence {sequence} saved as JSON at {json_seq_path}")

            all_sequences.append(sequence_data)

cap.release()
cv2.destroyAllWindows()
