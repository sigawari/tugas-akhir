# data_collect.py
# Capture raw Mediapipe landmarks + optional video recording.
# Output (kalau PREVIEW_ONLY = False dan sequence valid):
#   data/raw/<kata>/data_json/sequence_X.json
#   data/raw/<kata>/data_video/sequence_X.mp4

import cv2
import mediapipe as mp
import json
import time
from pathlib import Path
import os

# KONFIGURASI GLOBAL

DEFAULT_NO_SEQUENCES = 30       
SEQUENCE_LENGTH = 30             
SAVE_VIDEO = True                
PREVIEW_ONLY = False              # Preview data
MIN_HAND_FRAMES = 3             

# SETUP PATH PROJECT

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
RAW_DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Root dataset: {RAW_DATA_ROOT}")
print("Struktur: data/raw/<kata>/{data_json,data_video}/sequence_X.*")
print(f"Mode: {'PREVIEW ONLY (tidak menyimpan file)' if PREVIEW_ONLY else 'RECORD (menyimpan JSON + video jika valid)'}")

# MEDIAPIPE SETUP===

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

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

hand_landmark_names = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# UTIL FUNCTION=====

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
            str(i): {
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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

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

def get_next_sequence_index(json_dir: Path):
    if not json_dir.exists():
        return 1
    existing = [f for f in json_dir.iterdir()
                if f.name.startswith("sequence_") and f.suffix == ".json"]
    indices = []
    for f in existing:
        try:
            num_str = f.stem.split("_")[1]  # "sequence_12" -> "12"
            indices.append(int(num_str))
        except Exception:
            continue
    return (max(indices) + 1) if indices else 1

# MAIN LOOP========

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    stop_all = False

    while not stop_all:
        action = input("\nMasukkan kata (atau ketik 'q' untuk selesai): ").strip()
        if action.lower() == 'q':
            break

        if not action:
            print("Kata tidak boleh kosong.")
            continue

        # Folder: data/raw/<kata>/data_json dan data/raw/<kata>/data_video
        session_dir = RAW_DATA_ROOT / action
        data_json_dir = session_dir / "data_json"
        data_video_dir = session_dir / "data_video"

        data_json_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_VIDEO and not PREVIEW_ONLY:
            data_video_dir.mkdir(parents=True, exist_ok=True)

        start_seq = get_next_sequence_index(data_json_dir)

        no_seq_input = input(
            f"Berapa sequence untuk kata '{action}'? (default {DEFAULT_NO_SEQUENCES}): "
        ).strip()
        try:
            no_sequences = int(no_seq_input) if no_seq_input else DEFAULT_NO_SEQUENCES
        except ValueError:
            print("Input tidak valid, pakai default.")
            no_sequences = DEFAULT_NO_SEQUENCES

        end_seq = start_seq + no_sequences - 1

        print(f"\n🔹 Mengumpulkan data untuk kata: {action}")
        print(f"   Sequence: {start_seq} s/d {end_seq}")
        print(f"   Folder JSON : {data_json_dir}")
        if SAVE_VIDEO and not PREVIEW_ONLY:
            print(f"   Folder Video: {data_video_dir}")
        print(f"   Minimal frame dengan tangan: {MIN_HAND_FRAMES}")

        for sequence in range(start_seq, end_seq + 1):
            print(f"\n  ▶ Sequence {sequence} ({sequence - start_seq + 1}/{no_sequences})")

            sequence_data = {
                "metadata": {
                    "video_id": f"{action}_sequence_{sequence}",
                    "fps": 30,
                    "duration_sec": SEQUENCE_LENGTH / 30,
                    "total_frames": SEQUENCE_LENGTH,
                    "model": "MediaPipe Holistic",
                    "action": action,
                },
                "frames": []
            }

            start_time_ms = int(round(time.time() * 1000))

            out = None
            video_filename = None
            hand_frames_count = 0  # berapa frame yang punya tangan (left/right)

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari kamera.")
                    stop_all = True
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Cek apa yang terdeteksi di frame ini
                has_pose = results.pose_landmarks is not None
                has_face = results.face_landmarks is not None
                has_left = results.left_hand_landmarks is not None
                has_right = results.right_hand_landmarks is not None
                has_any_hand = has_left or has_right
                if has_any_hand:
                    hand_frames_count += 1

                timestamp_ms = start_time_ms + int((frame_num / 30) * 1000)
                landmarks = extract_keypoints_dict(results)

                frame_data = {
                    "frame_index": frame_num,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": landmarks
                }
                sequence_data["frames"].append(frame_data)

                # UI teks status deteksi
                status_text = f"P:{'Y' if has_pose else 'N'} " \
                              f"F:{'Y' if has_face else 'N'} " \
                              f"LH:{'Y' if has_left else 'N'} " \
                              f"RH:{'Y' if has_right else 'N'}"

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (60, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                cv2.putText(image, f'{action} | Seq {sequence} | Frame {frame_num}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Inisialisasi VideoWriter hanya jika:
                # - bukan preview
                # - SAVE_VIDEO True
                # - dan ini frame pertama yang mau direkam
                if not PREVIEW_ONLY and SAVE_VIDEO and out is None:
                    video_filename = data_video_dir / f"sequence_{sequence}.mp4"
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(video_filename), fourcc, 30, (w, h))

                cv2.imshow('OpenCV Feed', image)

                if not PREVIEW_ONLY and SAVE_VIDEO and out is not None:
                    out.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop_all = True
                    break

            # beres satu sequence
            if SAVE_VIDEO and out is not None:
                out.release()

            if stop_all:
                break

            # LOGIKA VALIDASI SEQUENCE

            if hand_frames_count < MIN_HAND_FRAMES:
                print(f"  ⚠️ Sequence {sequence} DIBATALKAN: "
                      f"tangan terdeteksi hanya di {hand_frames_count} frame "
                      f"(min {MIN_HAND_FRAMES}).")

                # Kalau sudah terlanjur buat video file, hapus
                if not PREVIEW_ONLY and SAVE_VIDEO and video_filename is not None and Path(video_filename).exists():
                    try:
                        os.remove(video_filename)
                        print(f"  🗑️ Video dibuang: {video_filename}")
                    except Exception as e:
                        print(f"  ⚠️ Gagal menghapus video: {e}")

                # Jangan simpan JSON kalau under-threshold
                continue

            # Kalau mode preview saja → tidak simpan apa pun
            if PREVIEW_ONLY:
                print(f"  👀 PREVIEW ONLY: Sequence {sequence} tidak disimpan.")
                print(f"     Frame dengan tangan terdeteksi: {hand_frames_count}/{SEQUENCE_LENGTH}")
                continue

            # Mode rekam dan sequence valid → simpan JSON
            json_path = data_json_dir / f'sequence_{sequence}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ JSON  disimpan: {json_path}")
            if SAVE_VIDEO and video_filename is not None:
                print(f"  🎥 Video disimpan: {video_filename}")
            print(f"  👍 Frame dengan tangan terdeteksi: {hand_frames_count}/{SEQUENCE_LENGTH}")

cap.release()
cv2.destroyAllWindows()
print("\nPengumpulan data selesai!")
print(f"Semua data (kalau direkam) ada di: {RAW_DATA_ROOT}")
