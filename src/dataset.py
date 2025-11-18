# dataset.py
# PyTorch Dataset & DataLoader wrapper for NPY dataset.
# Loads X/y arrays → returns tensor per sample.
import os
import json
import numpy as np
from collections import defaultdict
import mediapipe as mp

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


# === Analisis Dataset JSON ===
def count_dataset_stats():
    """Count and analyze the JSON dataset"""
    print("🔍 ANALISIS DATASET JSON")
    print("=" * 50)
    
    total_files = 0
    total_frames = 0
    stats_by_action = defaultdict(lambda: {
        'sequences': 0,
        'total_frames': 0,
        'avg_frames_per_seq': 0,
        'files': []
    })
    
    # Check if data_json folder exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Folder {DATA_PATH} tidak ditemukan!")
        return
    
    # Scan each action folder
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        if not os.path.exists(action_path):
            print(f"⚠️  Folder untuk action '{action}' tidak ditemukan di {action_path}")
            continue
            
        print(f"\n📁 Action: {action}")
        print("-" * 30)
        
        # Count JSON files in action folder
        json_files = [f for f in os.listdir(action_path) if f.endswith('.json')]
        json_files.sort()  # Sort by name
        
        action_frames = 0
        valid_sequences = 0
        
        for json_file in json_files:
            json_path = os.path.join(action_path, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count frames in this sequence
                frames_count = len(data.get('frames', []))
                action_frames += frames_count
                valid_sequences += 1
                
                stats_by_action[action]['files'].append({
                    'filename': json_file,
                    'frames': frames_count,
                    'video_id': data.get('metadata', {}).get('video_id', 'N/A')
                })
                
                print(f"  ✅ {json_file}: {frames_count} frames")
                
            except Exception as e:
                print(f"  ❌ Error membaca {json_file}: {e}")
        
        # Update stats
        stats_by_action[action]['sequences'] = valid_sequences
        stats_by_action[action]['total_frames'] = action_frames
        stats_by_action[action]['avg_frames_per_seq'] = (
            action_frames / valid_sequences if valid_sequences > 0 else 0
        )
        
        total_files += valid_sequences
        total_frames += action_frames
        
        print(f"  📊 Total sequences: {valid_sequences}")
        print(f"  📊 Total frames: {action_frames}")
        if valid_sequences > 0:
            print(f"  📊 Rata-rata frames per sequence: {action_frames/valid_sequences:.1f}")
        else:
            print("  📊 Rata-rata frames per sequence: 0")
    
    # Overall summary
    print(f"\n🎯 RINGKASAN KESELURUHAN")
    print("=" * 50)
    print(f"Total Actions: {len([a for a in actions if stats_by_action[a]['sequences'] > 0])}")
    print(f"Total JSON Files: {total_files}")
    print(f"Total Frames: {total_frames}")
    print(f"Expected Files: {len(actions) * no_sequences}")
    print(f"Expected Frames: {len(actions) * no_sequences * sequence_length}")
    if no_sequences > 0:
        print(f"Completion: {(total_files / (len(actions) * no_sequences) * 100):.1f}%")
    print("=" * 50)


def read_sample_json(action_name, sequence_num=1):
    """Read and display sample JSON content"""
    print(f"\n📖 CONTOH DATA JSON - {action_name.upper()} SEQUENCE {sequence_num}")
    print("=" * 60)
    
    json_path = os.path.join(DATA_PATH, action_name, f'sequence_{sequence_num}.json')
    
    if not os.path.exists(json_path):
        print(f"❌ File tidak ditemukan: {json_path}")
        return
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Display metadata
        metadata = data.get('metadata', {})
        print("📋 METADATA:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Display frame structure
        frames = data.get('frames', [])
        print(f"\n🎬 FRAMES: {len(frames)} total")
        
        if frames:
            # Show first frame structure
            first_frame = frames[0]
            print(f"\n📸 Frame 0 Structure:")
            print(f"  frame_index: {first_frame.get('frame_index')}")
            print(f"  timestamp_ms: {first_frame.get('timestamp_ms')}")
            
            landmarks = first_frame.get('landmarks', {})
            print(f"\n🎯 Landmarks Summary:")
            
            # Count landmarks by type
            pose_count = len(landmarks.get('pose', {}))
            face_count = len(landmarks.get('face', {}))
            left_hand_count = len(landmarks.get('left_hand', {}))
            right_hand_count = len(landmarks.get('right_hand', {}))
            
            print(f"  Pose landmarks: {pose_count}")
            print(f"  Face landmarks: {face_count}")
            print(f"  Left hand landmarks: {left_hand_count}")
            print(f"  Right hand landmarks: {right_hand_count}")
            
            # Show sample pose landmarks
            if landmarks.get('pose'):
                print(f"\n👤 Sample Pose Landmarks:")
                for i, (name, coords) in enumerate(list(landmarks['pose'].items())[:5]):
                    print(f"  {name}: x={coords['x']:.3f}, y={coords['y']:.3f}, z={coords['z']:.3f}")
                if len(landmarks['pose']) > 5:
                    print(f"  ... dan {len(landmarks['pose']) - 5} landmarks lainnya")
            
            # Show sample hand landmarks if available
            if landmarks.get('left_hand'):
                print(f"\n👈 Sample Left Hand Landmarks:")
                for i, (name, coords) in enumerate(list(landmarks['left_hand'].items())[:3]):
                    print(f"  {name}: x={coords['x']:.3f}, y={coords['y']:.3f}, z={coords['z']:.3f}")
    
    except Exception as e:
        print(f"❌ Error membaca file: {e}")


# === Main Runner ===
def main():
    """Main function to run data analysis"""
    count_dataset_stats()
    
    if 'halo' in actions:
        read_sample_json('halo', 1)

if __name__ == "__main__":
    main()
