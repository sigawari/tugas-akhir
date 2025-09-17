import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe holistic setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                    
    results = model.process(image)                  
    image.flags.writeable = True                     
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Setup basic variables
actions = ['halo', 'terima_kasih']
no_sequences = 30
sequence_length = 30
DATA_PATH = 'MP_Data'

# Pastikan folder data ada
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.mkdir(action_path)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    all_data = {}  # Menyimpan semua data untuk tiap action
    
    for action in actions:
        print(f"Memulai pengumpulan data untuk action: {action}")
        all_sequences = []  # Menyimpan semua sequence untuk action ini
        
        for sequence in range(1, no_sequences + 1):
            sequence_data = []  # Menyimpan seluruh frame keypoints untuk 1 video sequence
            
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
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
                
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)  # Koleksi frame keypoints
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            # Simpan data per sequence (video) ke file terpisah
            npy_seq_path = os.path.join(DATA_PATH, action, f"sequence_{sequence}.npy")
            np.save(npy_seq_path, np.array(sequence_data))
            print(f"Sequence {sequence} saved at {npy_seq_path}")
            
            all_sequences.append(sequence_data)  
        
        # Menggabungkan sequence dalam 1 file
        all_data[action] = np.array(all_sequences)
        combined_npy_path = os.path.join(DATA_PATH, f"{action}_combined.npy")
        np.save(combined_npy_path, all_data[action])
        print(f"Combined data for action '{action}' saved at {combined_npy_path}")

cap.release()
cv2.destroyAllWindows()