import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os
import urllib.request
import math

# --- STEP 1: DOWNLOAD MODEL ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- STEP 2: MATH FUNCTIONS ---
def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_face_signature(landmarks):
    # Width of face (Ear to Ear) used to normalize measurements
    face_width = dist(landmarks[234], landmarks[454])
    if face_width == 0: return None 

    # Key Geometry Ratios
    eye_gap = dist(landmarks[33], landmarks[362]) / face_width
    jaw_len = dist(landmarks[152], landmarks[1]) / face_width
    nose_width = dist(landmarks[102], landmarks[331]) / face_width
    mouth_width = dist(landmarks[61], landmarks[291]) / face_width
    l_eye_w = dist(landmarks[33], landmarks[133]) / face_width
    r_eye_w = dist(landmarks[362], landmarks[263]) / face_width
    forehead = dist(landmarks[10], landmarks[4]) / face_width

    return np.array([eye_gap, jaw_len, nose_width, mouth_width, l_eye_w, r_eye_w, forehead])

# --- STEP 3: SETUP OPTIONS ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

# --- STEP 4: TRAIN (Load Images) ---
landmarker_static = vision.FaceLandmarker.create_from_options(options)
owner_signatures = []

# FIX: Match your actual folder name "Images"
image_folder = "Images" 

# Smart check: If we are already INSIDE the 'Images' folder, look in current directory '.'
if not os.path.exists(image_folder) and os.path.exists("check1.jpg"):
    print("Notice: You are running inside the Images folder.")
    image_folder = "."

print(f"--- TRAINING ON IMAGES IN '{image_folder}' ---")

for i in range(1, 9):
    # Construct path, e.g., "Images/check1.jpg"
    filename = os.path.join(image_folder, f"check{i}.jpg")
    
    if not os.path.exists(filename): 
        print(f"Skipping {filename} (Not found)")
        continue
    
    img = cv2.imread(filename)
    if img is None: continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    try:
        result = landmarker_static.detect(mp_image)
        if result.face_landmarks:
            sig = extract_face_signature(result.face_landmarks[0])
            if sig is not None:
                owner_signatures.append(sig)
                print(f" [OK] Learned face from {filename}")
        else:
            print(f" [X] No face found in {filename}")
    except Exception as e:
        print(f"Error on {filename}: {e}")

if not owner_signatures:
    print("\nCRITICAL ERROR: No valid reference images loaded.")
    print(f"Make sure you are in '~/facerecog' and that '~/facerecog/{image_folder}' exists.")
    exit()

owner_avg_signature = np.mean(owner_signatures, axis=0)
print(f"\nTraining Complete. Average Signature: {np.round(owner_avg_signature, 3)}")

# --- STEP 5: REAL-TIME RECOGNITION ---
def run_recognition():
    options.running_mode = vision.RunningMode.VIDEO
    landmarker_live = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    # 0.05 is strict, 0.08 is loose
    ERROR_THRESHOLD = 0.06

    print("Press 'q' or 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker_live.detect_for_video(mp_image, timestamp_ms)

        status = "SEARCHING..."
        color = (200, 200, 200)
        diff_score = 1.0 

        if result.face_landmarks:
            live_sig = extract_face_signature(result.face_landmarks[0])
            
            if live_sig is not None:
                diff_score = np.mean(np.abs(owner_avg_signature - live_sig))
                
                if diff_score < ERROR_THRESHOLD:
                    status = "ACCESS GRANTED"
                    color = (0, 255, 0) # Green
                else:
                    status = "ACCESS DENIED"
                    color = (0, 0, 255) # Red

            for lm in result.face_landmarks[0]:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (255, 255, 0), -1)

        # UI
        cv2.rectangle(frame, (0,0), (w, 100), (30, 30, 30), -1)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        match_quality = max(0, 1.0 - (diff_score / 0.1))
        
        cv2.putText(frame, f"Match: {int(match_quality*100)}% (Err: {diff_score:.3f})", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        bar_x, bar_y, bar_w = 350, 75, 200
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (100,100,100), -1)
        fill_w = int(bar_w * match_quality)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+20), color, -1)
        
        limit_x = bar_x + int(bar_w * (1.0 - (ERROR_THRESHOLD/0.1)))
        cv2.line(frame, (limit_x, 70), (limit_x, 100), (0,255,255), 2)

        cv2.imshow('Geometric Face Recog', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()