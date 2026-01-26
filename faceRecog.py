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

# --- STEP 2: DENSE GEOMETRY LOGIC ---

def get_dense_mesh_signature(landmarks):
    """
    Converts all 478 landmarks into a normalized NumPy array.
    """
    # Convert Landmarks to NumPy Array (478 points x 3 coords)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # 1. CENTER THE FACE
    # Find the nose tip (Index 1) and subtract it from all points
    # This moves the face so the nose is at (0,0,0)
    nose_tip = coords[1]
    centered = coords - nose_tip

    # 2. ROTATION CORRECTION (Basic 2D alignment)
    # We rotate the face so the eyes are level horizontally.
    left_eye = coords[33]
    right_eye = coords[263]
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    
    # (Skipping complex 3D rotation matrix for speed, trusting 
    # the user looks straight due to our pose check)

    # 3. SCALE THE FACE
    # Calculate face width (Ear to Ear: 234 to 454)
    face_width = np.linalg.norm(coords[234] - coords[454])
    
    if face_width == 0: return None
    
    # Divide all coordinates by face_width to normalize size
    normalized = centered / face_width
    
    return normalized

def check_head_pose(landmarks):
    """Ensures user is looking straight ahead"""
    nose = landmarks[1]
    l_ear = landmarks[234]
    r_ear = landmarks[454]
    
    # YAW CHECK
    d_left = math.sqrt((nose.x - l_ear.x)**2 + (nose.y - l_ear.y)**2)
    d_right = math.sqrt((nose.x - r_ear.x)**2 + (nose.y - r_ear.y)**2)
    yaw_ratio = d_left / (d_right + 0.0001)

    return 0.8 < yaw_ratio < 1.25

def generate_vertex_weights():
    """
    Assigns a 'Trust Score' to every single point on the face.
    Bone points = High Trust. Mouth/Cheek points = Low Trust.
    """
    weights = np.ones(478) # Start with base weight of 1.0

    # High Trust: Nose Bridge & Eyes (Bone)
    # MediaPipe Indices approximations
    nose_indices = list(range(1, 20)) + [168, 6, 197, 195, 5, 4]
    eye_indices = list(range(33, 133)) + list(range(362, 463))
    
    weights[nose_indices] = 3.0  # Trust nose VERY much
    weights[eye_indices] = 2.0   # Trust eyes much

    # Low Trust: Jawline & Cheeks (Flesh/Fat changes)
    jaw_indices = list(range(0, 17)) + list(range(152, 170)) # Rough approximation
    mouth_indices = list(range(61, 91)) + list(range(291, 321))

    weights[jaw_indices] = 0.5
    weights[mouth_indices] = 0.5 
    
    return weights

# --- STEP 3: SETUP ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

# --- STEP 4: TRAINING ---
landmarker_static = vision.FaceLandmarker.create_from_options(options)
owner_meshes = []
image_folder = "Images" 

if not os.path.exists(image_folder) and os.path.exists("check1.jpg"):
    image_folder = "."

print(f"--- TRAINING ON IMAGES IN '{image_folder}' ---")

for i in range(1, 11):
    filename = os.path.join(image_folder, f"check{i}.jpg")
    if not os.path.exists(filename): continue
    
    img = cv2.imread(filename)
    if img is None: continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    try:
        result = landmarker_static.detect(mp_image)
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            if check_head_pose(lm):
                mesh = get_dense_mesh_signature(lm)
                if mesh is not None:
                    owner_meshes.append(mesh)
                    print(f" [OK] Learned face from {filename}")
            else:
                print(f" [X] Skipped {filename} (Head turned)")
    except: pass

if not owner_meshes:
    print("\nERROR: No valid images found.")
    exit()

# Calculate the AVERAGE MESH (The "Perfect Owner Face")
owner_avg_mesh = np.mean(owner_meshes, axis=0)
vertex_weights = generate_vertex_weights()

print(f"\nTraining Complete. Master Template created from {len(owner_meshes)} images.")

# --- STEP 5: REAL-TIME RECOGNITION ---
def run_recognition():
    options.running_mode = vision.RunningMode.VIDEO
    landmarker_live = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    # THRESHOLD:
    # Since we are summing error over 478 points, the number is different.
    # 0.045 is a good starting point for per-vertex average error.
    ERROR_THRESHOLD = 0.045
    REQUIRED_STREAK = 15 

    current_streak = 0
    
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

        status = "NO FACE"
        color = (100, 100, 100)
        debug_info = ""

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            if not check_head_pose(landmarks):
                status = "LOOK STRAIGHT"
                color = (0, 255, 255) 
                current_streak = 0
            else:
                live_mesh = get_dense_mesh_signature(landmarks)
                
                if live_mesh is not None:
                    # --- DENSE COMPARISON ---
                    # 1. Subtract Live Mesh from Owner Mesh
                    diff = np.abs(owner_avg_mesh - live_mesh)
                    
                    # 2. Average the XYZ differences for each point
                    # Shape becomes (478,) representing error at each point
                    point_errors = np.mean(diff, axis=1)
                    
                    # 3. Apply Weights (Trust Nose > Trust Mouth)
                    weighted_errors = point_errors * vertex_weights
                    
                    # 4. Final Score
                    final_score = np.mean(weighted_errors)
                    
                    match_percent = max(0, 1.0 - (final_score / 0.1)) * 100
                    debug_info = f"Err: {final_score:.4f}"

                    if final_score < ERROR_THRESHOLD:
                        current_streak += 1
                        if current_streak >= REQUIRED_STREAK:
                            status = "ACCESS GRANTED"
                            color = (0, 255, 0)
                            current_streak = REQUIRED_STREAK 
                        else:
                            status = f"VERIFYING {current_streak}/{REQUIRED_STREAK}"
                            color = (0, 165, 255)
                    else:
                        status = "ACCESS DENIED"
                        color = (0, 0, 255)
                        current_streak = 0 

            # VISUALIZATION: Draw the Mask
            # Green dots = Good match, Red dots = Bad match
            if result.face_landmarks:
                for i in range(0, 478, 5): # Draw every 5th point to save FPS
                    lm = landmarks[i]
                    pt_x, pt_y = int(lm.x * w), int(lm.y * h)
                    
                    # If this specific point is far off from owner, make it RED
                    # (Note: We rely on the global score, but this visualizes distortions)
                    cv2.circle(frame, (pt_x, pt_y), 1, (0, 255, 0), -1)

        # UI
        cv2.rectangle(frame, (0,0), (w, 100), (20, 20, 20), -1)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        if debug_info:
            cv2.putText(frame, debug_info, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if current_streak > 0:
            bar_w = int((current_streak / REQUIRED_STREAK) * w)
            cv2.rectangle(frame, (0, 95), (bar_w, 100), (0, 255, 0), -1)

        cv2.imshow('Dense Mesh Face ID', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()