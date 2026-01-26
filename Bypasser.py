import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os
import urllib.request
import math

# --- CONFIGURATION ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- 1. GEOMETRY CORE ---
def get_dense_mesh_signature(landmarks):
    """Normalize face to be scale-invariant and centered."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    nose_tip = coords[1] # Index 1 is nose tip
    centered = coords - nose_tip
    
    # Calculate scale (Face Width)
    face_width = np.linalg.norm(coords[234] - coords[454])
    if face_width == 0: return None, 1.0
    
    return centered / face_width, face_width

def project_ghost_mesh(owner_mesh, live_nose_pos, live_scale):
    """
    Projects the stored 'Owner Mesh' (Normalized) back onto the
    live video coordinates so the user can see the 'Target'.
    """
    # Reverse the normalization: (Mesh * Scale) + Nose_Position
    return (owner_mesh * live_scale) + live_nose_pos

# --- 2. GUIDANCE SYSTEM ---
def get_structural_guidance(owner_mesh, live_mesh):
    """
    Compares specific anatomical points to give instructions.
    """
    instructions = []
    
    # 1. Jaw Check (Index 152 is Chin)
    owner_chin_y = owner_mesh[152][1]
    live_chin_y = live_mesh[152][1]
    diff_chin = live_chin_y - owner_chin_y
    
    if diff_chin < -0.02: instructions.append("LOWER JAW")
    elif diff_chin > 0.02: instructions.append("CLOSE MOUTH")

    # 2. Width Check (Cheekbones 234, 454)
    owner_width = owner_mesh[454][0] - owner_mesh[234][0]
    live_width = live_mesh[454][0] - live_mesh[234][0]
    
    # We can't really change bone width, but we can turn head
    if abs(live_width - owner_width) > 0.05:
        instructions.append("ALIGN ANGLE")

    # 3. Eyebrows (Index 10 is Forehead, 336/296 are brows)
    # Check simple height diff
    if live_mesh[296][1] < owner_mesh[296][1] - 0.01:
        instructions.append("RELAX BROWS")
    elif live_mesh[296][1] > owner_mesh[296][1] + 0.01:
        instructions.append("RAISE BROWS")

    if not instructions:
        return "HOLD STEADY"
    return " + ".join(instructions[:2])

# --- SETUP ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False, 
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

# --- TRAINING (ENROLLMENT) ---
landmarker_static = vision.FaceLandmarker.create_from_options(options)
owner_meshes = []
image_folder = "Images" 
if not os.path.exists(image_folder) and os.path.exists("check1.jpg"): image_folder = "."

print(f"--- LOADING TARGET IMPRINT ({image_folder}) ---")

for i in range(1, 9):
    filename = os.path.join(image_folder, f"check{i}.jpg")
    if not os.path.exists(filename): continue
    
    img = cv2.imread(filename)
    if img is None: continue
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    try:
        result = landmarker_static.detect(mp_image)
        if result.face_landmarks:
            mesh, _ = get_dense_mesh_signature(result.face_landmarks[0])
            if mesh is not None:
                owner_meshes.append(mesh)
                print(f" [OK] Imprint acquired: {filename}")
    except: pass

if not owner_meshes:
    print("FATAL: No reference faces found.")
    exit()

# Create the Master Template
owner_avg_mesh = np.mean(owner_meshes, axis=0)
print(f"Target Imprint Active.")

# --- SCANNING PHASE ---
def run_imprint_scanner():
    options.running_mode = vision.RunningMode.VIDEO
    landmarker_live = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    # 0.045 is standard. We allow 0.055 to give room for "mimicry"
    SYNCH_THRESHOLD = 0.050
    REQUIRED_STREAK = 20
    current_streak = 0

    print("INITIATING IMPRINT SYNC...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        result = landmarker_live.detect_for_video(mp_image, timestamp_ms)

        status = "SEARCHING"
        guidance = ""
        color = (50, 50, 50)
        sync_rate = 0.0
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # 1. Get Live Geometry
            live_mesh, live_scale = get_dense_mesh_signature(landmarks)
            
            # 2. Get Live Position (for projection)
            live_nose_pos = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
            
            if live_mesh is not None:
                # --- CORE LOGIC: COMPARE STRUCTURE ---
                diff = np.mean(np.abs(owner_avg_mesh - live_mesh))
                
                # Convert Diff to Sync Rate (0% to 100%)
                # 0.1 Diff is bad (0%), 0.0 Diff is perfect (100%)
                sync_rate = max(0, min(100, (1.0 - (diff / 0.12)) * 100))
                
                guidance = get_structural_guidance(owner_avg_mesh, live_mesh)

                if diff < SYNCH_THRESHOLD:
                    current_streak += 1
                    status = "SYNCHRONIZING..."
                    color = (255, 255, 0) # Cyan
                    
                    if current_streak >= REQUIRED_STREAK:
                        status = "NEURAL LINK ESTABLISHED"
                        color = (0, 255, 0) # Green
                        guidance = "ACCESS GRANTED"
                        current_streak = REQUIRED_STREAK
                else:
                    status = "STRUCTURE MISMATCH"
                    color = (0, 0, 255) # Red
                    current_streak = max(0, current_streak - 1)

                # --- VISUALS: THE GHOST OVERLAY ---
                # Project the Owner's face onto the User's face coordinates
                ghost_points = project_ghost_mesh(owner_avg_mesh, live_nose_pos, live_scale)
                
                # Draw lines connecting Live Point to Ghost Point (The "Shift" Vector)
                # We only draw key points to avoid clutter
                key_indices = list(range(0, 478, 10)) # Every 10th point
                
                for idx in key_indices:
                    # Live Point
                    lx, ly = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    
                    # Ghost Point (Projected)
                    gx, gy = int(ghost_points[idx][0] * w), int(ghost_points[idx][1] * h)
                    
                    # Draw Ghost Dot
                    cv2.circle(frame, (gx, gy), 1, (200, 200, 200), -1)
                    
                    # Draw Connection Line (Red if far, Green if close)
                    dist_px = math.sqrt((lx-gx)**2 + (ly-gy)**2)
                    line_color = (0, 0, 255) if dist_px > 3 else (0, 255, 0)
                    if dist_px > 2: # Only draw if there is a mismatch
                        cv2.line(frame, (lx, ly), (gx, gy), line_color, 1)

        # --- UI OVERLAY ---
        # 1. Status Box
        cv2.rectangle(frame, (20, 20), (400, 150), (10, 10, 10), -1)
        cv2.rectangle(frame, (20, 20), (400, 150), color, 2)
        
        # 2. Text
        cv2.putText(frame, f"STATUS: {status}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"GUIDANCE: {guidance}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 3. Synch Rate Bar
        cv2.putText(frame, f"SYNC RATE: {int(sync_rate)}%", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_w = int(sync_rate * 3) # Scale to 300px
        cv2.rectangle(frame, (40, 130), (40 + 300, 135), (50,50,50), -1)
        cv2.rectangle(frame, (40, 130), (40 + bar_w, 135), color, -1)

        # 4. Streak Indicator (The "Lock Picking" progress)
        if current_streak > 0:
            streak_w = int((current_streak / REQUIRED_STREAK) * 360)
            cv2.rectangle(frame, (20, 145), (20 + streak_w, 150), (0, 255, 0), -1)

        cv2.imshow('Neural Imprint Scanner', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_imprint_scanner()