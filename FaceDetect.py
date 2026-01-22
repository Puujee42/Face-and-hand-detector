import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os
import urllib.request

# --- STEP 1: DOWNLOAD MODEL ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- STEP 2: CUSTOM DRAWING (Highlight Eyes) ---
def draw_face_landmarks_manual(image, landmarks):
    h, w, _ = image.shape
    
    # MediaPipe Face Mesh Indices for Eyes (Approximate outline)
    # Left Eye: 33, 133, 157, 158, 159, 160, 161, 246...
    # Right Eye: 362, 263, 384, 385, 386, 387, 388, 466...
    # Simple range check: Eyes are mostly indices 33-246 and 362-478
    # We will just color specific ranges to highlight eyes.

    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Color Logic:
        # Iris/Pupils are usually the last landmarks (468-477)
        # Eye contours are generally scattered, but we'll highlight the Irises specifically.
        
        if i >= 468: 
            # IRISES (Cyan - Big Dots)
            cv2.circle(image, (cx, cy), 3, (255, 255, 0), -1)
        else:
            # REST OF FACE (Yellow - Small Dots)
            cv2.circle(image, (cx, cy), 1, (0, 255, 255), -1)

# --- STEP 3: ADVANCED EMOTION LOGIC ---
def get_emotion_with_eyes(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}

    # --- 1. GET BASIC MUSCLE GROUPS ---
    smile = (scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0)) / 2
    brow_down = (scores.get('browDownLeft', 0) + scores.get('browDownRight', 0)) / 2
    brow_inner_up = scores.get('browInnerUp', 0)
    sneer = (scores.get('noseSneerLeft', 0) + scores.get('noseSneerRight', 0)) / 2
    jaw_open = scores.get('jawOpen', 0)
    
    # --- 2. GET EYE SPECIFICS (The New Logic) ---
    # eyeBlink: 0 = Open, 1 = Closed
    eye_blink = (scores.get('eyeBlinkLeft', 0) + scores.get('eyeBlinkRight', 0)) / 2
    
    # eyeWide: 0 = Neutral, 1 = Very Wide (Shock)
    eye_wide = (scores.get('eyeLookUpLeft', 0) + scores.get('eyeLookUpRight', 0)) / 2 + \
               (scores.get('eyeWideLeft', 0) + scores.get('eyeWideRight', 0)) / 2
               
    # eyeSquint: 0 = Neutral, 1 = Squinting (Suspicion/Real Smile)
    eye_squint = (scores.get('eyeSquintLeft', 0) + scores.get('eyeSquintRight', 0)) / 2

    # --- 3. COMPLEX LOGIC CHAIN ---
    
    # PRIORITY 1: EXTREME EYE STATES
    if eye_wide > 0.5 and jaw_open > 0.4:
        return "SHOCKED", (255, 255, 0) # Cyan
    
    if eye_wide > 0.6 and brow_inner_up > 0.4:
        # Wide eyes + Inner brows up = Fear
        return "FEAR / TERRIFIED", (128, 0, 128) # Purple
    
    if eye_squint > 0.6 and smile < 0.2:
        # Squinting without smiling = Suspicion
        return "SUSPICIOUS", (0, 165, 255) # Orange

    # PRIORITY 2: MOUTH & BROWS
    if sneer > 0.4:
        return "DISGUSTED", (0, 0, 255) # Red
    
    if brow_down > 0.5:
        # If brows are down AND eyes are squinting, it's intense anger
        if eye_squint > 0.3:
            return "FURIOUS", (0, 0, 139) # Dark Blue
        return "ANGRY", (0, 0, 255) # Blue
    
    if smile > 0.4:
        # Duchenne Smile Check: Real smiles involve squinting eyes
        if eye_squint > 0.3:
            return "HAPPY (Real Smile)", (0, 255, 0) # Green
        else:
            return "HAPPY (Fake Smile)", (50, 205, 50) # Lime Green
    
    if scores.get('mouthFrownLeft', 0) > 0.4 or brow_inner_up > 0.5:
        return "SAD", (255, 0, 0) # Blue
    
    if eye_blink > 0.4:
        return "SLEEPY", (128, 128, 128) # Gray

    return "NEUTRAL", (200, 200, 200) # Light Gray

# --- STEP 4: MAIN EXECUTION ---
def run_face_emotion():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' or 'ESC' to exit.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                draw_face_landmarks_manual(frame, result.face_landmarks[0])

                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    text, color = get_emotion_with_eyes(blendshapes)
                    
                    # --- UI DISPLAY ---
                    # Background Box
                    cv2.rectangle(frame, (0, 0), (450, 80), (0,0,0), -1)
                    
                    # Main Emotion
                    cv2.putText(frame, text, (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Debug Info: Show Eye Stats in small text
                    scores = {b.category_name: b.score for b in blendshapes}
                    e_wide = (scores.get('eyeWideLeft',0) + scores.get('eyeWideRight',0))/2
                    e_squint = (scores.get('eyeSquintLeft',0) + scores.get('eyeSquintRight',0))/2
                    
                    debug_text = f"Eye Wide: {e_wide:.2f} | Eye Squint: {e_squint:.2f}"
                    cv2.putText(frame, debug_text, (20, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Advanced Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_emotion()