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

# --- STEP 2: CUSTOM DRAWING FUNCTION (No mp.solutions) ---
def draw_face_landmarks_manual(image, landmarks):
    """
    Draws dots for face landmarks using standard OpenCV.
    We don't use mp.solutions.drawing_utils here.
    """
    h, w, _ = image.shape
    
    # Loop through all 478 landmarks and draw a tiny dot for each
    for lm in landmarks:
        # Convert normalized coordinates (0.0 to 1.0) to pixels
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Draw a small yellow dot
        cv2.circle(image, (cx, cy), 1, (0, 255, 255), -1)

# --- STEP 3: EMOTION LOGIC (Blendshapes) ---
def get_emotion(blendshapes):
    # Convert list of categories to a dictionary for easier lookup
    scores = {b.category_name: b.score for b in blendshapes}

    # Extract muscle movements
    smile = (scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0)) / 2
    brow_down = (scores.get('browDownLeft', 0) + scores.get('browDownRight', 0)) / 2
    frown = (scores.get('mouthFrownLeft', 0) + scores.get('mouthFrownRight', 0)) / 2
    brow_inner_up = scores.get('browInnerUp', 0)
    sneer = (scores.get('noseSneerLeft', 0) + scores.get('noseSneerRight', 0)) / 2
    eye_blink = (scores.get('eyeBlinkLeft', 0) + scores.get('eyeBlinkRight', 0)) / 2
    jaw_open = scores.get('jawOpen', 0)

    # Determine Emotion
    if sneer > 0.4:
        return "DISGUSTED", (0, 0, 255) # Red
    
    if brow_down > 0.5:
        return "ANGRY", (0, 0, 139) # Dark Blue
    
    if smile > 0.4:
        return "HAPPY", (0, 255, 0) # Green
    
    if frown > 0.4 or (brow_inner_up > 0.4):
        return "SAD", (255, 0, 0) # Blue
    
    if jaw_open > 0.3 and eye_blink < 0.2:
        return "SURPRISED", (255, 255, 0) # Cyan

    if eye_blink > 0.4:
        return "SLEEPY/BORED", (128, 128, 128) # Gray

    return "NEUTRAL", (200, 200, 200) # Light Gray

# --- STEP 4: MAIN EXECUTION ---
def run_face_emotion():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True, # We need this for emotions
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

            # Flip and Convert
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            # Detect
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Visualization
            if result.face_landmarks:
                # 1. Draw Landmarks manually (Yellow Dots)
                # We access index 0 because we set num_faces=1
                draw_face_landmarks_manual(frame, result.face_landmarks[0])

                # 2. Check Emotions
                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    text, color = get_emotion(blendshapes)
                    
                    # Draw UI
                    cv2.rectangle(frame, (0, 0), (350, 60), (0,0,0), -1)
                    cv2.putText(frame, text, (20, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Face Emotion Detector (Tasks API)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_emotion()