import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os
import urllib.request

# --- STEP 1: DOWNLOAD MODEL ---
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- STEP 2: HELPER FUNCTIONS ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm connections
]

def to_pixel(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def draw_hand_landmarks_tasks_only(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    for hand_landmarks in hand_landmarks_list:
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]
        
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        
        for (x, y) in pts:
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

    return annotated

# --- STEP 3: GESTURE LOGIC ---
def get_gesture(landmarks):
    tips = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    wrist = landmarks[0]
    
    fingers_open = []
    for i in range(5):
        tip = landmarks[tips[i]]
        joint = landmarks[joints[i]]
        # Check distance to wrist
        dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        dist_joint = (joint.x - wrist.x)**2 + (joint.y - wrist.y)**2
        fingers_open.append(1 if dist_tip > dist_joint else 0)

    count = sum(fingers_open[1:]) # Sum index, middle, ring, pinky

    # Gesture Rules
    if count == 0: return "ROCK"
    if count == 4: return "PAPER"
    if fingers_open[1] == 1 and fingers_open[2] == 1 and fingers_open[3] == 0 and fingers_open[4] == 0:
        return "SCISSORS"
    
    return "Unknown"

def determine_winner(p1, p2):
    if p1 == p2: return "DRAW"
    if (p1 == "ROCK" and p2 == "SCISSORS") or \
       (p1 == "SCISSORS" and p2 == "PAPER") or \
       (p1 == "PAPER" and p2 == "ROCK"):
        return "P1_WINS"
    return "P2_WINS"

# --- STEP 4: MAIN GAME LOOP ---
def run_game():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)
    
    # Game Variables
    p1_score = 0
    p2_score = 0
    
    # State Machine: "WAITING" -> "COUNTDOWN" -> "RESULT" -> "COOLDOWN"
    game_state = "WAITING"
    timer_start = 0
    result_text = ""
    winner_color = (255, 255, 255)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # MediaPipe Processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw Hands if detected
            if result.hand_landmarks:
                frame = draw_hand_landmarks_tasks_only(frame, result.hand_landmarks)

            # --- GAME LOGIC START ---
            
            # 1. WAITING STATE: Look for 2 hands to start
            if game_state == "WAITING":
                cv2.putText(frame, "SHOW 2 HANDS TO START", (w//2 - 200, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    game_state = "COUNTDOWN"
                    timer_start = time.time()

            # 2. COUNTDOWN STATE: 3.. 2.. 1..
            elif game_state == "COUNTDOWN":
                elapsed = time.time() - timer_start
                if elapsed < 1:
                    msg = "ROCK..."
                elif elapsed < 2:
                    msg = "PAPER..."
                elif elapsed < 3:
                    msg = "SCISSORS..."
                else:
                    # Time is up! Capture the result now
                    game_state = "RESULT"
                    msg = "SHOOT!"
                
                cv2.putText(frame, msg, (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
                
                # If hands disappear during countdown, reset
                if not result.hand_landmarks or len(result.hand_landmarks) < 2:
                    game_state = "WAITING"

            # 3. RESULT STATE: Calculate Winner Once
            elif game_state == "RESULT":
                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    # Sort hands
                    h1, h2 = result.hand_landmarks[0], result.hand_landmarks[1]
                    if h1[0].x < h2[0].x:
                        p1_hand, p2_hand = h1, h2
                    else:
                        p1_hand, p2_hand = h2, h1

                    g1 = get_gesture(p1_hand)
                    g2 = get_gesture(p2_hand)
                    
                    outcome = determine_winner(g1, g2)
                    
                    if outcome == "P1_WINS":
                        p1_score += 1
                        result_text = f"P1 WINS! ({g1} beats {g2})"
                        winner_color = (0, 255, 0)
                    elif outcome == "P2_WINS":
                        p2_score += 1
                        result_text = f"P2 WINS! ({g2} beats {g1})"
                        winner_color = (0, 0, 255)
                    else:
                        result_text = "DRAW!"
                        winner_color = (200, 200, 200)

                    game_state = "COOLDOWN"
                    timer_start = time.time()
                else:
                    game_state = "WAITING"

            # 4. COOLDOWN STATE: Show result for 3 seconds
            elif game_state == "COOLDOWN":
                # Show the result text calculated in previous step
                text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                cv2.putText(frame, result_text, ((w - text_size[0])//2, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, winner_color, 3)

                if time.time() - timer_start > 3:
                    game_state = "WAITING"

            # --- ALWAYS SHOW SCORE ---
            cv2.rectangle(frame, (0,0), (w, 60), (50, 50, 50), -1)
            cv2.putText(frame, f"P1: {p1_score}", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"P2: {p2_score}", (w - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('RPS Battle', frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()