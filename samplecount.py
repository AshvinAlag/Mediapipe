import cv2
import time
import os
import urllib.request
import math
import platform
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
# Pinch Thresholds (Ratios relative to palm size)
PINCH_IN_THRESHOLD = 0.30   # Fist state ratio (Must hit this to prime an OPEN)
PINCH_OUT_THRESHOLD = 1.10  # Open palm state ratio (Must hit this to prime a CLOSE)

# Swipe Thresholds (Optimized for tiny wrist flicks)
SWIPE_THRESHOLD = 0.35      # Cut down significantly so a tiny shift triggers it
SWIPE_WINDOW_SEC = 0.25     # Narrow window to isolate quick, snappy movements
SWIPE_COOLDOWN_SEC = 0.6    # Prevents accidental multi-desktop skipping
# ==============================================================================

# 1. Download BOTH required models if they don't exist
models = {
    'hand_landmarker.task': "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    'face_landmarker.task': "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

for model_name, url in models.items():
    if not os.path.exists(model_name):
        print(f"Downloading {model_name} from Google...")
        urllib.request.urlretrieve(url, model_name)
        print(f"{model_name} downloaded!")

# 2. Setup Landmarker Options
hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=5,  
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    min_tracking_confidence=0.7)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]
FACE_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

def get_distance(p1, p2):
    return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

# --- PACE-DELAYED HOTKEY INJECTORS ---
def trigger_open_view():
    os_name = platform.system()
    if os_name == "Windows":
        pyautogui.hotkey('win', 'tab')
    else:
        pyautogui.keyDown('ctrl')
        time.sleep(0.05)
        pyautogui.press('up')
        time.sleep(0.05)
        pyautogui.keyUp('ctrl')
    print("--> OPENING Multitasking View")

def trigger_close_view():
    pyautogui.press('esc')
    print("<-- CLOSING Multitasking View")

def trigger_desktop_right():
    os_name = platform.system()
    if os_name == "Windows":
        pyautogui.keyDown('ctrl')
        time.sleep(0.02)
        pyautogui.keyDown('win')
        time.sleep(0.02)
        pyautogui.press('right')
        time.sleep(0.02)
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
    else:  # Mac
        pyautogui.keyDown('ctrl')
        time.sleep(0.05)
        pyautogui.press('right')
        time.sleep(0.05)
        pyautogui.keyUp('ctrl')
    print("======>> 4-FINGER FLICK LEFT: Switching Desktop Right")

def trigger_desktop_left():
    os_name = platform.system()
    if os_name == "Windows":
        pyautogui.keyDown('ctrl')
        time.sleep(0.02)
        pyautogui.keyDown('win')
        time.sleep(0.02)
        pyautogui.press('left')
        time.sleep(0.02)
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
    else:  # Mac
        pyautogui.keyDown('ctrl')
        time.sleep(0.05)
        pyautogui.press('left')
        time.sleep(0.05)
        pyautogui.keyUp('ctrl')
    print("<<====== 4-FINGER FLICK RIGHT: Switching Desktop Left")


# --- STATE TRACKING ---
view_is_open = False
ready_to_open = False   
ready_to_close = False  

motion_history = []     
last_swipe_time = 0     

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    current_time_sec = time.time()
    timestamp_ms = int(current_time_sec * 1000)

    hand_results = hand_detector.detect_for_video(mp_image, timestamp_ms)
    face_results = face_detector.detect_for_video(mp_image, timestamp_ms)

    # Face landmarks wireframe
    if face_results.face_landmarks:
        for face_landmarks in face_results.face_landmarks:
            face_lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection.start, connection.end
                if start_idx < len(face_lm_list) and end_idx < len(face_lm_list):
                    cv2.line(image, face_lm_list[start_idx], face_lm_list[end_idx], (255, 255, 0), 1) 

    # Hand Processing with Proximity Filtering
    if hand_results.hand_landmarks:
        valid_hands = []

        # 1. Parse all detected hands and calculate their palm size (proximity)
        for idx in range(len(hand_results.hand_landmarks)):
            landmarks = hand_results.hand_landmarks[idx]
            handedness_data = hand_results.handedness[idx][0]
            
            pixel_lms = []
            for id, lm in enumerate(landmarks):
                pixel_lms.append([id, int(lm.x * w), int(lm.y * h)])
            
            if len(pixel_lms) > 20: # Ensure complete landmark array structure
                palm_sz = get_distance(pixel_lms[0], pixel_lms[9])
                valid_hands.append({
                    'landmarks': pixel_lms,
                    'handedness': 'Left' if handedness_data.category_name == 'Right' else 'Right',
                    'palm_size': palm_sz
                })

        # 2. Sort hands by palm size descending (Largest size = closest to camera)
        valid_hands = sorted(valid_hands, key=lambda x: x['palm_size'], reverse=True)
        closest_hands = valid_hands[:2]

        # 3. Process only our filtered closest hands
        for hand in closest_hands:
            lm_list = hand['landmarks']
            handedness = hand['handedness']
            palm_distance = max(1, hand['palm_size'])

            # Render landmarks and connections
            for id, cx, cy in lm_list:
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(lm_list) and end_idx < len(lm_list):
                    cv2.line(image, (lm_list[start_idx][1], lm_list[start_idx][2]), 
                             (lm_list[end_idx][1], lm_list[end_idx][2]), (0, 0, 255), 2)

            wrist_x = lm_list[0][1]
            raw_pinch_distance = get_distance(lm_list[4], lm_list[8])
            pinch_ratio = raw_pinch_distance / palm_distance
            
            cv2.line(image, (lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]), (0, 255, 255), 2)

            # --------------------------------------------------------------
            # A. PINCH DETECTOR
            # --------------------------------------------------------------
            if pinch_ratio <= PINCH_IN_THRESHOLD:
                ready_to_open = True
                if view_is_open and ready_to_close:
                    trigger_close_view()
                    view_is_open = False
                    ready_to_close = False

            elif pinch_ratio >= PINCH_OUT_THRESHOLD:
                ready_to_close = True
                if not view_is_open and ready_to_open:
                    trigger_open_view()
                    view_is_open = True
                    ready_to_open = False

            # --------------------------------------------------------------
            # B. 4-FINGER ONLY SWIPE DETECTOR
            # --------------------------------------------------------------
            # Check which fingers are extended (Tip Y-value is less than Knuckle Y-value)
            fingers_extended = []
            
            # 4 Main fingers tracking: Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
            for tip, knuckle in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                if lm_list[tip][2] < lm_list[knuckle][2]:
                    fingers_extended.append(True)
                else:
                    fingers_extended.append(False)
            
            # Thumb tracking: Check horizontal span separation relative to index base (4 vs 5)
            # Depending on handedness, thumb extension direction changes
            thumb_is_extended = False
            if handedness == 'Right':
                thumb_is_extended = lm_list[4][1] < lm_list[5][1]
            else:
                thumb_is_extended = lm_list[4][1] > lm_list[5][1]

            # Condition: Exactly 4 fingers up AND the thumb must be down/tucked
            four_fingers_exactly = (sum(fingers_extended) == 4) and not thumb_is_extended

            if four_fingers_exactly:
                # Add current position data points into rolling tracker cache
                motion_history.append({'time': current_time_sec, 'x': wrist_x})
            else:
                # Instantly drop data history tracking if hand posture breaks composition
                motion_history.clear()
                
            # Trim history cache strictly to our configured window profile
            motion_history = [pt for pt in motion_history if current_time_sec - pt['time'] <= SWIPE_WINDOW_SEC]

            if (current_time_sec - last_swipe_time) > SWIPE_COOLDOWN_SEC and len(motion_history) > 3:
                min_entry = min(motion_history, key=lambda k: k['x'])
                max_entry = max(motion_history, key=lambda k: k['x'])
                
                total_span = max_entry['x'] - min_entry['x']
                normalized_span = total_span / palm_distance

                if normalized_span >= SWIPE_THRESHOLD:
                    if max_entry['time'] < min_entry['time']:
                        trigger_desktop_right()
                        last_swipe_time = current_time_sec
                        motion_history.clear()
                    elif min_entry['time'] < max_entry['time']:
                        trigger_desktop_left()
                        last_swipe_time = current_time_sec
                        motion_history.clear()

            # Text Feedback UI Layout Elements
            wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]
            text_x, text_y = max(10, wrist_x - 50), max(30, wrist_y + 40)

            finger_count = sum(fingers_extended) + (1 if thumb_is_extended else 0)
            cv2.putText(image, f'{handedness} | Fingers: {finger_count}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Pinch Ratio: {pinch_ratio:.2f}', (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            status_txt = "Ready to Open" if ready_to_open else ("Ready to Close" if ready_to_close else "Neutral")
            cv2.putText(image, f'Status: {status_txt}', (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    else:
        ready_to_close = False
        ready_to_open = False
        motion_history.clear()

    cv2.imshow('MediaPipe Tracking', image)
    
    key = cv2.waitKeyEx(5)
    if key in [45, 0x2D, 0x2d0000]: # Insert key kills program
        print("Closing Python script via Insert key...")
        break

cap.release()
cv2.destroyAllWindows()