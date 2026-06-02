import cv2
import time
import os
import urllib.request
import math
import platform
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
# Left Hand Pinch Thresholds (Ratios relative to palm size)
LEFT_PINCH_IN_THRESHOLD = 0.30   # Fist/Pinch state ratio (Must hit this to trigger CLICK)
LEFT_PINCH_OUT_THRESHOLD = 0.60  # Open palm state ratio (Must hit this to prime CLICK)

# Right Hand Pinch Thresholds (Ratios relative to palm size)
RIGHT_PINCH_OUT_THRESHOLD = 1.10  # Open palm state ratio (Must hit this to trigger OPEN)

# Head Tilt Keyboard/Uppercase Thresholds (in degrees)
TILT_RIGHT_TRIGGER = 15.0  # Trigger angle for tilting head right (Opens Keyboard)
TILT_LEFT_TRIGGER = -12.0  # Trigger angle for tilting head left (Activates Uppercase)
TILT_NEUTRAL_LIMIT = 5.0   # Return to neutral angle to prime again

# Head Flick Virtual Desktop Thresholds
FLICK_THRESHOLD = 0.07     # Horizontal normalized distance traveled by nose in 150ms
FLICK_COOLDOWN_SEC = 1.0   # Time in seconds to wait before allowing another flick trigger

# Head Pitch (Vertical Scroll) Thresholds
PITCH_LOOK_UP_THRESHOLD = 0.28   # Pitch ratio below this scrolls UP
PITCH_LOOK_DOWN_THRESHOLD = 0.52 # Pitch ratio above this scrolls DOWN

# Volume Control Thresholds
VOLUME_STEP_PIXELS = 4     # Vertical pixels of fist movement per volume step (very sensitive)

# Zoom Control Thresholds
ZOOM_STEP_RATIO = 0.04     # Pinch ratio change per zoom step (high sensitivity)

# PyAutoGUI Configuration
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0  # Set delay after pyautogui actions to 0 for maximum speed

# Cursor Mapping Active Box (normalized coordinates [0.0 - 1.0] of camera frame)
ACTIVE_BOX_X_MIN = 0.25
ACTIVE_BOX_X_MAX = 0.75
ACTIVE_BOX_Y_MIN = 0.25
ACTIVE_BOX_Y_MAX = 0.75

# Cursor Smoothing (Exponential Moving Average factor: 0.0 - 1.0)
SMOOTHING_FACTOR = 0.25
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

# ==============================================================================
#                      $1 UNISTROKE GESTURE RECOGNIZER
# ==============================================================================
def path_length(points):
    d = 0.0
    for i in range(1, len(points)):
        d += math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
    return d

def resample(points, n):
    if len(points) == 0:
        return []
    length = path_length(points)
    if length == 0:
        return [points[0]] * n
    I = length / (n - 1)
    D = 0.0
    new_points = [points[0]]
    pts = list(points)
    i = 1
    while i < len(pts):
        d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        if D + d >= I:
            if d == 0:
                d = 0.001
            qx = pts[i-1][0] + ((I - D) / d) * (pts[i][0] - pts[i-1][0])
            qy = pts[i-1][1] + ((I - D) / d) * (pts[i][1] - pts[i-1][1])
            new_points.append((qx, qy))
            pts.insert(i, (qx, qy))
            D = 0.0
        else:
            D += d
        i += 1
    if len(new_points) < n:
        new_points.append(pts[-1])
    return new_points[:n]

def bounding_box(points):
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    return min_x, min_y, max_x - min_x, max_y - min_y

def scale_to(points, size):
    min_x, min_y, w, h = bounding_box(points)
    new_points = []
    for p in points:
        qx = p[0] * (size / w) if w != 0 else p[0]
        qy = p[1] * (size / h) if h != 0 else p[1]
        new_points.append((qx, qy))
    return new_points

def centroid(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y

def translate_to(points, pt):
    c = centroid(points)
    new_points = []
    for p in points:
        qx = p[0] + pt[0] - c[0]
        qy = p[1] + pt[1] - c[1]
        new_points.append((qx, qy))
    return new_points

def path_distance(pts1, pts2):
    d = 0.0
    for p1, p2 in zip(pts1, pts2):
        d += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return d / len(pts1)

def preprocess_path(points, num_points=32):
    pts = resample(list(points), num_points)
    pts = scale_to(pts, 100.0)
    pts = translate_to(pts, (0.0, 0.0))
    return pts

RAW_TEMPLATES = {
    'A': [(0, 100), (50, 0), (100, 100), (75, 50), (25, 50)],
    'B': [(0, 100), (0, 0), (100, 25), (0, 50), (100, 75), (0, 100)],
    'C': [(100, 0), (0, 0), (0, 100), (100, 100)],
    'D': [(0, 100), (0, 0), (100, 50), (0, 100)],
    'E': [(100, 0), (0, 0), (0, 100), (100, 100), (0, 100), (0, 50), (80, 50)],
    'F': [(100, 0), (0, 0), (0, 100), (0, 50), (80, 50)],
    'G': [(100, 0), (0, 0), (0, 100), (100, 100), (100, 50), (50, 50)],
    'H': [(0, 0), (0, 100), (0, 50), (100, 50), (100, 0), (100, 100)],
    'I': [(50, 0), (50, 100)],
    'J': [(100, 0), (100, 100), (0, 100)],
    'K': [(0, 0), (0, 100), (0, 50), (100, 0), (0, 50), (100, 100)],
    'L': [(0, 0), (0, 100), (100, 100)],
    'M': [(0, 100), (0, 0), (50, 50), (100, 0), (100, 100)],
    'N': [(0, 100), (0, 0), (100, 100), (100, 0)],
    'O': [(50, 0), (100, 50), (50, 100), (0, 50), (50, 0)],
    'P': [(0, 100), (0, 0), (100, 0), (100, 50), (0, 50)],
    'Q': [(50, 0), (100, 50), (50, 100), (0, 50), (50, 0), (50, 50), (100, 100)],
    'R': [(0, 100), (0, 0), (100, 0), (100, 50), (0, 50), (100, 100)],
    'S': [(100, 0), (0, 25), (100, 75), (0, 100)],
    'T': [(0, 0), (100, 0), (50, 0), (50, 100)],
    'U': [(0, 0), (0, 100), (100, 100), (100, 0)],
    'V': [(0, 0), (50, 100), (100, 0)],
    'W': [(0, 0), (25, 100), (50, 50), (75, 100), (100, 0)],
    'X': [(0, 0), (100, 100), (50, 50), (100, 0), (0, 100)],
    'Y': [(0, 0), (50, 50), (100, 0), (50, 50), (50, 100)],
    'Z': [(0, 0), (100, 0), (0, 100), (100, 100)],
    
    # Unistroke Digits 0-9
    '0': [(50, 0), (100, 25), (100, 75), (50, 100), (0, 75), (0, 25), (50, 0)],
    '1': [(45, 0), (45, 100)],
    '2': [(0, 25), (50, 0), (100, 25), (0, 100), (100, 100)],
    '3': [(0, 25), (50, 0), (100, 25), (50, 50), (100, 75), (50, 100), (0, 75)],
    '4': [(50, 0), (0, 50), (100, 50), (100, 0), (100, 100)],
    '5': [(100, 0), (0, 0), (0, 50), (100, 50), (100, 100), (0, 100)],
    '6': [(100, 0), (0, 50), (0, 100), (100, 100), (100, 50), (0, 50)],
    '7': [(0, 0), (100, 0), (0, 100)],
    '8': [(50, 50), (0, 25), (50, 0), (100, 25), (50, 50), (0, 75), (50, 100), (100, 75), (50, 50)],
    '9': [(100, 50), (0, 50), (0, 0), (100, 0), (100, 100), (0, 100)]
}

TEMPLATES = {char: preprocess_path(pts) for char, pts in RAW_TEMPLATES.items()}

def recognize(points):
    if len(points) < 5:
        return None
    try:
        processed = preprocess_path(points)
        best_char = None
        min_dist = float('inf')
        for char, template in TEMPLATES.items():
            dist = path_distance(processed, template)
            if dist < min_dist:
                min_dist = dist
                best_char = char
        print(f"Recognized: {best_char} (Score/Distance: {min_dist:.2f})")
        if min_dist < 45.0:
            return best_char
    except Exception as e:
        print(f"Recognition error: {e}")
    return None
# ==============================================================================

def get_distance(p1, p2):
    return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

def count_fingers_up(lm_list, handedness):
    """Count extended fingers. Returns integer 0-5."""
    if len(lm_list) < 21:
        return 0
    count = 0
    
    # Determine if palm is facing the camera vs back of hand
    if handedness == 'Right':
        is_palm_facing = lm_list[5][1] < lm_list[17][1]
    else:
        is_palm_facing = lm_list[5][1] > lm_list[17][1]
        
    # Thumb: compare tip (4) to IP joint (3) horizontally
    if handedness == 'Right':
        if is_palm_facing:
            if lm_list[4][1] < lm_list[3][1]:
                count += 1
        else:
            if lm_list[4][1] > lm_list[3][1]:
                count += 1
    else:
        if is_palm_facing:
            if lm_list[4][1] > lm_list[3][1]:
                count += 1
        else:
            if lm_list[4][1] < lm_list[3][1]:
                count += 1
                
    # Index: tip (8) above PIP (6)
    if lm_list[8][2] < lm_list[6][2]:
        count += 1
    # Middle: tip (12) above PIP (10)
    if lm_list[12][2] < lm_list[10][2]:
        count += 1
    # Ring: tip (16) above PIP (14)
    if lm_list[16][2] < lm_list[14][2]:
        count += 1
    # Pinky: tip (20) above PIP (18)
    if lm_list[20][2] < lm_list[18][2]:
        count += 1
    return count

def is_three_fingers_up(lm_list):
    if len(lm_list) < 21:
        return False
    index_up = lm_list[8][2] < lm_list[6][2]
    middle_up = lm_list[12][2] < lm_list[10][2]
    ring_up = lm_list[16][2] < lm_list[14][2]
    pinky_up = lm_list[20][2] < lm_list[18][2]
    return index_up and middle_up and ring_up and not pinky_up

def is_rock_on_gesture(lm_list, handedness):
    """Detect rock-on/spider-man gesture: index up, pinky up, middle down, ring down, thumb out."""
    if len(lm_list) < 21:
        return False
    index_up = lm_list[8][2] < lm_list[6][2]
    middle_down = lm_list[12][2] > lm_list[10][2]
    ring_down = lm_list[16][2] > lm_list[14][2]
    pinky_up = lm_list[20][2] < lm_list[18][2]
    
    # Thumb out: tip (4) is far from index MCP (5) horizontally
    thumb_out = abs(lm_list[4][1] - lm_list[5][1]) > abs(lm_list[3][1] - lm_list[5][1])
    
    return index_up and middle_down and ring_down and pinky_up and thumb_out

def draw_glowing_skeleton(canvas, lm_list, connections, color):
    """Draws a premium hologram/glowing hand skeleton on canvas."""
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(lm_list) and end_idx < len(lm_list):
            p1 = (lm_list[start_idx][1], lm_list[start_idx][2])
            p2 = (lm_list[end_idx][1], lm_list[end_idx][2])
            # Outer diffuse glow
            cv2.line(canvas, p1, p2, (color[0]//5, color[1]//5, color[2]//5), 10, cv2.LINE_AA)
            # Inner bright glow
            cv2.line(canvas, p1, p2, (color[0]//2, color[1]//2, color[2]//2), 5, cv2.LINE_AA)
            # High-intensity core
            cv2.line(canvas, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
            
    # Draw joints
    for id, cx, cy in lm_list:
        cv2.circle(canvas, (cx, cy), 7, (color[0]//3, color[1]//3, color[2]//3), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

# --- PACE-DELAYED HOTKEY INJECTORS ---
def trigger_open_view():
    os_name = platform.system()
    if os_name == "Windows":
        pyautogui.keyDown('win')
        time.sleep(0.05)
        pyautogui.press('tab')
        time.sleep(0.05)
        pyautogui.keyUp('win')
    else:
        pyautogui.keyDown('ctrl')
        time.sleep(0.05)
        pyautogui.press('up')
        time.sleep(0.05)
        pyautogui.keyUp('ctrl')
    print("--> OPENING Multitasking View")

def trigger_onscreen_keyboard():
    print("--> OPENING On-Screen Keyboard")
    try:
        pyautogui.keyDown('win')
        pyautogui.keyDown('ctrl')
        time.sleep(0.05)
        pyautogui.press('o')
        time.sleep(0.05)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('win')
    except Exception as e:
        print(f"Error opening keyboard: {e}")

def trigger_next_desktop():
    print("--> SWITCHING to Next Desktop (Right)")
    try:
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('win')
        time.sleep(0.05)
        pyautogui.press('right')
        time.sleep(0.05)
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
    except Exception as e:
        print(f"Error switching desktop: {e}")

def trigger_prev_desktop():
    print("<-- SWITCHING to Previous Desktop (Left)")
    try:
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('win')
        time.sleep(0.05)
        pyautogui.press('left')
        time.sleep(0.05)
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
    except Exception as e:
        print(f"Error switching desktop: {e}")


# --- STATE TRACKING ---
# Left Hand Click State
left_ready_to_click = False  

# Right Hand Swipe History for Backspace Gesture
right_hand_history = []
last_swipe_time = 0.0
SWIPE_LEFT_THRESHOLD = 0.8  # Threshold relative to palm size
SWIPE_COOLDOWN_SEC = 0.4

# Replicating Mode States
replicating_mode = False
last_mode_toggle_time = 0.0
MODE_TOGGLE_COOLDOWN = 1.0  # Cooldown between state switches

# Right Hand Multitasking State
right_ready_to_open = False   

# Right Hand Volume Control State
volume_mode_active = False
volume_anchor_y = None

# Zoom Mode State
zoom_mode_active = False
zoom_anchor_ratio = None

# Head Tilt States
head_ready_to_open_keyboard = True
head_tilted_up = False
head_tilted_down = False
head_tilted_left = False

# Head Flick State
nose_history = []
last_flick_time = 0.0

# Drawing Mode State
drawing_active = False
drawing_points = []

# Cursor Smoothing History
prev_screen_x = None
prev_screen_y = None

# Screen Resolution (Read once)
screen_w, screen_h = pyautogui.size()
print(f"Screen resolution detected: {screen_w}x{screen_h}")

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

    # Reset frame head states
    head_tilted_left = False
    head_tilted_up = False
    head_tilted_down = False

    hand_results = hand_detector.detect_for_video(mp_image, timestamp_ms)
    face_results = face_detector.detect_for_video(mp_image, timestamp_ms)

    # --- INSTANT PRE-SCAN HAND LANDMARKS & ISOLATION STATE ---
    left_hand_seen = False
    right_hand_seen = False
    left_fist_active = False
    is_drawing = False
    is_zooming = False
    left_hand_lms = None
    right_hand_lms = None
    closest_hands = []

    if hand_results.hand_landmarks:
        valid_hands = []
        for idx in range(len(hand_results.hand_landmarks)):
            landmarks = hand_results.hand_landmarks[idx]
            handedness_data = hand_results.handedness[idx][0]
            
            pixel_lms = []
            for id, lm in enumerate(landmarks):
                pixel_lms.append([id, int(lm.x * w), int(lm.y * h)])
            
            if len(pixel_lms) > 9:
                palm_sz = get_distance(pixel_lms[0], pixel_lms[9])
                valid_hands.append({
                    'landmarks': pixel_lms,
                    'raw_landmarks': landmarks,
                    'handedness': 'Left' if handedness_data.category_name == 'Right' else 'Right',
                    'palm_size': palm_sz
                })

        valid_hands = sorted(valid_hands, key=lambda x: x['palm_size'], reverse=True)
        closest_hands = valid_hands[:2]

        # Scan Hand states first
        for hand in closest_hands:
            if hand['handedness'] == 'Left':
                left_hand_seen = True
                left_hand_lms = hand['landmarks']
                if is_three_fingers_up(hand['landmarks']):
                    is_drawing = True
                elif is_rock_on_gesture(hand['landmarks'], hand['handedness']):
                    is_zooming = True
                if count_fingers_up(hand['landmarks'], 'Left') == 0:
                    left_fist_active = True
            elif hand['handedness'] == 'Right':
                right_hand_seen = True
                right_hand_lms = hand['landmarks']

    # Face landmarks wireframe & tilt/flick/pitch detection
    face_pixel_lms = None
    if face_results.face_landmarks:
        for face_landmarks in face_results.face_landmarks:
            face_pixel_lms = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection.start, connection.end
                if start_idx < len(face_pixel_lms) and end_idx < len(face_pixel_lms):
                    cv2.line(image, face_pixel_lms[start_idx], face_pixel_lms[end_idx], (255, 255, 0), 1) 

            # Calculate head coordinates and rotation angles (Pitch/Roll/Yaw)
            p_left_left = face_landmarks[33]
            p_right_right = face_landmarks[263]
            p_left_eye = p_left_left
            p_right_eye = p_right_right

            dx = p_right_eye.x - p_left_eye.x
            dy = p_right_eye.y - p_left_eye.y
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            y_nose_bridge = face_landmarks[6].y
            y_nose = face_landmarks[4].y
            y_chin = face_landmarks[152].y
            upper_dist = max(0.001, y_nose - y_nose_bridge)
            lower_dist = max(0.001, y_chin - y_nose)
            pitch_ratio = upper_dist / lower_dist

            head_tilted_up = pitch_ratio < PITCH_LOOK_UP_THRESHOLD
            head_tilted_down = pitch_ratio > PITCH_LOOK_DOWN_THRESHOLD
            head_tilted_left = angle_deg < TILT_LEFT_TRIGGER

            avg_eye_x = int((p_left_eye.x + p_right_eye.x) / 2 * w)
            avg_eye_y = int((p_left_eye.y + p_right_eye.y) / 2 * h)

            # Only trigger standard head actions if Left Hand is NOT in a fist and NOT in replicating mode
            if not left_fist_active and not replicating_mode:
                # Keyboard Open (Roll Tilt Right)
                if abs(angle_deg) < TILT_NEUTRAL_LIMIT:
                    head_ready_to_open_keyboard = True
                elif angle_deg > TILT_RIGHT_TRIGGER:
                    if head_ready_to_open_keyboard:
                        trigger_onscreen_keyboard()
                        head_ready_to_open_keyboard = False

                if head_tilted_left:
                    tilt_status = "Uppercase Active"
                elif not head_ready_to_open_keyboard:
                    tilt_status = "Keyboard Triggered"
                else:
                    tilt_status = "Neutral"
                    
                cv2.putText(image, f"Tilt: {angle_deg:.1f} deg ({tilt_status})", 
                            (avg_eye_x - 100, avg_eye_y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Vertical Scrolling (Pitch Tilt)
                if head_tilted_up:
                    try:
                        pyautogui.scroll(75)
                        scroll_status = "SCROLLING UP"
                    except Exception as e:
                        print(f"Scroll Up Error: {e}")
                        scroll_status = "Error"
                elif head_tilted_down:
                    try:
                        pyautogui.scroll(-75)
                        scroll_status = "SCROLLING DOWN"
                    except Exception as e:
                        print(f"Scroll Down Error: {e}")
                        scroll_status = "Error"
                else:
                    scroll_status = "Neutral Pitch"
                
                case_color = (0, 255, 0) if scroll_status != "Neutral Pitch" else (255, 0, 0)
                cv2.putText(image, f"Pitch: {pitch_ratio:.2f} ({scroll_status})", 
                            (avg_eye_x - 100, avg_eye_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, case_color, 2)

                # Flick Virtual Desktop (Yaw Sweep)
                p_nose = face_landmarks[4]
                current_nose_x = p_nose.x
                nose_history.append((current_time_sec, current_nose_x))
                nose_history = [(t, x) for t, x in nose_history if current_time_sec - t <= 0.3]
                
                target_time = current_time_sec - 0.15
                best_x = None
                min_diff = float('inf')
                for t, x in nose_history:
                    diff = abs(t - target_time)
                    if diff < min_diff:
                        min_diff = diff
                        best_x = x
                
                if min_diff < 0.05 and best_x is not None:
                    dx_nose = current_nose_x - best_x
                    if current_time_sec - last_flick_time > FLICK_COOLDOWN_SEC:
                        if dx_nose < -FLICK_THRESHOLD:
                            trigger_next_desktop()
                            last_flick_time = current_time_sec
                        elif dx_nose > FLICK_THRESHOLD:
                            trigger_prev_desktop()
                            last_flick_time = current_time_sec
            else:
                # Visual feed indication that head controls are locked
                cv2.putText(image, "Head Gestures: LOCKED (Left Fist active)", (30, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- REPLICATING MODE STATE MACHINE TRIGGER ---
    if left_hand_lms is not None and right_hand_lms is not None:
        left_fist = (count_fingers_up(left_hand_lms, 'Left') == 0)
        right_fist = (count_fingers_up(right_hand_lms, 'Right') == 0)
        
        if left_fist and right_fist:
            if head_tilted_up and not replicating_mode:
                if current_time_sec - last_mode_toggle_time > MODE_TOGGLE_COOLDOWN:
                    replicating_mode = True
                    last_mode_toggle_time = current_time_sec
                    print("--> ENTERED REPLICATING MODE")
            elif head_tilted_down and replicating_mode:
                if current_time_sec - last_mode_toggle_time > MODE_TOGGLE_COOLDOWN:
                    replicating_mode = False
                    last_mode_toggle_time = current_time_sec
                    print("--> EXITED REPLICATING MODE")

    # Detailed Hands Processing
    right_index_pixel_pos = None

    for hand in closest_hands:
        lm_list = hand['landmarks']
        raw_lms = hand['raw_landmarks']
        handedness = hand['handedness']
        palm_distance = max(1, hand['palm_size'])
        finger_count = count_fingers_up(lm_list, handedness)

        # Render landmarks and connections on main webcam frame
        for id, cx, cy in lm_list:
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(lm_list) and end_idx < len(lm_list):
                cv2.line(image, (lm_list[start_idx][1], lm_list[start_idx][2]), 
                         (lm_list[end_idx][1], lm_list[end_idx][2]), (0, 0, 255), 2)

        raw_pinch_distance = get_distance(lm_list[4], lm_list[8])
        pinch_ratio = raw_pinch_distance / palm_distance
        
        cv2.line(image, (lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]), (0, 255, 255), 2)

        # Text Feedback positioning
        wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]
        text_x, text_y = max(10, wrist_x - 50), max(30, wrist_y + 40)

        # Display finger count on webcam frame
        mid_tip_x, mid_tip_y = lm_list[12][1], lm_list[12][2]
        fc_x, fc_y = max(10, mid_tip_x - 15), max(30, mid_tip_y - 20)
        cv2.putText(image, f'{finger_count}', (fc_x, fc_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
        cv2.putText(image, f'{finger_count}', (fc_x, fc_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

        # --------------------------------------------------------------
        # HAND-SPECIFIC ACTION LOGIC
        # --------------------------------------------------------------
        if handedness == 'Left':
            if replicating_mode:
                cv2.putText(image, f'{handedness} (Replicating Active)', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                continue
                
            left_three_fingers = is_drawing
            
            if is_zooming and finger_count != 2:
                cv2.putText(image, f'{handedness} (Zoom Mode) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif left_three_fingers:
                cv2.putText(image, f'{handedness} (Drawing Active) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif left_fist_active:
                cv2.putText(image, f'{handedness} (Swipe Mode) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(image, f'Swipe right hand left to delete', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # LEFT HAND ACTION LOGIC: Click/Press triggers
                if pinch_ratio <= LEFT_PINCH_IN_THRESHOLD:
                    if left_ready_to_click:
                        try:
                            pyautogui.click()
                            print("--> Left hand CLICKED")
                        except Exception as e:
                            print(f"Error executing click: {e}")
                        left_ready_to_click = False

                elif pinch_ratio >= LEFT_PINCH_OUT_THRESHOLD:
                    left_ready_to_click = True

                click_status = "Primed" if left_ready_to_click else "Needs Prime"
                cv2.putText(image, f'{handedness} (Click) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, f'Pinch Ratio: {pinch_ratio:.2f}', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(image, f'Click: {click_status}', (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        elif handedness == 'Right':
            right_index_pixel_pos = (lm_list[8][1], lm_list[8][2])
            
            # Update right hand wrist history for swipe tracking
            right_hand_history.append((current_time_sec, lm_list[0][1], palm_distance))
            right_hand_history = [(t, x, ps) for t, x, ps in right_hand_history if current_time_sec - t <= 0.3]
            
            # Check for swipe left
            if left_fist_active and not replicating_mode and len(right_hand_history) > 1:
                recent_pts = [pt for pt in right_hand_history if current_time_sec - pt[0] <= 0.25]
                if recent_pts:
                    min_pt = min(recent_pts, key=lambda pt: pt[1])
                    min_x = min_pt[1]
                    t_min = min_pt[0]
                    curr_x = lm_list[0][1]
                    
                    swipe_dist = (curr_x - min_x) / palm_distance
                    
                    if swipe_dist > SWIPE_LEFT_THRESHOLD and (current_time_sec - t_min) > 0.05:
                        if current_time_sec - last_swipe_time > SWIPE_COOLDOWN_SEC:
                            try:
                                pyautogui.press('backspace')
                                print("--> SWIPE LEFT detected: BACKSPACE pressed")
                            except Exception as e:
                                print(f"Error pressing backspace: {e}")
                            last_swipe_time = current_time_sec
                            right_hand_history = []

            # ISOLATION BARRIER: If Left Fist is closed OR Replicating Mode is active, block standard mouse actions
            if left_fist_active or replicating_mode:
                mode_label = "Replicating Mode" if replicating_mode else "Swiping Mode"
                cv2.putText(image, f'{handedness} ({mode_label}) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if left_fist_active and not replicating_mode and len(right_hand_history) > 1:
                    recent_pts = [pt for pt in right_hand_history if current_time_sec - pt[0] <= 0.25]
                    if recent_pts:
                        min_pt = min(recent_pts, key=lambda pt: pt[1])
                        min_x = min_pt[1]
                        curr_x = lm_list[0][1]
                        swipe_dist = (curr_x - min_x) / palm_distance
                        cv2.putText(image, f'Swipe Dist: {swipe_dist:.2f} / {SWIPE_LEFT_THRESHOLD}', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                continue

            # Standard Right Hand Actions (Cursor, Volume, Zoom)
            if is_zooming and finger_count != 2:
                volume_mode_active = False
                volume_anchor_y = None
                
                if not zoom_mode_active:
                    zoom_mode_active = True
                    zoom_anchor_ratio = pinch_ratio
                else:
                    delta_ratio = zoom_anchor_ratio - pinch_ratio
                    if abs(delta_ratio) >= ZOOM_STEP_RATIO:
                        steps = int(delta_ratio / ZOOM_STEP_RATIO)
                        if steps > 0:
                            for _ in range(steps):
                                pyautogui.keyDown('ctrl')
                                pyautogui.scroll(24)
                                pyautogui.keyUp('ctrl')
                            print(f"ZOOM IN x{steps}")
                        elif steps < 0:
                            for _ in range(abs(steps)):
                                pyautogui.keyDown('ctrl')
                                pyautogui.scroll(-24)
                                pyautogui.keyUp('ctrl')
                            print(f"ZOOM OUT x{abs(steps)}")
                        zoom_anchor_ratio -= steps * ZOOM_STEP_RATIO
                
                zoom_dir = "IN" if zoom_anchor_ratio and pinch_ratio < zoom_anchor_ratio else ("OUT" if zoom_anchor_ratio and pinch_ratio > zoom_anchor_ratio else "---")
                cv2.putText(image, f'{handedness} (Zoom: {zoom_dir}) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(image, f'Pinch Ratio: {pinch_ratio:.2f}', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            elif finger_count == 0 and not is_drawing:
                zoom_mode_active = False
                zoom_anchor_ratio = None
                right_ready_to_open = True
                
                current_wrist_y = lm_list[0][2]
                
                if not volume_mode_active:
                    volume_mode_active = True
                    volume_anchor_y = current_wrist_y
                else:
                    delta_y = volume_anchor_y - current_wrist_y
                    if abs(delta_y) >= VOLUME_STEP_PIXELS:
                        steps = int(delta_y / VOLUME_STEP_PIXELS)
                        if steps > 0:
                            for _ in range(steps):
                                pyautogui.press('volumeup')
                            print(f"VOL UP x{steps}")
                        elif steps < 0:
                            for _ in range(abs(steps)):
                                pyautogui.press('volumedown')
                            print(f"VOL DOWN x{abs(steps)}")
                        volume_anchor_y -= steps * VOLUME_STEP_PIXELS
                
                if volume_anchor_y is not None:
                    delta_display = volume_anchor_y - current_wrist_y
                    vol_dir = "UP" if delta_display > 0 else ("DOWN" if delta_display < 0 else "---")
                    cv2.putText(image, f'{handedness} (Volume: {vol_dir}) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                    cv2.putText(image, f'Move: {delta_display:+d}px', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                    cv2.line(image, (wrist_x - 40, volume_anchor_y), (wrist_x + 40, volume_anchor_y), (255, 128, 0), 2)
            
            else:
                zoom_mode_active = False
                zoom_anchor_ratio = None
                volume_mode_active = False
                volume_anchor_y = None
                
                if not is_drawing:
                    cursor_lm = raw_lms[8]
                    
                    clamped_x = max(ACTIVE_BOX_X_MIN, min(ACTIVE_BOX_X_MAX, cursor_lm.x))
                    clamped_y = max(ACTIVE_BOX_Y_MIN, min(ACTIVE_BOX_Y_MAX, cursor_lm.y))
                    
                    norm_x = (clamped_x - ACTIVE_BOX_X_MIN) / (ACTIVE_BOX_X_MAX - ACTIVE_BOX_X_MIN)
                    norm_y = (clamped_y - ACTIVE_BOX_Y_MIN) / (ACTIVE_BOX_Y_MAX - ACTIVE_BOX_Y_MIN)
                    
                    target_x = int(norm_x * screen_w)
                    target_y = int(norm_y * screen_h)
                    
                    if prev_screen_x is None:
                        smooth_x = target_x
                        smooth_y = target_y
                    else:
                        smooth_x = int(SMOOTHING_FACTOR * target_x + (1 - SMOOTHING_FACTOR) * prev_screen_x)
                        smooth_y = int(SMOOTHING_FACTOR * target_y + (1 - SMOOTHING_FACTOR) * prev_screen_y)
                    
                    smooth_x = max(2, min(screen_w - 2, smooth_x))
                    smooth_y = max(2, min(screen_h - 2, smooth_y))
                    
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y)
                    except pyautogui.FailSafeException:
                        pass
                    
                    prev_screen_x = smooth_x
                    prev_screen_y = smooth_y

                    if pinch_ratio >= RIGHT_PINCH_OUT_THRESHOLD:
                        if right_ready_to_open:
                            trigger_open_view()
                            right_ready_to_open = False

                    view_status = "Ready to Open" if right_ready_to_open else "Neutral"
                    cv2.putText(image, f'{handedness} (Cursor/View) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(image, f'Pinch Ratio: {pinch_ratio:.2f}', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(image, f'Multiview: {view_status}', (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    cv2.putText(image, f'{handedness} (Drawing) [{finger_count}F]', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 3D-MODEL HOLOGRAM REPLICATOR WINDOW ---
    if replicating_mode:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw background coordinates grid
        grid_size = 40
        for x in range(0, w, grid_size):
            cv2.line(canvas, (x, 0), (x, h), (20, 20, 30), 1)
        for y in range(0, h, grid_size):
            cv2.line(canvas, (0, y), (w, y), (20, 20, 30), 1)

        # Draw glowing neon skeletons
        if left_hand_lms is not None:
            draw_glowing_skeleton(canvas, left_hand_lms, HAND_CONNECTIONS, (255, 0, 255)) # Glowing Pink
        if right_hand_lms is not None:
            draw_glowing_skeleton(canvas, right_hand_lms, HAND_CONNECTIONS, (255, 255, 0)) # Glowing Cyan

        # Draw Face Mesh Replication (Neon Cybernetic wireframe)
        if face_pixel_lms is not None:
            # Enlarge face relative to its centroid
            xs = [p[0] for p in face_pixel_lms]
            ys = [p[1] for p in face_pixel_lms]
            if xs and ys:
                cx = sum(xs) // len(face_pixel_lms)
                cy = sum(ys) // len(face_pixel_lms)
                
                scale_factor = 1.6  # 1.6x larger
                scaled_face = []
                for px, py in face_pixel_lms:
                    new_x = int(cx + (px - cx) * scale_factor)
                    new_y = int(cy + (py - cy) * scale_factor)
                    scaled_face.append((new_x, new_y))
                    
                # Bind points to canvas coordinates to prevent rendering visual errors
                face_pixel_lms = scaled_face

            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection.start, connection.end
                if start_idx < len(face_pixel_lms) and end_idx < len(face_pixel_lms):
                    p1 = face_pixel_lms[start_idx]
                    p2 = face_pixel_lms[end_idx]
                    # Glowing ice-blue connection lines
                    cv2.line(canvas, p1, p2, (255, 180, 0), 1, cv2.LINE_AA) # BGR: ice blue/light cyan
                    cv2.line(canvas, p1, p2, (255, 255, 255), 1, cv2.LINE_AA) # White core

        # Add visual hologram HUD headers
        cv2.putText(canvas, "SYSTEM: DIGITAL TWIN REPLICATOR ACTIVE", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "EXIT: BOTH FISTS + HEAD TILT DOWN", (20, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Display telemetry coordinates next to index tips
        if left_hand_lms is not None:
            l_idx = left_hand_lms[8]
            cv2.putText(canvas, f"L INDEX: ({l_idx[1]},{l_idx[2]})", (l_idx[1]+15, l_idx[2]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        if right_hand_lms is not None:
            r_idx = right_hand_lms[8]
            cv2.putText(canvas, f"R INDEX: ({r_idx[1]},{r_idx[2]})", (r_idx[1]+15, r_idx[2]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Twin Replicator', canvas)
    else:
        # Gracefully close the window if exiting replicating mode
        try:
            if cv2.getWindowProperty('Twin Replicator', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Twin Replicator')
        except Exception:
            pass

    # Reset zoom when left hand drops rock-on gesture
    if not is_zooming:
        zoom_mode_active = False
        zoom_anchor_ratio = None

    # --- 5. DRAWING MODE STATE MACHINE ---
    if is_drawing:
        if not drawing_active:
            drawing_active = True
            drawing_points = []
            print("--> START DRAWING MODE")
        
        if right_hand_seen and right_index_pixel_pos is not None:
            if not drawing_points or math.hypot(right_index_pixel_pos[0] - drawing_points[-1][0], 
                                                right_index_pixel_pos[1] - drawing_points[-1][1]) > 3:
                drawing_points.append(right_index_pixel_pos)
    else:
        if drawing_active:
            print("--> PROCESSING DRAWN LETTER...")
            if len(drawing_points) > 5:
                letter = recognize(drawing_points)
                if letter:
                    try:
                        typed_char = letter.upper() if head_tilted_left else letter.lower()
                        pyautogui.write(typed_char)
                        print(f"--> TYPED LETTER: {typed_char}")
                    except Exception as e:
                        print(f"Error typing letter: {e}")
            drawing_active = False
            drawing_points = []
            print("--> END DRAWING MODE")

    if drawing_active and len(drawing_points) > 1:
        cv2.putText(image, "DRAWING MODE: Trace a letter. Lower Left fingers to TYPE.", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for i in range(1, len(drawing_points)):
            cv2.line(image, drawing_points[i-1], drawing_points[i], (0, 0, 255), 4)

    # General overlay indicator on webcam frame when Replicating Mode is active
    if replicating_mode:
        cv2.putText(image, "REPLICATING ACTIVE (Lock on Twin)", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not right_hand_seen:
        volume_mode_active = False
        volume_anchor_y = None
        zoom_mode_active = False
        zoom_anchor_ratio = None
        prev_screen_x = None
        prev_screen_y = None
        right_hand_history = []

    cv2.imshow('MediaPipe Tracking', image)
    
    key = cv2.waitKeyEx(5)
    if key in [45, 0x2D, 0x2d0000]:
        print("Closing Python script via Insert key...")
        break

cap.release()
cv2.destroyAllWindows()