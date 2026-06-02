import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================
# PREVENT PYAUTOGUI FAILSAFE
# ============================================

pyautogui.FAILSAFE = False

# ============================================
# MEDIAPIPE HAND CONNECTIONS
# ============================================

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

# ============================================
# HAND TRACKER SETUP
# ============================================

hand_base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)

# ============================================
# FACE TRACKER SETUP
# ============================================

face_base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)

# ============================================
# KEY STATES
# ============================================

keys_down = {
    "w": False,
    "a": False,
    "s": False,
    "d": False
}

def hold_key(key, active):

    global keys_down

    if active and not keys_down[key]:

        pyautogui.keyDown(key)
        keys_down[key] = True

    elif not active and keys_down[key]:

        pyautogui.keyUp(key)
        keys_down[key] = False

# ============================================
# HELPER FUNCTIONS
# ============================================

def finger_up(tip, pip, hand):
    return hand[tip].y < hand[pip].y

# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # ============================================
    # RESET STATES
    # ============================================

    w_active = False
    a_active = False
    s_active = False
    d_active = False

    # ============================================
    # HAND TRACKING
    # ============================================

    hand_result = hand_detector.detect(mp_image)

    if hand_result.hand_landmarks and hand_result.handedness:

        for i, hand in enumerate(hand_result.hand_landmarks):

            handedness = hand_result.handedness[i][0].category_name

            # ============================================
            # DRAW HAND CONNECTIONS
            # ============================================

            for connection in HAND_CONNECTIONS:

                start_idx = connection[0]
                end_idx = connection[1]

                start = hand[start_idx]
                end = hand[end_idx]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,255),
                    2
                )

            # ============================================
            # DRAW HAND LANDMARKS
            # ============================================

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0,255,0),
                    -1
                )

            # ============================================
            # SHOW HANDEDNESS
            # ============================================

            wrist = hand[0]

            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            cv2.putText(
                frame,
                handedness,
                (wrist_x, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            # ============================================
            # FINGER STATES
            # ============================================

            index_up = finger_up(8, 6, hand)
            middle_up = finger_up(12, 10, hand)
            ring_up = finger_up(16, 14, hand)
            pinky_up = finger_up(20, 18, hand)

            # ============================================
            # GESTURES
            # ============================================

            # THREE FINGERS = H
            three_fingers = (
                index_up and
                middle_up and
                ring_up and
                not pinky_up
            )

            # OPEN PALM
            open_palm = (
                index_up and
                middle_up and
                ring_up and
                pinky_up
            )

            # FIST
            fist = (
                not index_up and
                not middle_up and
                not ring_up and
                not pinky_up
            )

            # ============================================
            # MOVEMENT KEYS
            # ============================================

            # LEFT HAND PALM = W
            if handedness == "Right" and open_palm:

                w_active = True

                cv2.putText(
                    frame,
                    "W",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            # RIGHT HAND PALM = S
            if handedness == "Left" and open_palm:

                s_active = True

                cv2.putText(
                    frame,
                    "S",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            # ============================================
            # TAP KEYS
            # ============================================

            # H = THREE FINGERS
            if three_fingers:

                pyautogui.press("h")

                cv2.putText(
                    frame,
                    "H",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,0),
                    2
                )

            # SPACE = FIST
            if fist:

                pyautogui.press("space")

                cv2.putText(
                    frame,
                    "SPACE",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,0),
                    2
                )

    # ============================================
    # FACE TRACKING
    # ============================================

    face_result = face_detector.detect(mp_image)

    if face_result.face_landmarks:

        for face in face_result.face_landmarks:

            # ============================================
            # DRAW FACE LANDMARKS
            # ============================================

            for landmark in face:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (255,0,0),
                    -1
                )

            # ============================================
            # HEAD TILT DETECTION
            # ============================================

            left_eye = face[33]
            right_eye = face[263]

            eye_difference = left_eye.y - right_eye.y

            # HEAD TILT LEFT = A
            if eye_difference > 0.02:

                a_active = True

                cv2.putText(
                    frame,
                    "A",
                    (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )

            # HEAD TILT RIGHT = D
            if eye_difference < -0.02:

                d_active = True

                cv2.putText(
                    frame,
                    "D",
                    (500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )

    # ============================================
    # APPLY HOLD KEYS
    # ============================================

    hold_key("w", w_active)
    hold_key("a", a_active)
    hold_key("s", s_active)
    hold_key("d", d_active)

    # ============================================
    # DISPLAY
    # ============================================

    cv2.imshow(
        "Gesture Controller",
        frame
    )

    # ESC TO QUIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================
# RELEASE ALL KEYS
# ============================================

for key in keys_down:

    if keys_down[key]:
        pyautogui.keyUp(key)

# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()