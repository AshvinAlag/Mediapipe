import pygame
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import random
import math
import os
import urllib.request

# ==============================================================================
#                               CONFIGURATIONS
# ==============================================================================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 750
ROAD_WIDTH = 320
LANE_WIDTH = ROAD_WIDTH / 3
CURB_WIDTH = 15
SLICE_H = 10
NUM_SLICES = SCREEN_HEIGHT // SLICE_H + 2
PLAYER_Y = 580

# Hand Tracking Sensitivity
STEERING_SENSITIVITY = 1.8

# Model task file URL
MODEL_FILE = 'hand_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# MediaPipe Hand Connections for skeleton rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# ==============================================================================
#                            THREAD-SAFE HAND STATE
# ==============================================================================
class HandState:
    def __init__(self):
        self.steering = 0.0      # -1.0 (left) to 1.0 (right)
        self.drs_active = False  # True when finger guns are detected
        self.camera_frame = None # Resized RGB frame for PIP (200x150)
        self.hand_detected = False

hand_state = HandState()
state_lock = threading.Lock()
running = True
camera_error = False

# ==============================================================================
#                         BACKGROUND CAMERA WORKER
# ==============================================================================
def draw_glowing_skeleton(canvas, lm_list, connections, color):
    """Draws a premium glowing hand skeleton on the camera frame."""
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(lm_list) and end_idx < len(lm_list):
            p1 = lm_list[start_idx]
            p2 = lm_list[end_idx]
            # Outer diffuse glow
            cv2.line(canvas, p1, p2, (color[0]//5, color[1]//5, color[2]//5), 10, cv2.LINE_AA)
            # Inner bright glow
            cv2.line(canvas, p1, p2, (color[0]//2, color[1]//2, color[2]//2), 5, cv2.LINE_AA)
            # High-intensity core
            cv2.line(canvas, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
            
    # Draw joints
    for pt in lm_list:
        cv2.circle(canvas, pt, 6, (color[0]//3, color[1]//3, color[2]//3), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

def count_fingers_up(landmarks, handedness):
    """Count extended fingers using MediaPipe normalized landmarks. Returns 0-5.
    landmarks: list of NormalizedLandmark objects from MediaPipe.
    handedness: 'Left' or 'Right' (already swapped for mirror).
    """
    # Determine if palm is facing camera
    if handedness == 'Right':
        is_palm_facing = landmarks[5].x < landmarks[17].x
    else:
        is_palm_facing = landmarks[5].x > landmarks[17].x

    count = 0

    # Thumb: compare tip (4) to IP joint (3) horizontally
    if handedness == 'Right':
        if is_palm_facing:
            if landmarks[4].x < landmarks[3].x:
                count += 1
        else:
            if landmarks[4].x > landmarks[3].x:
                count += 1
    else:
        if is_palm_facing:
            if landmarks[4].x > landmarks[3].x:
                count += 1
        else:
            if landmarks[4].x < landmarks[3].x:
                count += 1

    # Index: tip (8) above PIP (6)
    if landmarks[8].y < landmarks[6].y:
        count += 1
    # Middle: tip (12) above PIP (10)
    if landmarks[12].y < landmarks[10].y:
        count += 1
    # Ring: tip (16) above PIP (14)
    if landmarks[16].y < landmarks[14].y:
        count += 1
    # Pinky: tip (20) above PIP (18)
    if landmarks[20].y < landmarks[18].y:
        count += 1

    return count

def check_finger_gun(landmarks, handedness):
    """Checks if the hand is in a 'finger gun' gesture (only thumb and index extended).
    landmarks: list of NormalizedLandmark objects from MediaPipe.
    handedness: 'Left' or 'Right'.
    """
    def dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    # Index: tip (8) to MCP (5) distance vs PIP (6) to MCP (5)
    index_extended = dist(landmarks[8], landmarks[5]) > dist(landmarks[6], landmarks[5]) * 1.1

    # Middle: tip (12) to MCP (9) vs PIP (10) to MCP (9)
    middle_folded = dist(landmarks[12], landmarks[9]) < dist(landmarks[10], landmarks[9]) * 0.95

    # Ring: tip (16) to MCP (13) vs PIP (14) to MCP (13)
    ring_folded = dist(landmarks[16], landmarks[13]) < dist(landmarks[14], landmarks[13]) * 0.95

    # Pinky: tip (20) to MCP (17) vs PIP (18) to MCP (17)
    pinky_folded = dist(landmarks[20], landmarks[17]) < dist(landmarks[18], landmarks[17]) * 0.95

    # Thumb: tip (4) to index MCP (5) vs IP (3) to index MCP (5)
    # Rotation-invariant check inspired by the handsfree template
    thumb_extended = dist(landmarks[4], landmarks[5]) > dist(landmarks[3], landmarks[5]) * 1.05

    return thumb_extended and index_extended and middle_folded and ring_folded and pinky_folded

def camera_thread_worker():
    global running, hand_state, camera_error
    
    # 1. Download Model if missing
    if not os.path.exists(MODEL_FILE):
        print(f"Model {MODEL_FILE} not found. Downloading...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            print("Model download complete!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            camera_error = True
            return

    # 2. Init MediaPipe Hand Landmarker
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        camera_error = True
        return

    # 3. Open Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        camera_error = True
        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Steering smoothing state (EMA)
    STEER_SMOOTHING = 0.25  # Lower = smoother (0.0-1.0)
    smoothed_steering = 0.0

    while running:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        # Horizontal flip for mirrored experience
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Format image for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        try:
            results = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        raw_steering = 0.0
        drs_active_val = False
        hand_detected_val = False

        hand_L_lms = None
        hand_R_lms = None

        if results.hand_landmarks and len(results.hand_landmarks) >= 2:
            # Sort hands horizontally by landmark 9's x coordinate
            hands = []
            for idx in range(len(results.hand_landmarks)):
                lms = results.hand_landmarks[idx]
                hands.append((lms[9].x, lms))
            
            # Sort by x coordinate (ascending)
            # The hand on the left of the screen (smaller x) is the user's left hand
            # The hand on the right of the screen (larger x) is the user's right hand
            hands.sort(key=lambda x: x[0])
            
            hand_L_lms = hands[0][1]
            hand_R_lms = hands[1][1]
            hand_detected_val = True

            y_L = hand_L_lms[9].y
            y_R = hand_R_lms[9].y

            # Left moves up (y_L decreases) and Right moves down (y_R increases) -> turns right (positive steering)
            # Right moves up (y_R decreases) and Left moves down (y_L increases) -> turns left (negative steering)
            dy = y_R - y_L

            # Steering calculation with a dead zone of 0.02 and max steering difference of 0.18
            if abs(dy) < 0.02:
                raw_steering = 0.0
            else:
                sign = 1.0 if dy > 0 else -1.0
                val = (abs(dy) - 0.02) / (0.18 - 0.02)
                raw_steering = sign * min(1.0, val)

            # Check for DRS (both hands in finger guns)
            is_L_fg = check_finger_gun(hand_L_lms, 'Left')
            is_R_fg = check_finger_gun(hand_R_lms, 'Right')
            if is_L_fg and is_R_fg:
                drs_active_val = True

        # Apply exponential smoothing to steering
        smoothed_steering += (raw_steering - smoothed_steering) * STEER_SMOOTHING

        # Draw glowing PIP visuals
        pip_canvas = frame.copy()
        if hand_L_lms and hand_R_lms:
            l_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_L_lms]
            r_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_R_lms]
            
            draw_glowing_skeleton(pip_canvas, l_coords, HAND_CONNECTIONS, (0, 255, 255))
            draw_glowing_skeleton(pip_canvas, r_coords, HAND_CONNECTIONS, (0, 255, 255))

            # HUD on camera PIP
            steer_percent = int(smoothed_steering * 100)
            steer_text = f"Steer: {'R' if steer_percent > 0 else 'L' if steer_percent < 0 else ''} {abs(steer_percent)}%"
            
            is_L_fg = check_finger_gun(hand_L_lms, 'Left')
            is_R_fg = check_finger_gun(hand_R_lms, 'Right')
            gesture_text = f"Guns: L:{'YES' if is_L_fg else 'NO'} R:{'YES' if is_R_fg else 'NO'}"
            drs_text = "DRS: ON" if drs_active_val else "DRS: OFF"

            cv2.putText(pip_canvas, steer_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pip_canvas, gesture_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pip_canvas, drs_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if drs_active_val else (0, 0, 255),
                        2, cv2.LINE_AA)
        else:
            # Draw any single detected hand in gray
            if results.hand_landmarks:
                for idx in range(len(results.hand_landmarks)):
                    lms = results.hand_landmarks[idx]
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                    draw_glowing_skeleton(pip_canvas, coords, HAND_CONNECTIONS, (128, 128, 128))

            cv2.putText(pip_canvas, "BOTH HANDS REQUIRED", (40, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(pip_canvas, "PLACE BOTH HANDS IN CAMERA VIEW", (40, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2, cv2.LINE_AA)

        # Crop and resize to exactly 200x150
        pip_resized = cv2.resize(pip_canvas, (200, 150))
        pip_rgb = cv2.cvtColor(pip_resized, cv2.COLOR_BGR2RGB)

        with state_lock:
            hand_state.steering = smoothed_steering
            hand_state.drs_active = drs_active_val
            hand_state.camera_frame = pip_rgb
            hand_state.hand_detected = hand_detected_val

        time.sleep(0.01)

    cap.release()
    detector.close()

# ==============================================================================
#                             PARTICLE CLASSES
# ==============================================================================
class Particle:
    def __init__(self, x, y, dx, dy, color, size, life, decay_type='size'):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.size = size
        self.max_life = life
        self.life = life
        self.decay_type = decay_type

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.life -= 1

    def draw(self, surface):
        if self.life <= 0:
            return
        ratio = self.life / self.max_life
        if self.decay_type == 'size':
            curr_size = max(1, int(self.size * ratio))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), curr_size)
        else:
            # Alpha decay (requires drawing on a temporary surface with SRCALPHA)
            alpha = int(255 * ratio)
            temp = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp, (self.color[0], self.color[1], self.color[2], alpha), (self.size, self.size), self.size)
            surface.blit(temp, (int(self.x - self.size), int(self.y - self.size)))

class SpeedLine:
    def __init__(self):
        self.x = random.randint(50, SCREEN_WIDTH - 50)
        self.y = random.randint(-400, 0)
        self.length = random.randint(40, 90)
        self.speed = random.uniform(15, 25)

    def update(self, scroll_speed):
        self.y += self.speed + scroll_speed

    def draw(self, surface):
        pygame.draw.line(surface, (230, 240, 255), (self.x, self.y), (self.x, self.y + self.length), 2)

# ==============================================================================
#                            GAME DRAWING HELPERS
# ==============================================================================
def draw_f1_car(surface, x, y, color, is_player=False, drs_active=False):
    """Draws a detailed, procedurally rendered F1 racing car."""
    # Tires (black blocks)
    tire_w, tire_h = 10, 18
    # Front tires
    pygame.draw.rect(surface, (20, 20, 20), (x - 24, y - 30, tire_w, tire_h), border_radius=3)
    pygame.draw.rect(surface, (20, 20, 20), (x + 14, y - 30, tire_w, tire_h), border_radius=3)
    # Rear tires
    pygame.draw.rect(surface, (20, 20, 20), (x - 26, y + 15, tire_w + 2, tire_h + 4), border_radius=3)
    pygame.draw.rect(surface, (20, 20, 20), (x + 14, y + 15, tire_w + 2, tire_h + 4), border_radius=3)
    
    # Axles
    pygame.draw.line(surface, (80, 80, 80), (x - 20, y - 21), (x + 20, y - 21), 4)
    pygame.draw.line(surface, (80, 80, 80), (x - 20, y + 25), (x + 20, y + 25), 5)
    
    # Sidepods
    pygame.draw.rect(surface, color, (x - 20, y - 10, 8, 30), border_radius=4)
    pygame.draw.rect(surface, color, (x + 12, y - 10, 8, 30), border_radius=4)
    # Intake details
    pygame.draw.rect(surface, (10, 10, 10), (x - 19, y - 9, 6, 6), border_radius=1)
    pygame.draw.rect(surface, (10, 10, 10), (x + 13, y - 9, 6, 6), border_radius=1)
    
    # Main body/nose cone
    pygame.draw.polygon(surface, color, [
        (x, y - 45),       # nose tip
        (x - 10, y - 25),   # mid left
        (x - 12, y + 28),   # rear left
        (x + 12, y + 28),   # rear right
        (x + 10, y - 25)    # mid right
    ])
    
    # Center stripe
    pygame.draw.line(surface, (255, 255, 255), (x, y - 40), (x, y), 2)
    
    # Front Wing
    pygame.draw.rect(surface, color, (x - 25, y - 38, 50, 6), border_radius=1)
    pygame.draw.rect(surface, (10, 10, 10), (x - 26, y - 38, 2, 8)) # endplates
    pygame.draw.rect(surface, (10, 10, 10), (x + 24, y - 38, 2, 8))
    
    # Rear Wing & DRS
    wing_y = y + 28
    if is_player and drs_active:
        # DRS OPEN: Thin line, wing flap visually split open, green active neon light
        pygame.draw.rect(surface, (10, 10, 10), (x - 24, wing_y, 48, 3))
        pygame.draw.rect(surface, (50, 255, 50), (x - 24, wing_y - 2, 48, 2)) # glowing DRS light
    else:
        # DRS CLOSED: Thick red wing wing board
        pygame.draw.rect(surface, color, (x - 24, wing_y, 48, 8), border_radius=1)
        
    # Rear wing endplates
    pygame.draw.rect(surface, (10, 10, 10), (x - 25, wing_y - 2, 2, 12))
    pygame.draw.rect(surface, (10, 10, 10), (x + 23, wing_y - 2, 2, 12))
    
    # Helmet
    pygame.draw.circle(surface, (20, 20, 20), (x, y - 3), 7)
    visor_color = (255, 215, 0) if is_player else (180, 180, 180) # yellow visor for player
    pygame.draw.arc(surface, visor_color, (x - 5, y - 7, 10, 10), 0.2, 3.0, 3)

# ==============================================================================
#                                 MAIN PROGRAM
# ==============================================================================
def main():
    global running, camera_error
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("f1 game")
    clock = pygame.time.Clock()
    
    # Safe Font Loader helper
    def get_font(name, size, bold=False):
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except:
            return pygame.font.Font(None, size)
            
    font_large = get_font('Arial', 64, bold=True)
    font_medium = get_font('Arial', 32, bold=True)
    font_small = get_font('Arial', 18, bold=True)
    font_digital = get_font('Courier New', 28, bold=True)
    
    # Start Hand Tracking Thread
    camera_thread = threading.Thread(target=camera_thread_worker)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Road curvature lists and positions
    road_centers = [SCREEN_WIDTH // 2] * NUM_SLICES
    scroll_accumulator = 0.0
    scroll_distance = 0.0
    current_curvature = 0.0
    target_curvature = 0.0
    track_timer = 0
    
    # Game Variables
    state = "START_SCREEN" # START_SCREEN, PLAYING, GAME_OVER
    player_x = SCREEN_WIDTH // 2
    player_steer_speed = 10.0
    
    current_speed = 0.0
    base_speed = 11.0 # ~240 km/h
    drs_speed = 18.0  # ~340 km/h
    offroad_speed = 5.0 # ~110 km/h
    
    score = 0.0
    highscore = 0
    
    # Particles and lists
    particles = []
    speed_lines = []
    traffic_cars = []
    
    game_over_timer = 0
    screen_shake_x = 0
    screen_shake_y = 0
    
    # Save High Score to local file
    highscore_file = "highscore.txt"
    if os.path.exists(highscore_file):
        try:
            with open(highscore_file, "r") as f:
                highscore = int(f.read().strip())
        except:
            pass

    def reset_game():
        nonlocal player_x, current_speed, score, road_centers, scroll_accumulator, scroll_distance
        nonlocal current_curvature, target_curvature, particles, speed_lines, traffic_cars, game_over_timer
        player_x = SCREEN_WIDTH // 2
        current_speed = 0.0
        score = 0.0
        road_centers = [SCREEN_WIDTH // 2] * NUM_SLICES
        scroll_accumulator = 0.0
        scroll_distance = 0.0
        current_curvature = 0.0
        target_curvature = 0.0
        particles = []
        speed_lines = []
        traffic_cars = []
        game_over_timer = 0
        
    def get_road_center_at_y(y):
        """Calculates interpolated road center at a specific y coordinate."""
        slice_idx = int((y - scroll_accumulator) / SLICE_H)
        if slice_idx < 0:
            return road_centers[0]
        elif slice_idx >= len(road_centers) - 1:
            return road_centers[-1]
        else:
            # Interpolate for ultra smooth movement
            y1 = slice_idx * SLICE_H + scroll_accumulator
            y2 = (slice_idx + 1) * SLICE_H + scroll_accumulator
            c1 = road_centers[slice_idx]
            c2 = road_centers[slice_idx + 1]
            ratio = (y - y1) / SLICE_H
            return c1 + (c2 - c1) * ratio

    # Game Loop
    game_running = True
    while game_running:
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
                
        # Read user controls (Gestures with Keyboard Override Fallback)
        keys = pygame.key.get_pressed()
        keyboard_steer = 0.0
        keyboard_drs = False
        keyboard_active = False
        
        if keys[pygame.K_LEFT]:
            keyboard_steer = -1.0
            keyboard_active = True
        elif keys[pygame.K_RIGHT]:
            keyboard_steer = 1.0
            keyboard_active = True
            
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            keyboard_drs = True
            keyboard_active = True

        # Extract values from thread safely
        with state_lock:
            cam_frame = hand_state.camera_frame
            hand_steer = hand_state.steering
            hand_drs = hand_state.drs_active
            hand_detected = hand_state.hand_detected

        # Fallback decision
        if keyboard_active or not hand_detected:
            steer = keyboard_steer
            drs_input = keyboard_drs
            control_type = "KEYBOARD"
        else:
            steer = hand_steer
            drs_input = hand_drs
            control_type = "HAND"
            
        # Initialize loop-scoped variables
        drs_active = False
        drs_blocked = False
        is_offroad = False
            
        # Draw on an intermediate surface for screen shake
        game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # ==============================================================================
        #                             STATE: START SCREEN & PLAYING
        # ==============================================================================
        if state in ["START_SCREEN", "PLAYING"]:
            if state == "PLAYING":
                # 1. Curve and Curvature logic
                track_timer += 1
                # Decide track shifts
                if track_timer > random.randint(120, 240):
                    track_timer = 0
                    # Self correction if road moves near the edges
                    if road_centers[0] < 350:
                        target_curvature = random.uniform(0.5, 2.5)
                    elif road_centers[0] > 650:
                        target_curvature = random.uniform(-2.5, -0.5)
                    else:
                        # Random turn curvature
                        target_curvature = random.choice([0.0, 0.0, random.uniform(1.2, 2.6), random.uniform(-2.6, -1.2)])
                
                # Smoothly transition curvature
                current_curvature += (target_curvature - current_curvature) * 0.04
                
                # 2. Physics & Off-road penalty
                road_center_player = get_road_center_at_y(PLAYER_Y)
                is_offroad = abs(player_x - road_center_player) > (ROAD_WIDTH / 2 - 20)
                
                drs_blocked = is_offroad or (abs(current_curvature) > 1.8) # DRS blocked if offroad or in heavy curve
                drs_active = drs_input and not drs_blocked
                
                # Interpolate target speed
                if is_offroad:
                    target_speed = offroad_speed
                elif drs_active:
                    target_speed = drs_speed
                else:
                    target_speed = base_speed
                    
                current_speed += (target_speed - current_speed) * 0.08
                
                # Adjust score
                score += current_speed * 0.08
                
                # Screen shake when driving at high speed (DRS active)
                if drs_active and current_speed > base_speed + 2:
                    screen_shake_x = random.randint(-3, 3)
                    screen_shake_y = random.randint(-3, 3)
                else:
                    screen_shake_x = 0
                    screen_shake_y = 0
                    
                # 3. Road scrolling
                scroll_accumulator += current_speed
                scroll_distance += current_speed
                
                while scroll_accumulator >= SLICE_H:
                    scroll_accumulator -= SLICE_H
                    road_centers.pop()
                    new_center = road_centers[0] + current_curvature
                    road_centers.insert(0, new_center)
            else:
                # START_SCREEN state: standby with road visible but not moving
                current_speed = 0.0
                drs_active = False
                is_offroad = False
                score = 0.0
                screen_shake_x = 0
                screen_shake_y = 0
                
                # Trigger Start
                if drs_input:
                    state = "PLAYING"
                    reset_game()
                    
            # 4. Move Player Car (active in both states so steering can be tested)
            player_x += steer * player_steer_speed
            # Bound inside screen
            player_x = max(40, min(SCREEN_WIDTH - 40, player_x))
            
            if state == "PLAYING":
                # 5. Spawning Traffic Cars
                if len(traffic_cars) < 3 and random.randint(1, 100) == 1:
                    # Avoid spawning immediately next to each other
                    lane = random.choice([-1, 0, 1])
                    lane_conflict = False
                    for car in traffic_cars:
                        if car['lane'] == lane and car['y'] < 200:
                            lane_conflict = True
                            break
                    if not lane_conflict:
                        traffic_cars.append({
                            'lane': lane,
                            'y': -100,
                            'speed': random.uniform(4.0, 7.5), # Moves slower than road
                            'color': random.choice([
                                (30, 144, 255),  # Blue
                                (255, 140, 0),   # Orange
                                (255, 215, 0),   # Gold
                                (0, 200, 100),   # Cyan
                                (200, 200, 200)  # Silver
                            ])
                        })

                # Update Traffic Cars
                for car in traffic_cars[:]:
                    # Moves relative to player speed
                    car['y'] += (current_speed - car['speed'])
                    if car['y'] > SCREEN_HEIGHT + 100 or car['y'] < -200:
                        traffic_cars.remove(car)
                        
                # 6. Spawning Particles
                # DRS exhaust plasma
                if drs_active:
                    for _ in range(2):
                        particles.append(Particle(
                            x=player_x + random.randint(-4, 4),
                            y=PLAYER_Y + 32,
                            dx=random.uniform(-1.5, 1.5),
                            dy=random.uniform(4.0, 8.0),
                            color=(0, random.randint(190, 255), 255), # Neon Blue
                            size=random.randint(4, 7),
                            life=random.randint(12, 22),
                            decay_type='size'
                        ))
                
                # Offroad tire dust sparks
                if is_offroad and current_speed > 1.0:
                    for tire_x_offset in [-20, 20]:
                        particles.append(Particle(
                            x=player_x + tire_x_offset + random.randint(-3, 3),
                            y=PLAYER_Y + 25,
                            dx=random.uniform(-3.5, 3.5),
                            dy=random.uniform(2.0, 5.0),
                            color=(139, 90, 43), # Brown dust
                            size=random.randint(3, 6),
                            life=random.randint(15, 30),
                            decay_type='size'
                        ))
                        
                # Manage speed lines in DRS
                if drs_active and len(speed_lines) < 8:
                    speed_lines.append(SpeedLine())
                    
            # Update Particles (common)
            for p in particles[:]:
                p.update()
                if p.life <= 0:
                    particles.remove(p)
                    
            for line in speed_lines[:]:
                line.update(current_speed)
                if line.y > SCREEN_HEIGHT:
                    speed_lines.remove(line)
                    
            # 7. Render Road & Grass Environment (Slices-based)
            y_offset = scroll_accumulator
            for i in range(NUM_SLICES - 1):
                y1 = i * SLICE_H + y_offset
                y2 = (i + 1) * SLICE_H + y_offset
                
                c1 = road_centers[i]
                c2 = road_centers[i + 1]
                
                # Grass background alternating bands (simulates motion)
                scroll_idx = int(scroll_distance / SLICE_H)
                is_band = ((i + scroll_idx) % 8 < 4)
                grass_color = (15, 110, 30) if is_band else (10, 85, 20)
                pygame.draw.rect(game_surface, grass_color, (0, int(y1), SCREEN_WIDTH, SLICE_H))
                
                # Asphalt Main Road
                pts_road = [
                    (c1 - ROAD_WIDTH / 2, y1),
                    (c1 + ROAD_WIDTH / 2, y1),
                    (c2 + ROAD_WIDTH / 2, y2),
                    (c2 - ROAD_WIDTH / 2, y2)
                ]
                pygame.draw.polygon(game_surface, (45, 45, 45), pts_road)
                
                # Striped Curbstones (Red / White)
                is_curb_band = ((i + scroll_idx) % 4 < 2)
                curb_color = (220, 0, 0) if is_curb_band else (240, 240, 240)
                
                # Left Curb
                pts_l_curb = [
                    (c1 - ROAD_WIDTH / 2 - CURB_WIDTH, y1),
                    (c1 - ROAD_WIDTH / 2, y1),
                    (c2 - ROAD_WIDTH / 2, y2),
                    (c2 - ROAD_WIDTH / 2 - CURB_WIDTH, y2)
                ]
                pygame.draw.polygon(game_surface, curb_color, pts_l_curb)
                # Right Curb
                pts_r_curb = [
                    (c1 + ROAD_WIDTH / 2, y1),
                    (c1 + ROAD_WIDTH / 2 + CURB_WIDTH, y1),
                    (c2 + ROAD_WIDTH / 2 + CURB_WIDTH, y2),
                    (c2 + ROAD_WIDTH / 2, y2)
                ]
                pygame.draw.polygon(game_surface, curb_color, pts_r_curb)
                
                # Central white dashed lane divider
                if (i + scroll_idx) % 6 < 3:
                    pygame.draw.line(game_surface, (235, 235, 235), (c1, y1), (c2, y2), 3)

            # 8. Render Traffic Cars
            for car in traffic_cars:
                tx = get_road_center_at_y(car['y']) + car['lane'] * (ROAD_WIDTH / 3.0)
                draw_f1_car(game_surface, int(tx), int(car['y']), car['color'], is_player=False)
                
            # 9. Render Particles & Speed Lines
            for p in particles:
                p.draw(game_surface)
            for line in speed_lines:
                line.draw(game_surface)

            # 10. Render Player Car
            draw_f1_car(game_surface, int(player_x), PLAYER_Y, (230, 20, 20), is_player=True, drs_active=drs_active)
            
            # Collisions Check (only when playing)
            if state == "PLAYING":
                player_rect = pygame.Rect(player_x - 16, PLAYER_Y - 30, 32, 60)
                for car in traffic_cars:
                    tx = get_road_center_at_y(car['y']) + car['lane'] * (ROAD_WIDTH / 3.0)
                    car_rect = pygame.Rect(tx - 16, car['y'] - 30, 32, 60)
                    if player_rect.colliderect(car_rect):
                        # Trigger explosion particles
                        for _ in range(80):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(3, 11)
                            particles.append(Particle(
                                x=player_x,
                                y=PLAYER_Y - 10,
                                dx=math.cos(angle) * speed,
                                dy=math.sin(angle) * speed,
                                color=random.choice([(255, 255, 255), (255, 200, 0), (255, 80, 0), (200, 0, 0)]),
                                size=random.randint(5, 10),
                                life=random.randint(30, 60),
                                decay_type='size'
                            ))
                        state = "GAME_OVER"
                        game_over_timer = 0
                        # Save High score
                        if int(score) > highscore:
                            highscore = int(score)
                            try:
                                with open(highscore_file, "w") as f:
                                    f.write(str(highscore))
                            except:
                                pass
                        break

            # 11. Draw Score Banner at top center
            score_surface = pygame.Surface((360, 40), pygame.SRCALPHA)
            score_surface.fill((10, 10, 20, 180))
            pygame.draw.rect(score_surface, (255, 255, 255), (0, 0, 360, 40), 1, border_radius=5)
            score_txt = font_digital.render(f"DIST: {int(score):04d}m  BEST: {highscore:04d}m", True, (255, 255, 255))
            score_surface.blit(score_txt, (20, 6))
            game_surface.blit(score_surface, (SCREEN_WIDTH // 2 - 180, 15))
            
            # Start Screen overlay (Teeny Text Box)
            if state == "START_SCREEN":
                box_w, box_h = 380, 70
                box_surface = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
                box_surface.fill((10, 10, 20, 220)) # semi-transparent dark back
                pygame.draw.rect(box_surface, (0, 255, 255), (0, 0, box_w, box_h), 2, border_radius=8)
                
                # Flashing text
                txt_color = (0, 255, 255) if int(time.time() * 2) % 2 == 0 else (240, 240, 240)
                msg_txt1 = font_medium.render("FINGER GUNS TO START", True, txt_color)
                msg_txt2 = font_small.render("OR PRESS SPACEBAR", True, (150, 150, 160))
                
                box_surface.blit(msg_txt1, (box_w // 2 - msg_txt1.get_width() // 2, 8))
                box_surface.blit(msg_txt2, (box_w // 2 - msg_txt2.get_width() // 2, 40))
                
                game_surface.blit(box_surface, (SCREEN_WIDTH // 2 - box_w // 2, SCREEN_HEIGHT // 2 - 100))

        # ==============================================================================
        #                             STATE: GAME OVER
        # ==============================================================================
        elif state == "GAME_OVER":
            game_over_timer += 1
            
            # Still draw road environment static under game over filter
            # Draw road in center
            game_surface.fill((10, 70, 20)) # dimmed green
            pygame.draw.rect(game_surface, (25, 25, 25), (SCREEN_WIDTH//2 - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
            pygame.draw.rect(game_surface, (120, 0, 0), (SCREEN_WIDTH//2 - ROAD_WIDTH//2 - 5, 0, 5, SCREEN_HEIGHT))
            pygame.draw.rect(game_surface, (120, 0, 0), (SCREEN_WIDTH//2 + ROAD_WIDTH//2, 0, 5, SCREEN_HEIGHT))
            
            # Draw explosion particles continuing to fly
            for p in particles:
                p.update()
                p.draw(game_surface)
                
            # Draw player car remnants (slightly blackened)
            draw_f1_car(game_surface, int(player_x), PLAYER_Y, (80, 10, 10), is_player=True, drs_active=False)
            
            # Game Over translucent overlay
            over_panel = pygame.Surface((500, 300), pygame.SRCALPHA)
            over_panel.fill((15, 0, 0, 220))
            pygame.draw.rect(over_panel, (255, 50, 50), (0, 0, 500, 300), 2, border_radius=10)
            
            go_text = font_large.render("GAME OVER", True, (255, 40, 40))
            over_panel.blit(go_text, (250 - go_text.get_width()//2, 30))
            
            res_txt = font_medium.render(f"Distance: {int(score)} meters", True, (255, 255, 255))
            over_panel.blit(res_txt, (250 - res_txt.get_width()//2, 120))
            
            best_txt = font_medium.render(f"Personal Best: {highscore}m", True, (255, 215, 0))
            over_panel.blit(best_txt, (250 - best_txt.get_width()//2, 160))
            
            game_surface.blit(over_panel, (SCREEN_WIDTH//2 - 250, 160))
            
            # Prompt user to restart with gesture (after a short cooldown)
            if game_over_timer > 90: # ~1.5s
                if int(time.time() * 2) % 2 == 0:
                    restart_txt = font_medium.render("MAKE FINGER GUNS TO RESTART", True, (0, 255, 255))
                    game_surface.blit(restart_txt, (SCREEN_WIDTH//2 - restart_txt.get_width()//2, 510))
                
                # Check for restart trigger
                if drs_input:
                    state = "PLAYING"
                    reset_game()

        # ==============================================================================
        #                               HUD OVERLAY
        # ==============================================================================
        # Render a unified transparent HUD at the bottom of the screen
        hud_surface = pygame.Surface((SCREEN_WIDTH, 100), pygame.SRCALPHA)
        hud_surface.fill((10, 10, 20, 195)) # semi-transparent back
        
        # 1. Circular Speedometer Dial
        # Convert speed to virtual km/h
        speed_kmh = int(current_speed * 18.8) # maps ~18 speed to ~340 km/h
        speed_ratio = min(1.0, current_speed / drs_speed)
        
        # Base arc (grey)
        pygame.draw.arc(hud_surface, (60, 60, 70), (40, -40, 140, 140), 3.14, 0.0, 5)
        # Active speed arc (cyan or green in DRS)
        arc_color = (50, 255, 50) if (drs_active and state == "PLAYING") else (0, 255, 255)
        if is_offroad and state == "PLAYING": arc_color = (255, 100, 0)
        end_angle = 3.14 - (speed_ratio * 3.14)
        if speed_ratio > 0.01:
            pygame.draw.arc(hud_surface, arc_color, (40, -40, 140, 140), end_angle, 3.14, 7)
            
        # Draw Speed text
        speed_val_txt = font_digital.render(f"{speed_kmh:03d}", True, (255, 255, 255))
        unit_txt = font_small.render("km/h", True, (150, 160, 170))
        hud_surface.blit(speed_val_txt, (110 - speed_val_txt.get_width()//2, 35))
        hud_surface.blit(unit_txt, (110 - unit_txt.get_width()//2, 65))
        
        # 2. DRS Status box (Center)
        drs_panel_x = SCREEN_WIDTH // 2 - 110
        if state == "PLAYING":
            if drs_active:
                drs_col = (50, 255, 50) # Glowing green
                drs_lbl = "DRS ACTIVE"
                # Flashing text
                if int(time.time() * 6) % 2 == 0: drs_col = (20, 150, 20)
            elif drs_blocked:
                drs_col = (255, 60, 60) # Blocked/Red (e.g. offroad)
                drs_lbl = "DRS BLOCKED"
            else:
                drs_col = (0, 220, 255) # Cyan (available)
                drs_lbl = "DRS READY"
        else:
            drs_col = (100, 100, 110)
            drs_lbl = "DRS CLOSED"
            
        pygame.draw.rect(hud_surface, drs_col, (drs_panel_x, 30, 220, 48), 2, border_radius=4)
        drs_txt = font_medium.render(drs_lbl, True, drs_col)
        hud_surface.blit(drs_txt, (SCREEN_WIDTH // 2 - drs_txt.get_width()//2, 38))
        
        # 3. Steering Horizontal Slider
        steer_x_base = SCREEN_WIDTH - 240
        pygame.draw.line(hud_surface, (100, 100, 110), (steer_x_base, 55), (steer_x_base + 180, 55), 4)
        pygame.draw.line(hud_surface, (255, 255, 255), (steer_x_base + 90, 48), (steer_x_base + 90, 62), 2) # Center line
        
        # Draw target steering position
        steer_pos = int(steer * 90) # maps -1..1 to -90..90
        pygame.draw.circle(hud_surface, arc_color, (steer_x_base + 90 + steer_pos, 55), 8)
        
        steer_lbl = font_small.render("STEERING INPUT", True, (150, 160, 170))
        hud_surface.blit(steer_lbl, (steer_x_base + 90 - steer_lbl.get_width()//2, 15))
        
        l_lbl = font_small.render("L", True, (120, 120, 130))
        r_lbl = font_small.render("R", True, (120, 120, 130))
        hud_surface.blit(l_lbl, (steer_x_base - 18, 45))
        hud_surface.blit(r_lbl, (steer_x_base + 192, 45))
        
        # Draw control type text indicator
        ctrl_lbl = font_small.render(f"CONTROL: {control_type}", True, (255, 215, 0))
        hud_surface.blit(ctrl_lbl, (SCREEN_WIDTH // 2 - ctrl_lbl.get_width()//2, 5))
        
        # Draw warning label if off-road
        if state == "PLAYING" and is_offroad:
            warn_txt = font_medium.render("OFF TRACK - SLOW DOWN!", True, (255, 80, 0))
            game_surface.blit(warn_txt, (SCREEN_WIDTH // 2 - warn_txt.get_width()//2, 100))

        # Assemble game_surface
        game_surface.blit(hud_surface, (0, SCREEN_HEIGHT - 100))

        # ==============================================================================
        #                           CAMERA PIP VIEWPORT
        # ==============================================================================
        # Render the resized webcam feed overlay in top-right corner
        if cam_frame is not None:
            try:
                # Convert raw camera frame array to Pygame Surface
                pip_surface = pygame.image.frombuffer(cam_frame.tobytes(), (200, 150), 'RGB')
                
                # Draw neon cyan borders around camera feed
                pygame.draw.rect(game_surface, (0, 255, 255), (SCREEN_WIDTH - 213, 12, 204, 154), 2, border_radius=4)
                game_surface.blit(pip_surface, (SCREEN_WIDTH - 211, 14))
                
                # Label overlay
                lbl = font_small.render("GESTURE CAMERA", True, (0, 255, 255))
                pygame.draw.rect(game_surface, (10, 10, 20), (SCREEN_WIDTH - 211, 14, 140, 22))
                game_surface.blit(lbl, (SCREEN_WIDTH - 206, 16))
            except Exception as e:
                print(f"Error drawing PIP: {e}")

        # Blit the entire gameplay surface with Screen Shake offsets to the screen
        screen.fill((0, 0, 0))
        screen.blit(game_surface, (screen_shake_x, screen_shake_y))
        
        pygame.display.flip()
        clock.tick(60)

    # Graceful Shutdown
    running = False
    camera_thread.join(timeout=1.0)
    pygame.quit()

if __name__ == "__main__":
    main()
