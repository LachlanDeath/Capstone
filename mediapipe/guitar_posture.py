import os
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Keep guitar_infer.py and guitar_landmarks.py in the same folder as this file.
# Set your Roboflow key in the terminal first (never type it into this file):
#   $env:ROBOFLOW_API_KEY = "your-key"
from guitar_infer import detect_guitar, get_api_key
from guitar_landmarks import draw_guitar, guitar_from_result

# Model files are looked up next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Guitar settings (Roboflow, runs in a background thread)
# ---------------------------------------------------------------
SEND_WIDTH = 640         # frames are shrunk to this width before upload (try 320-640)
MIN_INTERVAL = 0.05      # low when testing tracking, high otherwise (API call frequency)
MIN_CONFIDENCE = 0.5     # same idea as the `visibility > 0.5` check for the body

# ---------------------------------------------------------------
# Pose landmark indices (odd = subject's left, even = subject's right)
# 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips
# ---------------------------------------------------------------
KEYPOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

CONNECTIONS = [
    (7, 0), (0, 8),          # ears to nose
    (11, 12),                # shoulder line
    (23, 24),                # hip line
    (11, 23), (12, 24),      # torso sides
    (11, 13), (13, 15),      # left arm
    (12, 14), (14, 16),      # right arm
]

LEFT_IDS = {7, 11, 13, 15, 23}
RIGHT_IDS = {8, 12, 14, 16, 24}

# BGR colors
LEFT_COLOR = (255, 128, 0)      # blue-ish orange
RIGHT_COLOR = (0, 255, 0)       # green
CENTER_COLOR = (200, 200, 200)  # gray
JOINT_COLOR = (0, 0, 255)       # red


def side_color(a, b):
    if a in LEFT_IDS and b in LEFT_IDS:
        return LEFT_COLOR
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return RIGHT_COLOR
    return CENTER_COLOR


# Hand landmarks: 0 wrist, then 4 joints per finger (thumb -> pinky)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm edge
]


# ---------------------------------------------------------------
# Guitar worker: sends frames to Roboflow without freezing the video
# ---------------------------------------------------------------
class GuitarWorker(threading.Thread):
    """Sends the newest frame to Roboflow in the background and stores the latest result."""

    def __init__(self, api_key, min_interval=MIN_INTERVAL):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.pending = None      # (small_frame, scale) waiting to be sent
        self.result = None
        self.latency = None
        self.error = None
        self.running = True

    def submit(self, frame):
        h, w = frame.shape[:2]
        scale = w / SEND_WIDTH if w > SEND_WIDTH else 1.0
        small = cv2.resize(frame, (int(w / scale), int(h / scale))) if scale != 1.0 else frame.copy()
        with self.lock:
            self.pending = (small, scale)   # overwrite: we only ever care about the newest frame

    def latest(self):
        with self.lock:
            return self.result, self.latency, self.error

    def run(self):
        while self.running:
            with self.lock:
                job, self.pending = self.pending, None
            if job is None:
                time.sleep(0.01)
                continue

            small, scale = job
            t0 = time.time()
            try:
                result = detect_guitar(small, api_key=self.api_key)
                # Scale coordinates back up to the full-size camera frame
                for pred in result.get("predictions", []):
                    for k in ("x", "y", "width", "height"):
                        if k in pred:
                            pred[k] *= scale
                    for p in pred.get("points", []):
                        p["x"] *= scale
                        p["y"] *= scale
                with self.lock:
                    self.result, self.error = result, None
                    self.latency = time.time() - t0
            except (Exception, SystemExit) as e:
                with self.lock:
                    self.error = str(e) or type(e).__name__

            time.sleep(max(0.0, self.min_interval - (time.time() - t0)))


api_key = get_api_key()   # exits with a clear message if ROBOFLOW_API_KEY isn't set

# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ---------------------------------------------------------------
# Camera
# ---------------------------------------------------------------
cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

guitar_worker = GuitarWorker(api_key)
guitar_worker.start()

frame_count = 0
fps_start = time.time()

start = time.time()
last_ts = -1
last_guitar_result, guitar_pts, guitar_conf = None, None, 0.0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    # Mirror the image so it behaves like a mirror and Left/Right labels are correct.
    # Everything below (MediaPipe AND Roboflow) works on this same mirrored frame.
    frame = cv2.flip(frame, 1)

    # Hand the clean frame to the guitar thread BEFORE anything is drawn on it
    # (submit keeps its own copy, so later drawing never reaches the model)
    guitar_worker.submit(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Timestamps must be ints and strictly increasing
    timestamp_ms = int((time.time() - start) * 1000)
    if timestamp_ms <= last_ts:
        timestamp_ms = last_ts + 1
    last_ts = timestamp_ms

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame.shape[:2]

    # ----- Pose (upper body, both arms) -----
    if result.pose_landmarks:
        lm = result.pose_landmarks[0]  # first detected person

        pose_pts = {}
        for i in KEYPOINTS:
            if lm[i].visibility > 0.5:
                pose_pts[i] = (int(lm[i].x * w), int(lm[i].y * h))

        for a, b in CONNECTIONS:
            if a in pose_pts and b in pose_pts:
                cv2.line(frame, pose_pts[a], pose_pts[b], side_color(a, b), 2)
        for p in pose_pts.values():
            cv2.circle(frame, p, 5, JOINT_COLOR, -1)

    # ----- Hands (both) -----
    for hand, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        color = RIGHT_COLOR if label == "Right" else LEFT_COLOR

        hand_pts = [(int(p.x * w), int(p.y * h)) for p in hand]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, hand_pts[a], hand_pts[b], color, 2)
        for p in hand_pts:
            cv2.circle(frame, p, 3, JOINT_COLOR, -1)

    # ----- Guitar (from the background thread) -----
    guitar_result, guitar_latency, guitar_error = guitar_worker.latest()

    # Only redo the landmark math when a new Roboflow answer has actually arrived
    if guitar_result is not last_guitar_result:
        guitar_pts, guitar_conf = guitar_from_result(guitar_result, MIN_CONFIDENCE)
        last_guitar_result = guitar_result

    if guitar_pts:
        draw_guitar(frame, guitar_pts)

    if guitar_error:
        status, status_color = f"ERROR: {guitar_error[:70]}", (0, 0, 255)
    elif guitar_latency is None:
        status, status_color = "guitar: waiting for first result...", (0, 255, 255)
    elif guitar_pts:
        status = f"guitar: {len(guitar_pts)} landmarks   conf {guitar_conf:.2f}   round trip {guitar_latency * 1000:.0f} ms"
        status_color = (0, 255, 0)
    else:
        status, status_color = f"guitar: not found   round trip {guitar_latency * 1000:.0f} ms", (0, 165, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    frame_count += 1
    if time.time() - fps_start >= 2.0:
        print(f"Actual FPS: {frame_count / (time.time() - fps_start):.1f}")
        frame_count = 0
        fps_start = time.time()

    cv2.imshow("Pose + Guitar", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

guitar_worker.running = False
cap.release()
cv2.destroyAllWindows()