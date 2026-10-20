import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Model files are looked up next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Cello settings (local YOLO pose model, runs on your GPU)
# ---------------------------------------------------------------
# Copy runs\pose\runs\pose\cello\weights\best.pt next to this script and name it cello_best.pt
# (the other locations below are tried as well)
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "cello_best.pt"),
    os.path.join(BASE_DIR, "..", "runs", "pose", "runs", "pose", "cello", "weights", "best.pt"),
    os.path.join(BASE_DIR, "..", "runs", "pose", "cello", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "pose", "runs", "pose", "cello", "weights", "best.pt"),
]
BOX_CONF = 0.4           # minimum detection confidence for a cello
KPT_CONF = 0.5           # same idea as the `visibility > 0.5` check for the body
HAND_FRACTION = 0.14     # first-position hand: this fraction of the way from nut to bridge
YOLO_SIZE = 640          # try 480 or 416 if you need more speed (cello is large in the frame)
CELLO_THREAD = True      # run the cello model in a background thread so the video never waits for it
CELLO_ENABLED = True     # False = skip the cello completely (use this to see how fast MediaPipe is on its own)
CELLO_MAX_HZ = 15        # cello model runs at most this many times per second (lower = more CPU left for MediaPipe)

# ---------------------------------------------------------------
# Recording (press R): countdown, then a fixed-length video of exactly what is drawn on screen
# ---------------------------------------------------------------
RECORD_SECONDS = 10                # seconds of video to record
RECORD_COUNTDOWN = 3                 # seconds to get into position after pressing R (not recorded)
RECORD_FPS = 25                      # frames are repeated/skipped as needed so the video plays at real speed
RECORD_DIR = os.path.join(BASE_DIR, "recordings")


def open_writer(width, height):
    os.makedirs(RECORD_DIR, exist_ok=True)
    path = os.path.join(RECORD_DIR, time.strftime("cello_%Y%m%d_%H%M%S.mp4"))
    for codec in ("avc1", "mp4v"):   # avc1 (H.264) plays everywhere; mp4v is the fallback
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), RECORD_FPS, (width, height))
        if writer.isOpened():
            print(f"Recording to {path} ({codec})")
            return writer, path
        writer.release()
    return None, path


# Keypoint order the model was trained with
FROG, TIP, NUT, BRIDGE = 0, 1, 2, 3

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
CELLO_COLOR = (0, 255, 255)     # yellow  (nut -> bridge)
BOW_COLOR = (0, 165, 255)       # orange  (frog -> tip)


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
# Cello detection (local YOLO pose model)
# ---------------------------------------------------------------
def load_cello_model():
    path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        sys.exit("Cello model not found. Copy best.pt next to this script and name it cello_best.pt.\nLooked in:\n  "
                 + "\n  ".join(os.path.abspath(p) for p in MODEL_CANDIDATES))
    try:
        import torch
        torch.set_num_threads(1)          # stops torch's CPU threads from fighting with MediaPipe
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    model = YOLO(path)
    for _ in range(3):                                                                              # warm-up
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=YOLO_SIZE, device=device, verbose=False)
    print("Cello model:", os.path.abspath(path), "| device:", device)
    return model, device


def detect_cello(model, device, frame):
    """Returns ({'frog','tip','nut','bridge','hand': (x, y)} for the points that are confident, box confidence)."""
    res = model.predict(frame, imgsz=YOLO_SIZE, conf=BOX_CONF, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None, 0.0
    box_conf = res.boxes.conf.cpu().numpy()
    best = int(box_conf.argmax())
    xy = res.keypoints.xy.cpu().numpy()[best]
    kc = res.keypoints.conf
    kc = kc.cpu().numpy()[best] if kc is not None else np.ones(len(xy))
    names = {FROG: "frog", TIP: "tip", NUT: "nut", BRIDGE: "bridge"}
    pts = {names[i]: (int(xy[i][0]), int(xy[i][1])) for i in names if kc[i] > KPT_CONF}
    if "nut" in pts and "bridge" in pts:
        nx, ny = pts["nut"]
        bx, by = pts["bridge"]
        pts["hand"] = (int(nx + HAND_FRACTION * (bx - nx)), int(ny + HAND_FRACTION * (by - ny)))
    return pts, float(box_conf[best])


def draw_cello(frame, pts):
    if "nut" in pts and "bridge" in pts:
        cv2.line(frame, pts["nut"], pts["bridge"], CELLO_COLOR, 2)
    if "frog" in pts and "tip" in pts:
        cv2.line(frame, pts["frog"], pts["tip"], BOW_COLOR, 2)
    for name, p in pts.items():
        if name == "hand":
            cv2.circle(frame, p, 9, JOINT_COLOR, 2)       # ring = where the left hand should be in first position
        else:
            cv2.circle(frame, p, 5, JOINT_COLOR, -1)


class CelloWorker(threading.Thread):
    """Runs the cello model on the newest frame in the background and keeps the latest answer."""

    def __init__(self, model, device):
        super().__init__(daemon=True)
        self.model, self.device = model, device
        self.lock = threading.Lock()
        self.pending = None
        self.pts, self.conf, self.ms, self.error = None, 0.0, 0.0, None
        self.running = True

    def submit(self, frame):
        with self.lock:
            self.pending = frame.copy()      # overwrite: only the newest frame matters

    def latest(self):
        with self.lock:
            return self.pts, self.conf, self.ms, self.error

    def run(self):
        while self.running:
            with self.lock:
                frame, self.pending = self.pending, None
            if frame is None:
                time.sleep(0.002)
                continue
            t0 = time.time()
            try:
                pts, conf = detect_cello(self.model, self.device, frame)
                with self.lock:
                    self.pts, self.conf, self.ms, self.error = pts, conf, (time.time() - t0) * 1000, None
            except Exception as e:
                with self.lock:
                    self.error = str(e) or type(e).__name__
            time.sleep(max(0.0, 1.0 / CELLO_MAX_HZ - (time.time() - t0)))


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
cello_model, cello_device, cello_worker = None, "off", None
if CELLO_ENABLED:
    cello_model, cello_device = load_cello_model()
    cello_worker = CelloWorker(cello_model, cello_device)
    if CELLO_THREAD:
        cello_worker.start()

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

frame_count = 0
fps_start = time.time()

start = time.time()
last_ts = -1
cello_pts, cello_conf, cello_ms, cello_error = None, 0.0, 0.0, None
t_mp_sum = 0.0
rec_state, rec_start, rec_t0, rec_written, writer, rec_path, saved_msg, saved_until = None, 0.0, 0.0, 0, None, "", "", 0.0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    # Mirror the image so it behaves like a mirror and Left/Right labels are correct.
    # MediaPipe AND the cello model both work on this same mirrored frame.
    frame = cv2.flip(frame, 1)

    # Cello model gets the clean frame (nothing drawn on it yet)
    if CELLO_ENABLED and CELLO_THREAD:
        cello_worker.submit(frame)            # keeps its own copy; the video does not wait for the answer
    elif CELLO_ENABLED:
        t0 = time.time()
        cello_pts, cello_conf = detect_cello(cello_model, cello_device, frame)
        cello_ms = (time.time() - t0) * 1000

    t_mp0 = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Timestamps must be ints and strictly increasing
    timestamp_ms = int((time.time() - start) * 1000)
    if timestamp_ms <= last_ts:
        timestamp_ms = last_ts + 1
    last_ts = timestamp_ms

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    t_mp_sum += time.time() - t_mp0

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

    # ----- Cello + bow -----
    if CELLO_ENABLED and CELLO_THREAD:
        cello_pts, cello_conf, cello_ms, cello_error = cello_worker.latest()
    if not CELLO_ENABLED:
        status, status_color = "cello: OFF (CELLO_ENABLED = True )", (200, 200, 200)
    elif cello_error:
        status, status_color = f"ERROR: {cello_error[:70]}", (0, 0, 255)
    elif cello_pts:
        status = f"cello: {len(cello_pts) - ('hand' in cello_pts)}/4 points   conf {cello_conf:.2f}   {cello_ms:.0f} ms"
        status_color = (0, 255, 0)
    else:
        status = f"cello: not found   {cello_ms:.0f} ms"
        status_color = (0, 165, 255)
    if cello_pts:
        draw_cello(frame, cello_pts)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    frame_count += 1
    if time.time() - fps_start >= 2.0:
        print(f"Actual FPS: {frame_count / (time.time() - fps_start):.1f}   |   MediaPipe {t_mp_sum / frame_count * 1000:.0f} ms/frame"
              f"   |   cello model {cello_ms:.0f} ms ({'thread' if CELLO_THREAD else 'in loop'}, device {cello_device})")
        t_mp_sum = 0.0
        frame_count = 0
        fps_start = time.time()

    # ----- Recording: write the finished frame BEFORE the on-screen recording overlay is added -----
    now = time.time()
    if rec_state == "countdown" and now >= rec_start:
        writer, rec_path = open_writer(w, h)
        rec_state, rec_t0, rec_written = ("recording" if writer else None), now, 0
        if not writer:
            saved_msg, saved_until = "Could not open the video writer", now + 4
    if rec_state == "recording":
        elapsed = now - rec_t0
        target = min(int(elapsed * RECORD_FPS) + 1, RECORD_SECONDS * RECORD_FPS)
        while rec_written < target:          # repeat the frame if we are slower than RECORD_FPS
            writer.write(frame)
            rec_written += 1
        if elapsed >= RECORD_SECONDS:
            writer.release()
            writer, rec_state = None, None
            saved_msg, saved_until = f"Saved {os.path.basename(rec_path)}", now + 5
            print(f"Saved {rec_path}")

    # On-screen only (not in the video)
    if rec_state == "countdown":
        cv2.putText(frame, f"Recording in {int(np.ceil(rec_start - now))}", (w // 2 - 170, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    elif rec_state == "recording":
        cv2.circle(frame, (w - 130, 24), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {now - rec_t0:4.1f}/{RECORD_SECONDS}s", (w - 115, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif now < saved_until:
        cv2.putText(frame, saved_msg, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if rec_state is None:
        cv2.putText(frame, f"R = record {RECORD_SECONDS}s video    Q = quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Pose + Cello", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # press q to quit
        break
    if key == ord("r") and rec_state is None:
        rec_state, rec_start = "countdown", time.time() + RECORD_COUNTDOWN

if writer:
    writer.release()                         # finish a recording that was still running when you quit
    print(f"Saved (stopped early) {rec_path}")
if cello_worker:
    cello_worker.running = False
cap.release()
cv2.destroyAllWindows()