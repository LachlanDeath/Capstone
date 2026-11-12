import atexit
import csv
import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model files are looked up next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# roboflow_workflow.py and the .task model files are looked for in this folder and up to 3 folders above it,
# so the script works from a sub-folder such as code\\tools as well as from the main folder
SEARCH_DIRS = [BASE_DIR] + [os.path.abspath(os.path.join(BASE_DIR, *[".."] * n)) for n in (1, 2, 3)]
sys.path[:0] = [d for d in SEARCH_DIRS if d not in sys.path]


def find_file(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, name)      # not found anywhere: MediaPipe will name this path in its error

# ---------------------------------------------------------------
# Cello settings (Roboflow workflow "cello-posture-keypoints-v2" = model cello-posture-keypoints/2, runs in the cloud)
# Needs:  $env:ROBOFLOW_API_KEY = "your-key"   (never put the key in this file)
# ---------------------------------------------------------------
KPT_CONF = 0.30              # minimum confidence for a cello/bow point to be drawn (same as test_cello_live.py)
CELLO_ENABLED = True         # False = skip the cello completely (see how fast MediaPipe is on its own)
ROBOFLOW_MAX_HZ = 5          # calls per second to Roboflow. EVERY call is billed to your Roboflow credits
ROBOFLOW_MAX_CALLS = 3000    # safety net: stop calling Roboflow after this many calls in one run
ROBOFLOW_SEND_WIDTH = 640    # frames wider than this are shrunk before upload
HAND_FRACTION = 0.18         # first-position hand point: this fraction of the way from scroll to bridge

# ---------------------------------------------------------------
# Session recording: the WHOLE session is saved automatically, from start until you press Q.
#   recordings\session_<date>_<time>.mp4  = exactly what you see (body + hands + cello points)
#   recordings\session_<date>_<time>.csv  = per frame: time, cello confidence, Roboflow round trip, each point's x/y, errors
# ---------------------------------------------------------------
RECORD_FPS = 25                       # frames are repeated/skipped as needed so the video plays at real speed
RECORD_DIR = os.path.join(BASE_DIR, "recordings")
SESSION_MAX_SECONDS = 600             # safety: stops by itself after this long (also keeps the file small enough to send)


def open_writer(width, height, path):
    os.makedirs(RECORD_DIR, exist_ok=True)
    for codec in ("avc1", "mp4v"):   # avc1 (H.264) plays everywhere and is smaller; mp4v is the fallback
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), RECORD_FPS, (width, height))
        if writer.isOpened():
            print(f"Recording the whole session to {path} ({codec})")
            return writer
        writer.release()
    return None


# ---------------------------------------------------------------
# Pose landmark indices (odd = subject's left, even = subject's right)
# 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists
# ---------------------------------------------------------------
KEYPOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]      # hips (23, 24) removed

CONNECTIONS = [
    (7, 0), (0, 8),          # ears to nose
    (11, 12),                # shoulder line
    (11, 13), (13, 15),      # left arm
    (12, 14), (14, 16),      # right arm
]

LEFT_IDS = {7, 11, 13, 15}
RIGHT_IDS = {8, 12, 14, 16}

# BGR colors
LEFT_COLOR = (255, 128, 0)      # blue-ish orange
RIGHT_COLOR = (0, 255, 0)       # green
CENTER_COLOR = (200, 200, 200)  # gray
JOINT_COLOR = (0, 0, 255)       # red
CELLO_COLOR = (0, 255, 255)     # yellow  (scroll -> bridge)
BOW_COLOR = (0, 165, 255)       # orange  (frog -> tip)


def side_color(a, b):
    if a in LEFT_IDS and b in LEFT_IDS:
        return LEFT_COLOR
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return RIGHT_COLOR
    return CENTER_COLOR


# Which hand holds the bow: "auto" = the hand whose grip point is closest to the bow tip (robust to Left/Right mix-ups),
# or "Right" / "Left" to force one (MediaPipe's label, as seen on the mirrored image)
BOW_HAND = "auto"
BOW_GRIP_LANDMARK = 9      # the one hand point that is drawn (0 wrist, 5 index knuckle, 9 middle knuckle, 17 pinky knuckle)

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
# Cello detection (Roboflow workflow)
# ---------------------------------------------------------------
def detect_cello(frame):
    """Returns ({'bridge','scroll','tip',...,'hand': (x, y)} for the confident points, box confidence)."""
    h, w = frame.shape[:2]
    send = frame if w <= ROBOFLOW_SEND_WIDTH else cv2.resize(frame, (ROBOFLOW_SEND_WIDTH, int(h * ROBOFLOW_SEND_WIDTH / w)))
    entry = rf.run_workflow(send, api_key=ROBOFLOW_KEY)
    raw, box_conf, size = rf.parse_keypoints(entry, KPT_CONF)
    sw, sh = size if all(size) else (send.shape[1], send.shape[0])
    pts = {name: (int(x * w / sw), int(y * h / sh)) for name, (x, y, _c) in raw.items()}   # back to full-frame pixels
    if "bridge" in pts and "scroll" in pts:
        sx, sy = pts["scroll"]
        bx, by = pts["bridge"]
        pts["hand"] = (int(sx + HAND_FRACTION * (bx - sx)), int(sy + HAND_FRACTION * (by - sy)))
    return (pts or None), box_conf


def draw_cello(frame, pts):
    if "scroll" in pts and "bridge" in pts:
        cv2.line(frame, pts["scroll"], pts["bridge"], CELLO_COLOR, 2)
    if "frog" in pts and "tip" in pts:
        cv2.line(frame, pts["frog"], pts["tip"], BOW_COLOR, 2)
    for name, p in pts.items():
        if name == "hand":
            cv2.circle(frame, p, 9, JOINT_COLOR, 2)       # ring = where the left hand should be in first position
        else:
            cv2.circle(frame, p, 5, JOINT_COLOR, -1)


class CelloWorker(threading.Thread):
    """Runs the cello model on the newest frame in the background and keeps the latest answer."""

    def __init__(self, detect_fn, max_hz, max_calls=None):
        super().__init__(daemon=True)
        self.detect_fn, self.max_hz, self.max_calls, self.calls = detect_fn, max_hz, max_calls, 0
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
            if self.max_calls is not None and self.calls >= self.max_calls:
                with self.lock:
                    self.error = f"call limit reached ({self.max_calls}); restart to continue"
                self.running = False
                break
            self.calls += 1
            t0 = time.time()
            try:
                pts, conf = self.detect_fn(frame)
                with self.lock:
                    self.pts, self.conf, self.ms, self.error = pts, conf, (time.time() - t0) * 1000, None
            except Exception as e:
                with self.lock:
                    self.error = str(e) or type(e).__name__
                if type(e).__name__ == "RoboflowAuthError":      # a wrong/revoked key will not fix itself
                    self.running = False
                    break
            time.sleep(max(0.0, 1.0 / self.max_hz - (time.time() - t0)))


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
cello_worker = None
if CELLO_ENABLED:
    sys.path.insert(0, BASE_DIR)
    import roboflow_workflow as rf              # roboflow_workflow.py must sit next to this file
    ROBOFLOW_KEY = rf.get_api_key()             # reads $env:ROBOFLOW_API_KEY, exits with a message if it is missing
    print(f"Cello tracker: Roboflow workflow {rf.WORKFLOW_ID}, at most {ROBOFLOW_MAX_HZ} calls/s and {ROBOFLOW_MAX_CALLS} calls per run")
    cello_worker = CelloWorker(detect_cello, ROBOFLOW_MAX_HZ, ROBOFLOW_MAX_CALLS)
    cello_worker.start()

pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=find_file("pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=find_file("hand_landmarker.task")),
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
session_stamp = time.strftime("session_%Y%m%d_%H%M%S")
video_path = os.path.join(RECORD_DIR, session_stamp + ".mp4")
log_path = os.path.join(RECORD_DIR, session_stamp + ".csv")
os.makedirs(RECORD_DIR, exist_ok=True)
LOG_POINTS = ["bridge", "scroll", "tip", "frog", "hand"]
log_file = open(log_path, "w", newline="", encoding="utf-8")
log = csv.writer(log_file)
log.writerow(["frame", "t_seconds", "cello_conf", "roboflow_ms", "n_points"]
             + [f"{n}_{a}" for n in LOG_POINTS for a in ("x", "y")] + ["bow_grip_x", "bow_grip_y", "error"])
writer, rec_written, n_frames, session_t0 = None, 0, 0, time.time()


def finish_session():
    global writer
    if writer:
        writer.release()
        writer = None
        print(f"Saved video: {video_path}")
    if not log_file.closed:
        log_file.close()
        print(f"Saved log:   {log_path}")
    if cello_worker:
        cello_worker.running = False


atexit.register(finish_session)          # also runs if the script crashes or you press Ctrl+C


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    # Mirror the image so it behaves like a mirror and Left/Right labels are correct.
    # MediaPipe AND the cello model both work on this same mirrored frame.
    frame = cv2.flip(frame, 1)

    # Cello model gets the clean frame (nothing drawn on it yet)
    if CELLO_ENABLED:
        cello_worker.submit(frame)            # keeps its own copy; the video does not wait for the answer

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
    grips = []                       # (label, x, y) of the one grip point of every detected hand
    for hand, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        color = RIGHT_COLOR if label == "Right" else LEFT_COLOR

        # Only ONE point per hand: landmark 9 = middle-finger knuckle, the centre of the grip on the bow frog.
        # It stays steady when the fingers curl around the bow (fingertips and wrist move much more).
        gx, gy = int(hand[BOW_GRIP_LANDMARK].x * w), int(hand[BOW_GRIP_LANDMARK].y * h)
        cv2.circle(frame, (gx, gy), 7, color, -1)
        cv2.circle(frame, (gx, gy), 9, (0, 0, 0), 1)
        grips.append((label, gx, gy))

    # ----- Cello + bow -----
    if CELLO_ENABLED:
        cello_pts, cello_conf, cello_ms, cello_error = cello_worker.latest()
    if not CELLO_ENABLED:
        status, status_color = "cello: OFF (set CELLO_ENABLED = True)", (200, 200, 200)
    elif cello_error:
        status, status_color = f"ERROR: {cello_error[:70]}", (0, 0, 255)
    elif cello_pts:
        status = f"cello: {len(cello_pts) - ('hand' in cello_pts)} points   conf {cello_conf:.2f}   {cello_ms:.0f} ms"
        status_color = (0, 255, 0)
    else:
        status = f"cello: nothing above {KPT_CONF:.2f}   {cello_ms:.0f} ms"
        status_color = (0, 165, 255)
    if cello_pts:
        draw_cello(frame, cello_pts)
    # ----- Bow line: knuckle of the bow hand -> bow tip -----
    bow_grip = None
    if cello_pts and "tip" in cello_pts:
        tx, ty = cello_pts["tip"]
        cands = [g for g in grips if BOW_HAND == "auto" or g[0] == BOW_HAND]
        if cands:
            bow_grip = min(cands, key=lambda g: (g[1] - tx) ** 2 + (g[2] - ty) ** 2)
            cv2.line(frame, (bow_grip[1], bow_grip[2]), (tx, ty), BOW_COLOR, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    frame_count += 1
    if time.time() - fps_start >= 2.0:
        print(f"Actual FPS: {frame_count / (time.time() - fps_start):.1f}   |   MediaPipe {t_mp_sum / frame_count * 1000:.0f} ms/frame"
              f"   |   cello (Roboflow) {cello_ms:.0f} ms round trip")
        t_mp_sum = 0.0
        frame_count = 0
        fps_start = time.time()

    # ----- Record the finished frame (before the on-screen REC overlay is added) and log the cello result -----
    now = time.time()
    elapsed = now - session_t0
    if writer is None and rec_written == 0:
        writer = open_writer(w, h, video_path)
        if writer is None:
            print("Could not open a video writer - nothing will be recorded")
            rec_written = -1
    if writer:
        target = int(elapsed * RECORD_FPS) + 1
        while rec_written < target:          # repeat the frame if we are slower than RECORD_FPS
            writer.write(frame)
            rec_written += 1
    n_frames += 1
    p = cello_pts or {}
    log.writerow([n_frames, f"{elapsed:.3f}", f"{cello_conf:.3f}", f"{cello_ms:.0f}", len([k for k in p if k != "hand"])]
                 + [v for n in LOG_POINTS for v in (p[n] if n in p else ("", ""))]
                 + ([bow_grip[1], bow_grip[2]] if bow_grip else ["", ""]) + [cello_error or ""])
    if n_frames % 50 == 0:
        log_file.flush()

    # On-screen only (not in the video)
    cv2.circle(frame, (w - 110, 24), 8, (0, 0, 255), -1)
    cv2.putText(frame, f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}", (w - 95, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Press Q to stop and save", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Pose + Cello (recording)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or elapsed >= SESSION_MAX_SECONDS:
        break

finish_session()
cap.release()
cv2.destroyAllWindows()
