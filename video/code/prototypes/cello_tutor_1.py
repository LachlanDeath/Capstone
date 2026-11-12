"""Cello Tutor 1.0 - overlay a target posture (a "ghost") on top of the player's live posture.

    python cello_tutor_1.py                          # live webcam: ghost + your own skeleton, cello and bow
    python cello_tutor_1.py --no-cello               # body only (no Roboflow calls, no cost)
    python cello_tutor_1.py --make-reference photo.jpg [--bow-side left|right]
                                                     # build the target posture from a clean photo of a good player

How the ghost is placed: the player's two EAR points (the face) are lined up with the ear points of the target posture.
The ghost is moved so its ears sit on yours, and scaled so its head is as wide as yours. Everything else in the ghost
(shoulders, elbows, wrists, cello neck, bow) then shows where those parts should be relative to your head.

Keys:  G = show / hide the ghost     Q = quit
Needs: $env:ROBOFLOW_API_KEY = "your-key"  (unless --no-cello)   pose_landmarker_lite.task and roboflow_workflow.py next to this file.
Optional: reference_posture.json next to this file replaces the built-in target (made by --make-reference).
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

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
sys.path.insert(0, BASE_DIR)
import roboflow_workflow as rf                      # noqa: E402  (must sit next to this file)

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
LIVE_MIRRORED = True         # the webcam view is flipped like a mirror (same as pose_cello_session.py)
BOW_ARM = "right"            # which of the PLAYER'S real arms holds the bow: "right" (normal) or "left"
SCALE_BASE = "face"          # "face" = ghost head as wide as yours (as asked) | "shoulders" = ghost shoulders as wide as yours
SMOOTH = 0.6                 # 0 = jumpy, 0.9 = very smooth but slow to follow
GHOST_COLOR = (255, 255, 0)  # BGR cyan
GHOST_ALPHA = 0.55
HINT_THRESHOLD = 0.8         # a joint must be off by more than this many head-widths before a hint is shown
KPT_CONF = 0.30              # minimum confidence for a cello point from Roboflow
CELLO_ENABLED = True
ROBOFLOW_MAX_HZ = 5          # calls per second. EVERY call is billed to your Roboflow credits
ROBOFLOW_MAX_CALLS = 3000    # safety net per run
ROBOFLOW_SEND_WIDTH = 640
REFERENCE_FILE = os.path.join(BASE_DIR, "reference_posture.json")

# Built-in target posture, read off the example photo (pixel positions in that 1920x1200 picture, NOT mirrored).
# l / r = the viewer's left / right in that photo. Replace it with your own using --make-reference.
DEFAULT_REFERENCE = {
    "bow_on_viewer_left": True,
    "points": {
        "nose": [962, 282], "face_l": [922, 254], "face_r": [1006, 251],
        "sh_l": [862, 336], "sh_r": [1066, 344],
        "el_l": [702, 321], "el_r": [1207, 440],
        "wr_l": [752, 418], "wr_r": [1099, 308],
        "scroll": [1050, 243], "bridge": [969, 600], "tip": [1133, 600],
    },
}

# MediaPipe pose index of each body point. The model names sides by where they appear in the picture:
# even numbers = viewer's left, odd = viewer's right, for a person facing the camera (mirrored or not).
IDX = {"nose": 0, "face_r": 7, "face_l": 8, "sh_r": 11, "sh_l": 12, "el_r": 13, "el_l": 14, "wr_r": 15, "wr_l": 16}
CONNECTIONS = [(7, 0), (0, 8), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
GHOST_BONES = [("face_l", "nose"), ("nose", "face_r"), ("sh_l", "sh_r"), ("sh_l", "el_l"), ("el_l", "wr_l"),
               ("sh_r", "el_r"), ("el_r", "wr_r")]
LEFT_IDS = {7, 11, 13, 15}
RIGHT_IDS = {8, 12, 14, 16}
LEFT_COLOR, RIGHT_COLOR, CENTER_COLOR = (255, 128, 0), (0, 255, 0), (200, 200, 200)
JOINT_COLOR, CELLO_COLOR, BOW_COLOR = (0, 0, 255), (0, 255, 255), (0, 165, 255)

LAST_HINTS = []              # what the last frame told the player (also used by the tests)


def swap_side(name):
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return name


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def side_color(a, b):
    if a in LEFT_IDS and b in LEFT_IDS:
        return LEFT_COLOR
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return RIGHT_COLOR
    return CENTER_COLOR


# ---------------------------------------------------------------
# Target posture (ghost)
# ---------------------------------------------------------------
def load_reference():
    if os.path.exists(REFERENCE_FILE):
        try:
            with open(REFERENCE_FILE, encoding="utf-8") as f:
                ref = json.load(f)
            print("Target posture:", REFERENCE_FILE)
            return ref
        except (OSError, ValueError) as e:
            print(f"Could not read {REFERENCE_FILE} ({e}); using the built-in target posture")
    return DEFAULT_REFERENCE


def make_ghost(ref, body, flip, k_state):
    """Place the target posture on the player. Returns {name: (x, y)} or None if the player's face is not found."""
    pts = {n: tuple(map(float, p)) for n, p in ref["points"].items()}
    if "face_l" not in pts or "face_r" not in pts or "face_l" not in body or "face_r" not in body:
        return None
    a_ref = ((pts["face_l"][0] + pts["face_r"][0]) / 2, (pts["face_l"][1] + pts["face_r"][1]) / 2)
    a_pl = ((body["face_l"][0] + body["face_r"][0]) / 2, (body["face_l"][1] + body["face_r"][1]) / 2)
    head_pl = dist(body["face_l"], body["face_r"])
    k = head_pl / max(dist(pts["face_l"], pts["face_r"]), 1e-6)
    if SCALE_BASE == "shoulders" and all(n in body for n in ("sh_l", "sh_r")) and all(n in pts for n in ("sh_l", "sh_r")):
        k = dist(body["sh_l"], body["sh_r"]) / max(dist(pts["sh_l"], pts["sh_r"]), 1e-6)
    if k_state["k"] is None:                                   # smooth the placement so the ghost does not shake
        k_state.update(k=k, anchor=a_pl, head=head_pl)
    else:
        s = SMOOTH
        k_state["k"] = s * k_state["k"] + (1 - s) * k
        k_state["anchor"] = (s * k_state["anchor"][0] + (1 - s) * a_pl[0], s * k_state["anchor"][1] + (1 - s) * a_pl[1])
        k_state["head"] = s * k_state["head"] + (1 - s) * head_pl
    k, anchor = k_state["k"], k_state["anchor"]
    ghost = {}
    for name, (x, y) in pts.items():
        dx, dy = x - a_ref[0], y - a_ref[1]
        if flip:                                               # mirror the target left <-> right
            dx, name = -dx, swap_side(name)
        ghost[name] = (anchor[0] + k * dx, anchor[1] + k * dy)
    return ghost


def draw_ghost(frame, ghost, bow_side):
    layer = frame.copy()
    ip = lambda p: (int(p[0]), int(p[1]))
    for a, b in GHOST_BONES:
        if a in ghost and b in ghost:
            cv2.line(layer, ip(ghost[a]), ip(ghost[b]), GHOST_COLOR, 5)
    if "scroll" in ghost and "bridge" in ghost:
        cv2.line(layer, ip(ghost["scroll"]), ip(ghost["bridge"]), GHOST_COLOR, 5)          # target cello neck
    if f"wr_{bow_side}" in ghost and "tip" in ghost:
        cv2.line(layer, ip(ghost[f"wr_{bow_side}"]), ip(ghost["tip"]), GHOST_COLOR, 5)      # target bow
    for p in ghost.values():
        cv2.circle(layer, ip(p), 8, GHOST_COLOR, -1)
    cv2.addWeighted(layer, GHOST_ALPHA, frame, 1 - GHOST_ALPHA, 0, frame)


def make_hints(ghost, body, cello_pts, head_w, bow_side):
    """The biggest differences between the player and the ghost, as short instructions (screen directions, like a mirror)."""
    neck_side = "r" if bow_side == "l" else "l"
    items = [("Bow elbow", f"el_{bow_side}", body.get(f"el_{bow_side}")),
             ("Bow wrist", f"wr_{bow_side}", body.get(f"wr_{bow_side}")),
             ("Fingering elbow", f"el_{neck_side}", body.get(f"el_{neck_side}")),
             ("Fingering wrist", f"wr_{neck_side}", body.get(f"wr_{neck_side}")),
             ("Cello scroll", "scroll", (cello_pts or {}).get("scroll"))]
    found = []
    for label, gname, player in items:
        if player is None or gname not in ghost:
            continue
        mx, my = ghost[gname][0] - player[0], ghost[gname][1] - player[1]
        off = math.hypot(mx, my) / max(head_w, 1e-6)
        if off > HINT_THRESHOLD:
            if abs(my) > abs(mx):
                what = "raise" if my < 0 else "lower"
            else:
                what = "move left" if mx < 0 else "move right"
            found.append((off, f"{label}: {what}"))
    return [t for _, t in sorted(found, reverse=True)[:3]]


# ---------------------------------------------------------------
# Roboflow cello model (same as the other scripts)
# ---------------------------------------------------------------
def detect_cello(frame, key):
    h, w = frame.shape[:2]
    send = frame if w <= ROBOFLOW_SEND_WIDTH else cv2.resize(frame, (ROBOFLOW_SEND_WIDTH, int(h * ROBOFLOW_SEND_WIDTH / w)))
    entry = rf.run_workflow(send, api_key=key)
    raw, box_conf, size = rf.parse_keypoints(entry, KPT_CONF)
    sw, sh = size if all(size) else (send.shape[1], send.shape[0])
    return {n: (int(x * w / sw), int(y * h / sh)) for n, (x, y, _c) in raw.items()}, box_conf


class CelloWorker(threading.Thread):
    def __init__(self, key):
        super().__init__(daemon=True)
        self.key, self.lock, self.pending = key, threading.Lock(), None
        self.pts, self.conf, self.ms, self.error, self.calls, self.running = None, 0.0, 0.0, None, 0, True

    def submit(self, frame):
        with self.lock:
            self.pending = frame.copy()

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
            if self.calls >= ROBOFLOW_MAX_CALLS:
                with self.lock:
                    self.error = f"call limit ({ROBOFLOW_MAX_CALLS}) reached - restart to continue"
                return
            self.calls += 1
            t0 = time.time()
            try:
                pts, conf = detect_cello(frame, self.key)
                with self.lock:
                    self.pts, self.conf, self.ms, self.error = (pts or None), conf, (time.time() - t0) * 1000, None
            except Exception as e:
                with self.lock:
                    self.error = str(e) or type(e).__name__
                if isinstance(e, rf.RoboflowAuthError):
                    return
            time.sleep(max(0.0, 1.0 / ROBOFLOW_MAX_HZ - (time.time() - t0)))


# ---------------------------------------------------------------
# Build a target posture from a photo
# ---------------------------------------------------------------
def body_points(lm, w, h, min_vis=0.5):
    return {name: (lm[i].x * w, lm[i].y * h) for name, i in IDX.items() if lm[i].visibility > min_vis}


def make_reference(path, bow_on_viewer_left):
    img = cv2.imdecode(np.fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read {path} as an image.")
    h, w = img.shape[:2]
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=find_file("pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.IMAGE))
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not res.pose_landmarks:
        sys.exit("No person found in that photo.")
    pts = body_points(res.pose_landmarks[0], w, h)
    if "face_l" not in pts or "face_r" not in pts:
        sys.exit("Both ears must be visible in the reference photo (the face is the anchor for the overlay).")
    try:
        cello, _ = detect_cello(img, rf.get_api_key())
        pts.update({n: p for n, p in cello.items() if n in ("scroll", "bridge", "tip")})
    except rf.RoboflowError as e:
        print("Roboflow error - the target will have no cello/bow:", e)
    missing = [n for n in ("scroll", "bridge", "tip") if n not in pts]
    if missing:
        print("Warning: not found in the photo:", ", ".join(missing))
    ref = {"bow_on_viewer_left": bool(bow_on_viewer_left), "image": [w, h],
           "points": {n: [round(x, 1), round(y, 1)] for n, (x, y) in pts.items()}}
    with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1)
    print(f"Saved target posture ({len(pts)} points) to {REFERENCE_FILE}")
    print("It is used automatically the next time you run:  python cello_tutor_1.py")


# ---------------------------------------------------------------
# Live tutor
# ---------------------------------------------------------------
def main():
    global LAST_HINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-reference", metavar="PHOTO")
    ap.add_argument("--bow-side", choices=["left", "right"], default="left",
                    help="where the bow hand is in the reference photo as you look at it (default: left)")
    ap.add_argument("--no-cello", action="store_true")
    a = ap.parse_args()
    if a.make_reference:
        return make_reference(a.make_reference, a.bow_side == "left")

    ref = load_reference()
    ref_bow_left = bool(ref.get("bow_on_viewer_left", True))
    # Which side of the screen the player's bow arm is on, and therefore whether the target must be mirrored
    player_bow_left = (BOW_ARM == "right") != LIVE_MIRRORED
    flip = player_bow_left != ref_bow_left
    bow_side = "l" if player_bow_left else "r"

    worker = None
    if CELLO_ENABLED and not a.no_cello:
        worker = CelloWorker(rf.get_api_key())
        worker.start()
        print(f"Cello tracker: Roboflow workflow {rf.WORKFLOW_ID}, at most {ROBOFLOW_MAX_HZ} calls/s, {ROBOFLOW_MAX_CALLS} calls per run")

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=find_file("pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened(), "| G = show/hide ghost, Q = quit")

    show_ghost = True
    k_state = {"k": None, "anchor": None, "head": None}
    start, last_ts, no_face_frames = time.time(), -1, 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break
        if LIVE_MIRRORED:
            frame = cv2.flip(frame, 1)
        if worker:
            worker.submit(frame)                                # clean frame goes out; drawings are added afterwards
        h, w = frame.shape[:2]

        ts = int((time.time() - start) * 1000)
        ts = max(ts, last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        body = body_points(result.pose_landmarks[0], w, h) if result.pose_landmarks else {}
        cello_pts, cello_conf, cello_ms, cello_error = worker.latest() if worker else (None, 0.0, 0.0, None)

        # 1) ghost first (semi-transparent), 2) the player's own skeleton, cello and bow on top
        ghost, hints = None, []
        if show_ghost:
            ghost = make_ghost(ref, body, flip, k_state)
            no_face_frames = 0 if ghost else no_face_frames + 1
            if ghost:
                draw_ghost(frame, ghost, bow_side)
                hints = make_hints(ghost, body, cello_pts, k_state["head"], bow_side)
        LAST_HINTS = hints

        for ia, ib in CONNECTIONS:
            na = next((n for n, i in IDX.items() if i == ia), None)
            nb = next((n for n, i in IDX.items() if i == ib), None)
            if na in body and nb in body:
                cv2.line(frame, tuple(map(int, body[na])), tuple(map(int, body[nb])), side_color(ia, ib), 2)
        for p in body.values():
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, JOINT_COLOR, -1)
        if cello_pts:
            if "scroll" in cello_pts and "bridge" in cello_pts:
                cv2.line(frame, cello_pts["scroll"], cello_pts["bridge"], CELLO_COLOR, 2)
            for n in ("scroll", "bridge", "tip"):
                if n in cello_pts:
                    cv2.circle(frame, cello_pts[n], 5, JOINT_COLOR, -1)
            if "tip" in cello_pts and f"wr_{bow_side}" in body:
                cv2.line(frame, tuple(map(int, body[f"wr_{bow_side}"])), cello_pts["tip"], BOW_COLOR, 3)

        # Text
        cv2.putText(frame, "Cello Tutor 1.0   cyan ghost = target posture", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GHOST_COLOR, 2)
        if not show_ghost:
            msg, col = "ghost hidden (G)", (200, 200, 200)
        elif ghost is None:
            msg, col = ("Face not found - both ears must be visible" if no_face_frames > 5 else ""), (0, 165, 255)
        elif hints:
            msg, col = "", (255, 255, 255)
        else:
            msg, col = "Posture matches the target", (0, 255, 0)
        if msg:
            cv2.putText(frame, msg, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for n, t in enumerate(hints):
            cv2.putText(frame, t, (10, 52 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if worker:
            if cello_error:
                cv2.putText(frame, f"cello ERROR: {cello_error[:70]}", (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"cello: {len(cello_pts or {})} points   {cello_ms:.0f} ms", (10, h - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if cello_pts else (0, 165, 255), 2)
        cv2.putText(frame, "G = ghost on/off    Q = quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Cello Tutor 1.0", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_ghost = not show_ghost
            k_state.update(k=None, anchor=None, head=None)
    if worker:
        worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
