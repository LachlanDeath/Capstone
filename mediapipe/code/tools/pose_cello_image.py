"""Landmark ONE image with the same models and drawing as pose_cello_session.py:
MediaPipe body + one point per hand, and the Roboflow cello model (bridge, scroll, bow tip), plus the bow line.

    python pose_cello_image.py                 # a file-picker window opens: choose your image
    python pose_cello_image.py photo.jpg       # or give the path directly

The result is saved next to your image as <name>_landmarked.jpg and shown on screen (press any key to close it).
Needs:  $env:ROBOFLOW_API_KEY = "your-key"   (one Roboflow call per image is billed to your credits)
Files needed next to this script: pose_landmarker_lite.task, hand_landmarker.task, roboflow_workflow.py
"""
import os
import sys

import cv2
import mediapipe as mp
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
# Settings (same values as pose_cello_session.py)
# ---------------------------------------------------------------
KPT_CONF = 0.30              # minimum confidence for a cello/bow point to be drawn
ROBOFLOW_SEND_WIDTH = 640    # images wider than this are shrunk before upload
HAND_FRACTION = 0.18         # first-position hand ring: this fraction of the way from scroll to bridge
BOW_ARM = "right"            # which of the PLAYER'S arms holds the bow: "right" (normal) or "left" (left-handed bow hold)
BOW_GRIP_LANDMARK = 9        # the one hand point drawn (0 wrist, 5 index knuckle, 9 middle knuckle, 17 pinky knuckle)
MIRROR_IMAGE = False         # True = flip the image left-right first (to match the mirrored live view)
SHOW_WINDOW_MAX = (1280, 800)    # the window is shrunk to fit this; the saved file is full size

KEYPOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]
CONNECTIONS = [(7, 0), (0, 8), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
LEFT_IDS = {7, 11, 13, 15}
RIGHT_IDS = {8, 12, 14, 16}
LEFT_COLOR = (255, 128, 0)
RIGHT_COLOR = (0, 255, 0)
CENTER_COLOR = (200, 200, 200)
JOINT_COLOR = (0, 0, 255)
CELLO_COLOR = (0, 255, 255)     # scroll -> bridge
BOW_COLOR = (0, 165, 255)       # hand knuckle -> bow tip


def side_color(a, b):
    if a in LEFT_IDS and b in LEFT_IDS:
        return LEFT_COLOR
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return RIGHT_COLOR
    return CENTER_COLOR


def choose_image():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip('"')
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Choose an image of you playing the cello",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return input("Path to the image: ").strip().strip('"')


def detect_cello(img):
    """Returns ({'bridge','scroll','tip',...,'hand': (x, y)} for the confident points, box confidence)."""
    h, w = img.shape[:2]
    send = img if w <= ROBOFLOW_SEND_WIDTH else cv2.resize(img, (ROBOFLOW_SEND_WIDTH, int(h * ROBOFLOW_SEND_WIDTH / w)))
    entry = rf.run_workflow(send, api_key=rf.get_api_key())
    raw, box_conf, size = rf.parse_keypoints(entry, KPT_CONF)
    sw, sh = size if all(size) else (send.shape[1], send.shape[0])
    pts = {name: (int(x * w / sw), int(y * h / sh)) for name, (x, y, _c) in raw.items()}
    if "bridge" in pts and "scroll" in pts:
        sx, sy = pts["scroll"]
        bx, by = pts["bridge"]
        pts["hand"] = (int(sx + HAND_FRACTION * (bx - sx)), int(sy + HAND_FRACTION * (by - sy)))
    return (pts or None), box_conf


def main():
    path = choose_image()
    if not path:
        sys.exit("No image chosen.")
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    img = cv2.imdecode(__import__("numpy").fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)   # also works with non-ASCII paths
    if img is None:
        sys.exit(f"Could not read {path} as an image.")
    if MIRROR_IMAGE:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    print(f"Image: {path}  ({w} x {h})")

    # Cello model on the clean image (drawings are added afterwards)
    cello_pts, cello_conf, error = None, 0.0, None
    try:
        cello_pts, cello_conf = detect_cello(img)
    except rf.RoboflowError as e:
        error = str(e)
        print("Roboflow error:", error)

    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=find_file("pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.IMAGE))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=find_file("hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = pose.detect(mp_image)
    hand_result = hands.detect(mp_image)

    out = img.copy()

    # ----- Pose (upper body, both arms) -----
    n_body = 0
    pose_pts = {}
    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        pose_pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in KEYPOINTS if lm[i].visibility > 0.5}
        n_body = len(pose_pts)
        for a, b in CONNECTIONS:
            if a in pose_pts and b in pose_pts:
                cv2.line(out, pose_pts[a], pose_pts[b], side_color(a, b), 2)
        for p in pose_pts.values():
            cv2.circle(out, p, 5, JOINT_COLOR, -1)

    # ----- Hands: ONE point per hand (the grip knuckle), coloured like the arm it belongs to -----
    # MediaPipe's Left/Right hand label is unreliable on single photos, so each hand is matched to the nearest body wrist instead.
    wrists = {i: pose_pts[i] for i in (15, 16) if i in pose_pts}
    grips = []                                       # (x, y, colour)
    for hand in hand_result.hand_landmarks:
        gx, gy = int(hand[BOW_GRIP_LANDMARK].x * w), int(hand[BOW_GRIP_LANDMARK].y * h)
        if wrists:
            near = min(wrists, key=lambda i: (wrists[i][0] - gx) ** 2 + (wrists[i][1] - gy) ** 2)
            color = RIGHT_COLOR if near in RIGHT_IDS else LEFT_COLOR
        else:
            color = CENTER_COLOR
        cv2.circle(out, (gx, gy), 7, color, -1)
        cv2.circle(out, (gx, gy), 9, (0, 0, 0), 1)
        grips.append((gx, gy, color))

    # ----- Cello points + scroll->bridge line -----
    if cello_pts:
        if "scroll" in cello_pts and "bridge" in cello_pts:
            cv2.line(out, cello_pts["scroll"], cello_pts["bridge"], CELLO_COLOR, 2)
        if "frog" in cello_pts and "tip" in cello_pts:
            cv2.line(out, cello_pts["frog"], cello_pts["tip"], BOW_COLOR, 2)
        for name, p in cello_pts.items():
            if name == "hand":
                cv2.circle(out, p, 9, JOINT_COLOR, 2)
            else:
                cv2.circle(out, p, 5, JOINT_COLOR, -1)

    # ----- Bow line: grip of the BOW hand -> bow tip -----
    # The bow hand is the one attached to the player's bow arm (found through the body wrist), NOT simply the hand nearest the tip:
    # the other hand (on the fingerboard) is often closer to the tip and was being picked by mistake.
    bow_source = None
    if cello_pts and "tip" in cello_pts:
        tx, ty = cello_pts["tip"]
        wrist_id = 16 if BOW_ARM == "right" else 15
        if MIRROR_IMAGE:
            wrist_id = 31 - wrist_id                 # a flipped image swaps which side the arms appear on
        bw, be = pose_pts.get(wrist_id), pose_pts.get(wrist_id - 2)
        if bw:
            reach = ((bw[0] - be[0]) ** 2 + (bw[1] - be[1]) ** 2) ** 0.5 if be else 0.1 * w     # forearm length ~ hand-to-wrist range
            near = [g for g in grips if ((g[0] - bw[0]) ** 2 + (g[1] - bw[1]) ** 2) ** 0.5 <= 1.2 * reach]
            if near:
                g = min(near, key=lambda g: (g[0] - bw[0]) ** 2 + (g[1] - bw[1]) ** 2)
                start, bow_source = (g[0], g[1]), "hand tracker"
            else:                                    # the hand tracker missed the bow hand (blur, fingers wrapped round the frog)
                start, bow_source = bw, "body wrist"
                cv2.circle(out, bw, 9, BOW_COLOR, 2)  # hollow ring = this point comes from the body model, not the hand model
        elif grips:                                  # no body found at all: fall back to the hand nearest the tip
            g = min(grips, key=lambda g: (g[0] - tx) ** 2 + (g[1] - ty) ** 2)
            start, bow_source = (g[0], g[1]), "nearest hand (no body found)"
        if bow_source:
            cv2.line(out, start, (tx, ty), BOW_COLOR, 3)
            print("Bow line starts at the", bow_source)

    # ----- Status text (same wording as the live view) -----
    if error:
        status, col = f"ERROR: {error[:70]}", (0, 0, 255)
    elif cello_pts:
        found = len(cello_pts) - ("hand" in cello_pts)
        status, col = f"cello: {found} points   conf {cello_conf:.2f}", (0, 255, 0)
    else:
        status, col = f"cello: nothing above {KPT_CONF:.2f}", (0, 165, 255)
    scale = max(0.6, w / 1000)
    cv2.putText(out, status, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, col, max(2, int(2 * scale)))

    base, _ = os.path.splitext(path)
    out_path = base + "_landmarked.jpg"
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        sys.exit("Could not encode the output image.")
    buf.tofile(out_path)
    names = ", ".join(sorted(k for k in (cello_pts or {}) if k != "hand")) or "none"
    print(f"Body points: {n_body}   hands: {len(grips)}   cello points: {names}")
    print("Saved:", out_path)

    # Show it (press any key to close); skip quietly if there is no display
    try:
        s = min(1.0, SHOW_WINDOW_MAX[0] / w, SHOW_WINDOW_MAX[1] / h)
        view = out if s >= 1.0 else cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow("Landmarked image (press any key)", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
