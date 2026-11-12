import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

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
# Models
# ---------------------------------------------------------------
pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
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

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    # Mirror the image so it behaves like a mirror and Left/Right labels are correct
    frame = cv2.flip(frame, 1)

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

        pts = {}
        for i in KEYPOINTS:
            if lm[i].visibility > 0.5:
                pts[i] = (int(lm[i].x * w), int(lm[i].y * h))

        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], side_color(a, b), 2)
        for p in pts.values():
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

    frame_count += 1
    if time.time() - fps_start >= 2.0:
        print(f"Actual FPS: {frame_count / (time.time() - fps_start):.1f}")
        frame_count = 0
        fps_start = time.time()

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()