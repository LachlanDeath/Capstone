import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Upper-body landmark indices and the lines between them
# 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips

KEYPOINTS = [12, 14, 16]              # right shoulder, elbow, wrist
CONNECTIONS = [(12, 14), (14, 16)]    # shoulder→elbow, elbow→wrist

# Hand landmarks: 0 wrist, then 4 joints per finger (thumb → pinky)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm edge
]

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEA
)
landmarker = vision.PoseLandmarker.create_from_options(options)

TRACKED_HAND = "Right"   # matches the right arm tracked above (subject's perspective)

hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,   # detect both so we can pick out the tracked-side hand below
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())   # sanity check

start = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame.shape[:2]

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]      # first detected person

        pts = {}
        for i in KEYPOINTS:
            if lm[i].visibility > 0.5:
                pts[i] = (int(lm[i].x * w), int(lm[i].y * h))

        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts.values():
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

    for hand, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        if handedness[0].category_name != TRACKED_HAND:
            continue

        hand_pts = [(int(p.x * w), int(p.y * h)) for p in hand]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, hand_pts[a], hand_pts[b], (255, 128, 0), 2)
        for p in hand_pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):   # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
