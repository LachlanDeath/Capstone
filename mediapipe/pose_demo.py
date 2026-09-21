import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Upper-body landmark indices and the lines between them
# 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips
CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
               (11, 23), (12, 24), (23, 24)]
KEYPOINTS = [0, 11, 13, 15]

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]      # first detected person
        h, w = frame.shape[:2]
        pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in KEYPOINTS}

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts.values():
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):   # press q to quit
        break

cap.release()
cv2.destroyAllWindows()