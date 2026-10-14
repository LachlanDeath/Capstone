import threading
import time

import cv2

# Keep guitar_infer.py, guitar_landmarks.py and this file in the same folder
from guitar_infer import detect_guitar, get_api_key
from guitar_landmarks import draw_guitar, guitar_from_result

SEND_WIDTH = 640         # frames are shrunk to this width before upload (try 320-640)
MIN_INTERVAL = 0.05      # low when testing tracking, high otherwise (API call frequency)
MIN_CONFIDENCE = 0.5     # same idea as the `visibility > 0.5` check in pose_botharms.py


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


def main():
    worker = GuitarWorker(get_api_key())
    worker.start()

    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened())
    frame_count = 0
    fps_start = time.time()
    last_result, pts, conf = None, None, 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)   # mirror BEFORE sending, same as pose_botharms.py

        worker.submit(frame)
        result, latency, error = worker.latest()

        # Only redo the landmark math when a new result has actually arrived
        if result is not last_result:
            pts, conf = guitar_from_result(result, MIN_CONFIDENCE)
            last_result = result

        if pts:
            draw_guitar(frame, pts)

        if error:
            status, color = f"ERROR: {error[:70]}", (0, 0, 255)
        elif latency is None:
            status, color = "waiting for first result...", (0, 255, 255)
        elif pts:
            status, color = f"guitar landmarks: {len(pts)}   conf {conf:.2f}   round trip: {latency * 1000:.0f} ms", (0, 255, 0)
        else:
            status, color = f"no guitar found   round trip: {latency * 1000:.0f} ms", (0, 165, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1
        if time.time() - fps_start >= 2.0:
            print(f"Actual FPS: {frame_count / (time.time() - fps_start):.1f}")
            frame_count = 0
            fps_start = time.time()

        cv2.imshow("Guitar landmarks (live)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()