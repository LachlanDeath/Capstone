import threading
import time

import cv2
#$env:ROBOFLOW_API_KEY = "your_actual_roboflow_key_here"
# Reuses the API call and drawing code from guitar_infer.py (keep both files in the same folder)
from guitar_infer import detect_guitar, draw_predictions, get_api_key

SEND_WIDTH = 640         # frames are shrunk to this width before upload (try 320-640)
MIN_INTERVAL = 0.5     # low when testing tracking, high otherwise (API call frequency)

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

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)   # mirror BEFORE sending: the model and the display both use the mirrored frame

        worker.submit(frame)
        result, latency, error = worker.latest()

        if result:
            frame = draw_predictions(frame, result)

        if error:
            status, color = f"ERROR: {error[:70]}", (0, 0, 255)
        elif latency is None:
            status, color = "waiting for first result...", (0, 255, 255)
        else:
            n = len(result.get("predictions", []))
            status, color = f"guitars: {n}   round trip: {latency * 1000:.0f} ms", (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Guitar (live)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()