"""Live webcam test of the Roboflow cello model: draws the keypoints it finds (bridge, scroll, tip, ...) on your video.

Setup (PowerShell, once per terminal; never put the key in a file):
    $env:ROBOFLOW_API_KEY = "your-key"
Run (roboflow_workflow.py must be in the same folder):
    python test_cello_live.py
Keys: Q = quit.   Every call is billed to your Roboflow credits; it stops by itself after MAX_CALLS.
"""
import os
import sys
import threading
import time

import cv2

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

import roboflow_workflow as rf   # noqa: E402

MIN_CONF = 0.30      # draw keypoints at or above this confidence (the workflow itself returns everything >= 0.15)
MAX_HZ = 5           # calls per second
MAX_CALLS = 1500     # safety net (about 5 minutes at 5 calls/s)
SEND_WIDTH = 640     # frames are shrunk to this width before upload
COLORS = {"bridge": (255, 0, 255), "scroll": (255, 200, 0), "tip": (0, 255, 255), "frog": (0, 255, 0)}


class Worker(threading.Thread):
    def __init__(self, key):
        super().__init__(daemon=True)
        self.key, self.lock, self.pending = key, threading.Lock(), None
        self.result, self.ms, self.error, self.calls, self.running = None, 0.0, None, 0, True

    def submit(self, frame):
        with self.lock:
            self.pending = frame.copy()

    def latest(self):
        with self.lock:
            return self.result, self.ms, self.error, self.calls

    def run(self):
        while self.running:
            with self.lock:
                frame, self.pending = self.pending, None
            if frame is None:
                time.sleep(0.002)
                continue
            if self.calls >= MAX_CALLS:
                with self.lock:
                    self.error = f"call limit ({MAX_CALLS}) reached - restart to continue"
                return
            self.calls += 1
            t0 = time.time()
            try:
                h, w = frame.shape[:2]
                send = frame if w <= SEND_WIDTH else cv2.resize(frame, (SEND_WIDTH, int(h * SEND_WIDTH / w)))
                entry = rf.run_workflow(send, api_key=self.key)
                pts, box_conf, size = rf.parse_keypoints(entry, MIN_CONF)
                sw, sh = size if all(size) else (send.shape[1], send.shape[0])
                scaled = {n: (int(x * w / sw), int(y * h / sh), c) for n, (x, y, c) in pts.items()}
                with self.lock:
                    self.result, self.ms, self.error = (scaled, box_conf), (time.time() - t0) * 1000, None
            except Exception as e:
                with self.lock:
                    self.error = str(e) or type(e).__name__
                if isinstance(e, rf.RoboflowAuthError):
                    return
            time.sleep(max(0.0, 1.0 / MAX_HZ - (time.time() - t0)))


def main():
    worker = Worker(rf.get_api_key())
    worker.start()
    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened(), "| model workflow:", rf.WORKFLOW_ID, "| press Q to quit")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)               # mirror, same as the rest of the project
        worker.submit(frame)                     # clean frame goes out; drawings are added after
        result, ms, error, calls = worker.latest()
        if result:
            pts, box_conf = result
            for name, (x, y, c) in pts.items():
                col = COLORS.get(name, (255, 255, 255))
                cv2.circle(frame, (x, y), 8, col, -1)
                cv2.circle(frame, (x, y), 10, (0, 0, 0), 1)
                cv2.putText(frame, f"{name} {c:.2f}", (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            if "scroll" in pts and "bridge" in pts:
                cv2.line(frame, pts["scroll"][:2], pts["bridge"][:2], (0, 255, 255), 2)
            status = f"found: {', '.join(sorted(pts)) or 'nothing above %.2f' % MIN_CONF}   {ms:.0f} ms   calls {calls}"
        else:
            status = "waiting for first answer..."
        if error:
            status = "ERROR: " + error[:90]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Cello model test", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
