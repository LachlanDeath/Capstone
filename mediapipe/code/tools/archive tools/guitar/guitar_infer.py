import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

MODEL_ID = "guitar-ojn01/2"   # Roboflow instance segmentation model
API_URL = "https://serverless.roboflow.com"

JPEG_QUALITY = 70        # lower = smaller upload (60-80 is usually fine for detection)
USE_MULTIPART = False    # True = send raw JPEG bytes instead of base64 (~25% less upload).
                         # Untested on this endpoint: if it errors, set back to False.

# One shared session keeps the HTTPS connection open between requests, so only the
# first call pays for the TCP + TLS handshake (often 100+ ms by itself).
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=0))


def get_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set.\n"
            "PowerShell (this window only):  $env:ROBOFLOW_API_KEY = 'your-key'"
        )
    return key


def detect_guitar(frame, api_key=None, timeout=15, jpeg_quality=JPEG_QUALITY):
    """frame: a BGR numpy image (e.g. from OpenCV). Returns Roboflow's JSON dict.

    The result gets an extra "_timing" entry (encode/request ms, payload sizes) so you
    can see where the time goes.
    """
    api_key = api_key or get_api_key()

    t0 = time.perf_counter()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Could not encode image as JPEG")
    jpg = buf.tobytes()

    headers = {"Authorization": f"Bearer {api_key}"}   # header auth, never in the URL
    url = f"{API_URL}/{MODEL_ID}"
    t1 = time.perf_counter()

    if USE_MULTIPART:
        resp = _session.post(url, headers=headers,
                             files={"file": ("frame.jpg", jpg, "image/jpeg")},
                             timeout=timeout)
    else:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        resp = _session.post(url, headers=headers,
                             data=base64.b64encode(jpg), timeout=timeout)
    t2 = time.perf_counter()

    if resp.status_code in (401, 403):
        sys.exit(f"Roboflow rejected the API key (HTTP {resp.status_code}). "
                 "Check ROBOFLOW_API_KEY.")
    resp.raise_for_status()
    result = resp.json()

    result["_timing"] = {
        "encode_ms": (t1 - t0) * 1000,
        "request_ms": (t2 - t1) * 1000,        # upload + server + download
        "payload_kb": len(jpg) / 1024,
        "response_kb": len(resp.content) / 1024,
    }
    return result


def draw_predictions(frame, result, color=(0, 200, 255)):
    """Draw each predicted guitar mask (polygon) and label on the frame."""
    overlay = frame.copy()
    for pred in result.get("predictions", []):
        label = f"{pred.get('class', 'guitar')} {pred.get('confidence', 0):.2f}"
        points = pred.get("points")

        if points:
            poly = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)
            anchor = tuple(int(v) for v in poly[poly[:, 1].argmin()])   # topmost point
        else:
            # Fallback: Roboflow boxes are center-based (x, y, width, height)
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            anchor = (x1, y1)

        cv2.putText(frame, label, (anchor[0], max(anchor[1] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


def grab_webcam_frame():
    cap = cv2.VideoCapture(0)
    frame = None
    for _ in range(10):          # let auto-exposure settle; keep the last good frame
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Run the guitar model on an image (or one webcam frame) and time it.")
    parser.add_argument("image", nargs="?", help="image path; omit to use one webcam frame")
    parser.add_argument("--repeat", type=int, default=1,
                        help="send the same frame N times to measure speed")
    parser.add_argument("--width", type=int, default=0,
                        help="shrink the frame to this width before sending (e.g. 320, 416, 640)")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read image: {args.image}")
        base, _ = os.path.splitext(args.image)
    else:
        frame = grab_webcam_frame()
        if frame is None:
            sys.exit("Could not read from the webcam.")
        base = "webcam_test"
        cv2.imwrite(base + ".jpg", frame)
        print("Grabbed a webcam frame and saved it as webcam_test.jpg")

    h, w = frame.shape[:2]
    if args.width and w > args.width:
        frame = cv2.resize(frame, (args.width, int(h * args.width / w)))
    print(f"Sending {frame.shape[1]}x{frame.shape[0]} frame, "
          f"JPEG quality {JPEG_QUALITY}, multipart={USE_MULTIPART}")

    result, timings = None, []
    for i in range(max(1, args.repeat)):
        result = detect_guitar(frame)
        t = result["_timing"]
        timings.append(t["request_ms"])
        print(f"call {i + 1}: request {t['request_ms']:.0f} ms | encode {t['encode_ms']:.0f} ms | "
              f"up {t['payload_kb']:.0f} KB | down {t['response_kb']:.0f} KB")
    if len(timings) > 1:
        rest = timings[1:]
        print(f"first call {timings[0]:.0f} ms (includes connection setup); "
              f"average of the rest {sum(rest) / len(rest):.0f} ms")

    preds = result.get("predictions", [])
    print(f"{len(preds)} detection(s)")
    for p in preds:
        print(f"  {p.get('class')}  conf={p.get('confidence', 0):.2f}  "
              f"polygon_points={len(p.get('points', []))}")

    with open(base + "_guitar.json", "w") as f:
        json.dump(result, f, indent=2)
    cv2.imwrite(base + "_guitar.jpg", draw_predictions(frame, result))
    print("Saved:", base + "_guitar.jpg and", base + "_guitar.json")


if __name__ == "__main__":
    main()