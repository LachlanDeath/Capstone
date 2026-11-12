"""Run the Roboflow Workflow "cello-posture-keypoints" on a webcam frame (plain HTTPS, one frame per call).

Why plain requests and not inference-sdk: the SDK (and its WebRTC streaming) does not install on Python 3.14,
and WebRTC takes over the webcam, which would clash with the MediaPipe loop. This is the same endpoint the SDK uses.

Setup (PowerShell, once per terminal - never type the key into a file):
    $env:ROBOFLOW_API_KEY = "your-key"

Try it:
    python roboflow_workflow.py --selftest          # offline check of the response parser (no key, no network)
    python roboflow_workflow.py                     # one real call on a webcam frame; prints what came back + timing
    python roboflow_workflow.py --image some.jpg --repeat 5

Real response shape (from a live probe of this workflow):
    [ { "predictions": { "image": {"width": W, "height": H},
                         "predictions": [ {"class": "bow",   "confidence": c, "x":.., "y":.., "width":.., "height":..,
                                           "keypoints": [ {"class": "tip",    "confidence": k, "x":.., "y":..}, ... ]},
                                          {"class": "cello", "confidence": c, ...,
                                           "keypoints": [ {"class": "bridge", ...}, {"class": "scroll", ...}, ... ]} ] },
        "inference_id": "...", "model_id": "cello-posture-keypoints/1" } ]
Keypoint coordinates are in pixels of the image that was SENT.
"""
import argparse
import base64
import json
import os
import sys
import time

import cv2
import requests
from requests.adapters import HTTPAdapter

WORKSPACE = "maxsloan2005-outlook-com"
WORKFLOW_ID = "cello-posture-keypoints-v2"   # uses model cello-posture-keypoints/2 (406 images); the original workflow is locked to model 1
API_URL = "https://serverless.roboflow.com"
JPEG_QUALITY = 70
TIMEOUT = 15            # seconds per request
RETRIES = 2             # extra attempts after a timeout / connection error / 429 / 5xx
BACKOFF = 0.3           # seconds, doubled each retry

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=2))   # reuse the connection between frames


class RoboflowError(RuntimeError):
    """Any failure talking to the workflow (network, timeout, bad response)."""


class RoboflowAuthError(RoboflowError):
    """401/403: the API key is missing, wrong, revoked, or has no access to this workflow."""


def get_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit('ROBOFLOW_API_KEY is not set in this terminal. Run:  $env:ROBOFLOW_API_KEY = "your-key"')
    return key


def run_workflow(frame, api_key=None, parameters=None, timeout=TIMEOUT, retries=RETRIES, jpeg_quality=JPEG_QUALITY):
    """Send one BGR frame, return the workflow's output dict for it (keys: predictions, inference_id, model_id).

    parameters: optional workflow parameters, e.g. {"confidence": 0.3}; sent next to the image in "inputs".
    The raw image is never logged or kept; error messages are trimmed.
    """
    api_key = api_key or get_api_key()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RoboflowError("could not JPEG-encode the frame")
    inputs = {"image": {"type": "base64", "value": base64.b64encode(buf.tobytes()).decode("ascii")}}
    if parameters:
        inputs.update(parameters)
    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last = None
    for attempt in range(retries + 1):
        try:
            r = _session.post(url, json={"inputs": inputs}, headers=headers, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = RoboflowError(f"network problem: {type(e).__name__}")
        else:
            if r.status_code in (401, 403):
                raise RoboflowAuthError(f"HTTP {r.status_code}: check ROBOFLOW_API_KEY (revoked, wrong, or no access to the workflow)")
            if r.status_code == 429 or r.status_code >= 500:
                last = RoboflowError(f"HTTP {r.status_code} from Roboflow")
            elif r.status_code != 200:
                raise RoboflowError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                try:
                    data = r.json()
                except ValueError:
                    raise RoboflowError("response was not JSON")
                if isinstance(data, dict) and isinstance(data.get("outputs"), list) and data["outputs"]:
                    data = data["outputs"]                  # the HTTP API wraps results as {"outputs": [ {...} ]}
                entry = data[0] if isinstance(data, list) and data else data
                if not isinstance(entry, dict) or "predictions" not in entry:
                    raise RoboflowError(f"unexpected response shape (keys: {list(entry)[:6] if isinstance(entry, dict) else type(entry).__name__})")
                return entry
        if attempt < retries:
            time.sleep(BACKOFF * (2 ** attempt))
    raise last


def parse_keypoints(entry, min_conf=0.5):
    """Collect keypoints from every prediction, keyed by keypoint class name (e.g. 'bridge', 'scroll', 'tip').

    Returns (points, box_conf, (sent_width, sent_height)):
      points   {name: (x, y, conf)} - only keypoints with conf >= min_conf; if a name appears twice the best one wins
      box_conf highest detection confidence among the predictions (0.0 if none)
    Tolerates the 'predictions' block being either {"image":..., "predictions":[...]} or a flat list.
    """
    block = entry.get("predictions") if isinstance(entry, dict) else None
    if isinstance(block, dict):
        preds, image = block.get("predictions") or [], block.get("image") or {}
    else:
        preds, image = block or [], {}
    size = (image.get("width"), image.get("height"))
    points, box_conf = {}, 0.0
    for p in preds:
        if not isinstance(p, dict):
            continue
        box_conf = max(box_conf, float(p.get("confidence", 0.0)))
        for k in p.get("keypoints") or []:
            name = k.get("class") or k.get("class_name")
            conf = float(k.get("confidence", 0.0))
            if name and conf >= min_conf and (name not in points or conf > points[name][2]):
                points[name] = (float(k["x"]), float(k["y"]), conf)
    return points, box_conf, size


# A real response captured from the workflow (256x192 test image); used by --selftest
SAMPLE = [{"predictions": {"image": {"width": 256, "height": 192}, "predictions": [
    {"width": 143.0, "height": 34.0, "x": 164.5, "y": 115.0, "confidence": 0.2758, "class_id": 0, "class": "bow",
     "keypoints": [{"class_id": 0, "class": "tip", "confidence": 0.9439, "x": 226.0, "y": 102.0}]},
    {"width": 74.0, "height": 171.0, "x": 147.0, "y": 105.5, "confidence": 0.2073, "class_id": 1, "class": "cello",
     "keypoints": [{"class_id": 0, "class": "bridge", "confidence": 0.9668, "x": 143.0, "y": 137.0},
                   {"class_id": 1, "class": "scroll", "confidence": 0.9722, "x": 154.0, "y": 38.0}]}]},
    "inference_id": "x", "model_id": "cello-posture-keypoints/1"}]


def selftest():
    pts, conf, size = parse_keypoints(SAMPLE[0], 0.5)
    assert set(pts) == {"tip", "bridge", "scroll"}, pts
    assert pts["bridge"][:2] == (143.0, 137.0) and size == (256, 192) and abs(conf - 0.2758) < 1e-6
    assert set(parse_keypoints(SAMPLE[0], 0.95)[0]) == {"bridge", "scroll"}               # confidence filter works
    assert parse_keypoints({"predictions": SAMPLE[0]["predictions"]["predictions"]})[0].keys() == pts.keys()   # flat list
    assert parse_keypoints({"predictions": {"image": {}, "predictions": []}}) == ({}, 0.0, (None, None))       # nothing found
    dup = {"predictions": {"image": {}, "predictions": [
        {"confidence": .5, "keypoints": [{"class": "tip", "confidence": .6, "x": 1, "y": 1}]},
        {"confidence": .9, "keypoints": [{"class": "tip", "confidence": .8, "x": 2, "y": 2}]}]}}
    assert parse_keypoints(dup, .5)[0]["tip"][:2] == (2.0, 2.0)                           # best duplicate wins
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    key = get_api_key()
    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"cannot read {a.image}")
    else:
        cap = cv2.VideoCapture(0)
        for _ in range(5):                       # let the camera settle
            ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("could not read from the webcam")
        frame = cv2.flip(frame, 1)               # same mirroring as the live script
    print("sending frame", frame.shape[1], "x", frame.shape[0])
    for i in range(a.repeat):
        t0 = time.time()
        entry = run_workflow(frame, key)
        ms = (time.time() - t0) * 1000
        pts, conf, size = parse_keypoints(entry, 0.0)
        print(f"[{i + 1}] {ms:.0f} ms  detections conf {conf:.2f}  image {size}  model {entry.get('model_id')}")
        for name, (x, y, c) in sorted(pts.items()):
            print(f"     {name:8s} ({x:6.1f}, {y:6.1f})  conf {c:.2f}")
        if i == 0:
            keys = {k: (type(v).__name__) for k, v in entry.items()}
            print("     top-level keys:", keys)


if __name__ == "__main__":
    main()