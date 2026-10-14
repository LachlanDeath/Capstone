"""Turn Roboflow's guitar mask (a polygon outline) into numbered landmarks, in the same
format and drawing style that pose_botharms.py uses for the body.

What matches pose_botharms.py:
  * a dict of  id -> (x, y)  in integer pixels of the MIRRORED frame  (its `pts` dict)
  * a CONNECTIONS list of (id_a, id_b) pairs, drawn as lines (thickness 2) with red dots (radius 5)
  * a confidence threshold that plays the role of its `visibility > 0.5` check
  * ids start at 100 so they never collide with MediaPipe's pose ids (0-32) when merged
"""
import cv2
import numpy as np

# ---- Landmark ids (head of the guitar -> end of the body) -----------------
HEAD, NECK_MID, NECK_BODY, BODY_MID, BODY_END = 100, 101, 102, 103, 104
GUITAR_KEYPOINTS = [HEAD, NECK_MID, NECK_BODY, BODY_MID, BODY_END]
GUITAR_NAMES = {
    HEAD: "headstock end",
    NECK_MID: "neck middle",
    NECK_BODY: "neck/body joint",
    BODY_MID: "body middle",
    BODY_END: "body end",
}
GUITAR_CONNECTIONS = [
    (HEAD, NECK_MID),
    (NECK_MID, NECK_BODY),
    (NECK_BODY, BODY_MID),
    (BODY_MID, BODY_END),
]

# ---- Colors (BGR), same drawing style as pose_botharms.py ------------------
GUITAR_COLOR = (0, 255, 255)   # yellow lines, so the guitar is easy to tell from the body
JOINT_COLOR = (0, 0, 255)      # same red dots as pose_botharms.py


# ---- Geometry --------------------------------------------------------------
def _resample_closed(poly, n=400):
    """Re-space polygon vertices evenly along the outline so dense vertices don't bias the math."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    if d[-1] < 1e-6:
        return None
    targets = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, d, closed[:, 0]),
                            np.interp(targets, d, closed[:, 1])])


def _profile(centered, u, v, bins):
    """Width of the shape (and where its middle line sits) in slices along the long axis."""
    t = centered @ u
    s = centered @ v
    t0, t1 = t.min(), t.max()
    edges = np.linspace(t0, t1, bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
    width = np.full(bins, np.nan)
    mid = np.full(bins, np.nan)
    for b in range(bins):
        sel = s[idx == b]
        if len(sel) >= 2:
            width[b] = sel.max() - sel.min()
            mid[b] = (sel.max() + sel.min()) / 2
    good = ~np.isnan(width)
    if good.sum() < bins // 2:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.interp(centers, centers[good], width[good])
    mid = np.interp(centers, centers[good], mid[good])
    return t0, t1, edges, centers, width, mid


def derive_landmarks(polygon, bins=20):
    """polygon: list of (x, y). Returns {id: (x, y)} in float pixels, or None if the shape is unusable.

    Method: find the long axis of the mask, decide which end is the body (the wider end),
    then place points along the guitar's middle line. The neck/body joint is where the
    mask first becomes wide and stays wide. These points are ESTIMATED from the outline,
    not detected by a keypoint model.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 6:
        return None
    pts = _resample_closed(poly)
    if pts is None:
        return None

    c = pts.mean(axis=0)
    centered = pts - c
    _, vecs = np.linalg.eigh(np.cov(centered.T))
    u = vecs[:, 1]                       # long axis (eigh sorts ascending)
    v = np.array([-u[1], u[0]])

    prof = _profile(centered, u, v, bins)
    if prof is None:
        return None
    k = max(2, bins // 4)
    if prof[4][:k].mean() > prof[4][-k:].mean():       # wider end is the body: flip so head is first
        u, v = -u, -v
        prof = _profile(centered, u, v, bins)
        if prof is None:
            return None
    t0, t1, edges, centers, width, mid = prof

    length = t1 - t0
    if length < 20:                      # too small to mean anything
        return None

    # Neck/body joint: first slice that is wide and stays wide for the next two slices
    above = width > 0.55 * width.max()
    t_joint = None
    for b in range(bins - 2):
        if above[b:b + 3].all():
            t_joint = edges[b]
            break
    if t_joint is None or not (t0 + 0.15 * length < t_joint < t0 + 0.85 * length):
        t_joint = t0 + 0.45 * length     # fallback guess when the outline is unusual

    def at(tv):
        sv = np.interp(tv, centers, mid)
        p = c + tv * u + sv * v
        return float(p[0]), float(p[1])

    return {
        HEAD: at(t0),
        NECK_MID: at((t0 + t_joint) / 2),
        NECK_BODY: at(t_joint),
        BODY_MID: at((t_joint + t1) / 2),
        BODY_END: at(t1),
    }


# ---- Helpers that mirror pose_botharms.py ----------------------------------
def landmarks_to_pts(landmarks):
    """Integer pixel dict, exactly like the `pts` dict in pose_botharms.py."""
    return {i: (int(x), int(y)) for i, (x, y) in landmarks.items()}


def to_normalized(pts, w, h):
    """0-1 coordinates, like MediaPipe's landmark.x / landmark.y."""
    return {i: (x / w, y / h) for i, (x, y) in pts.items()}


def guitar_from_result(result, min_confidence=0.5):
    """Roboflow result -> (pts dict, confidence) for the most confident guitar, or (None, conf)."""
    preds = [p for p in (result or {}).get("predictions", [])
             if p.get("confidence", 0) >= min_confidence and p.get("points")]
    if not preds:
        return None, 0.0
    best = max(preds, key=lambda p: p["confidence"])
    landmarks = derive_landmarks([(q["x"], q["y"]) for q in best["points"]])
    if landmarks is None:
        return None, best["confidence"]
    return landmarks_to_pts(landmarks), best["confidence"]


def draw_guitar(frame, pts, color=GUITAR_COLOR):
    """Same calls pose_botharms.py uses: lines (thickness 2) then red dots (radius 5)."""
    for a, b in GUITAR_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)
    for i in GUITAR_KEYPOINTS:
        if i in pts:
            cv2.circle(frame, pts[i], 5, JOINT_COLOR, -1)