"""Turn dataset/labels.json (from label_points.py) into a YOLO pose dataset.

Run from the same folder as label_points.py:
    python make_dataset.py
Creates  dataset_yolo/  with images/, labels/ (train + val) and data.yaml.

Keypoint order (fixed, do not change): 0 frog, 1 tip, 2 nut, 3 bridge
The first-position hand point is NOT trained; compute it after detection:
    hand = nut + 0.14 * (bridge - nut)
"""
import argparse, json, math, os, shutil
import cv2

KPTS = ["frog", "tip", "nut", "bridge"]
W, H = 640, 480


def auto_box(e):
    """Cello box from nut/bridge when no box was clicked: a rectangle along the nut-bridge axis,
    0.2 L above the nut, 0.55 L below the bridge, half-width 0.33 L (L = nut-bridge distance)."""
    n, b = e["nut"], e["bridge"]
    if n[2] == 0 or b[2] == 0:
        return None
    dx, dy = b[0] - n[0], b[1] - n[1]
    L = math.hypot(dx, dy)
    if L < 20:
        return None
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    a0, a1, hw = -0.2 * L, 1.55 * L, 0.33 * L
    xs, ys = [], []
    for a in (a0, a1):
        for s in (-hw, hw):
            xs.append(n[0] + ux * a + px * s)
            ys.append(n[1] + uy * a + py * s)
    return [min(xs), min(ys), max(xs), max(ys)]


def clip01(v):
    return min(1.0, max(0.0, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("dataset", "labels.json"))
    ap.add_argument("--images", default=os.path.join("dataset", "raw"))
    ap.add_argument("--out", default="dataset_yolo")
    ap.add_argument("--block", type=int, default=20, help="consecutive frames per block")
    ap.add_argument("--val_every", type=int, default=7, help="every Nth block goes to validation")
    a = ap.parse_args()

    data = json.load(open(a.labels))
    names = sorted(data)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", sp))
        os.makedirs(os.path.join(a.out, "labels", sp))

    counts = {"train": 0, "val": 0}
    negatives = 0
    kpt_vis = {k: 0 for k in KPTS}
    for idx, name in enumerate(names):
        e = data[name]
        # Split by BLOCKS of consecutive frames (frames 0.5 s apart look almost identical,
        # a random split would leak near-copies into the validation set)
        split = "val" if (idx // a.block) % a.val_every == a.val_every // 2 else "train"
        src = os.path.join(a.images, name)
        if not os.path.exists(src):
            print("missing image, skipped:", name); continue
        shutil.copy(src, os.path.join(a.out, "images", split, name))
        lines = []
        if e.get("cello"):
            box = e.get("box") or auto_box(e)
            if box is None:
                print("no usable box, treated as no-cello:", name)
            else:
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
                cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
                vals = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
                for k in KPTS:
                    x, y, v = e[k]
                    if v == 0 or not (0 <= x <= W and 0 <= y <= H):
                        vals.append("0 0 0")
                    else:
                        vals.append(f"{clip01(x / W):.6f} {clip01(y / H):.6f} {int(v)}")
                        kpt_vis[k] += 1
                lines.append(" ".join(vals))
        if not lines:
            negatives += 1
        open(os.path.join(a.out, "labels", split, os.path.splitext(name)[0] + ".txt"), "w").write("\n".join(lines))
        counts[split] += 1

    yaml = (
        f"path: {os.path.abspath(a.out)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [0, 1, 2, 3]\n"
        "names:\n  0: cello\n"
    ).replace("\\", "/")
    open(os.path.join(a.out, "data.yaml"), "w").write(yaml)
    print(f"train {counts['train']}  val {counts['val']}  no-cello images {negatives}")
    print("labelled keypoints:", kpt_vis)
    print("wrote", os.path.join(a.out, "data.yaml"))


if __name__ == "__main__":
    main()