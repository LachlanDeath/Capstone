"""Click-to-label tool for the cello keypoint model.

Run from the folder that contains dataset/raw  (e.g. C:\\Users\\maxsl\\Capstone\\mediapipe):
    python label_points.py
Options:  --images dataset/raw   --out dataset/labels.json   --scale 1.5

For every image you click, in this order:
    1. BRIDGE  - the exact CENTRE of the bridge (use the magnifier in the top-right corner)
    2. NUT     - where the strings leave the fingerboard and enter the pegbox (centre of the neck)
    3. FROG    - the bow end held by the right hand
    4. TIP     - the far thin end of the bow
Mouse:  LEFT click = clearly visible      RIGHT click = hidden/blurred but you can estimate where it is
Keys:   S = this point is not in the picture      U = undo      B = back one image      Q = save and quit
If the cello is only partly in the frame (nut or bridge missing but not both) you are also asked to click the
cello's TOP-LEFT and BOTTOM-RIGHT corners. If neither bridge nor nut is visible the image counts as "no cello".
Progress is saved after every image, so you can quit any time and re-run to continue where you stopped.
"""
import argparse, glob, json, os
import cv2

STEPS = {
    "bridge": ("Click the CENTRE of the BRIDGE", (255, 0, 255)),
    "nut": ("Click the NUT (strings enter the pegbox)", (255, 128, 0)),
    "frog": ("Click the bow FROG (right-hand end)", (0, 255, 0)),
    "tip": ("Click the bow TIP (far end)", (0, 255, 255)),
    "box_tl": ("Click the cello's TOP-LEFT corner", (255, 255, 255)),
    "box_br": ("Click the cello's BOTTOM-RIGHT corner", (255, 255, 255)),
}
BASE = ["bridge", "nut", "frog", "tip"]


class Labeler:
    def __init__(self, files, out, scale):
        self.files, self.out, self.scale = files, out, scale
        self.data = json.load(open(out)) if os.path.exists(out) else {}
        self.i = next((k for k, f in enumerate(files) if os.path.basename(f) not in self.data), len(files))
        self.cur = {}
        self.steps = list(BASE)
        self.mouse = (0, 0)
        self.img = None
        self.load()

    def load(self):
        self.cur, self.steps = {}, list(BASE)
        self.img = cv2.imread(self.files[self.i]) if self.i < len(self.files) else None

    def click(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and len(self.cur) < len(self.steps):
            v = 2 if event == cv2.EVENT_LBUTTONDOWN else 1
            self.set_point(round(x / self.scale, 1), round(y / self.scale, 1), v)

    def set_point(self, x, y, v):
        name = self.steps[len(self.cur)]
        self.cur[name] = [x, y, v]
        self.after_point()

    def after_point(self):
        if len(self.cur) == len(BASE) and len(self.steps) == len(BASE):
            has_b, has_n = self.cur["bridge"][2] > 0, self.cur["nut"][2] > 0
            if has_b != has_n:                      # only one of them in frame -> need a box
                self.steps += ["box_tl", "box_br"]
        if len(self.cur) == len(self.steps):
            self.finish()

    def finish(self):
        c = self.cur
        cello = c["bridge"][2] > 0 or c["nut"][2] > 0
        entry = {k: c.get(k, [0, 0, 0]) for k in BASE}
        entry["cello"] = cello
        entry["box"] = None
        if "box_tl" in c and "box_br" in c and c["box_tl"][2] and c["box_br"][2]:
            entry["box"] = [c["box_tl"][0], c["box_tl"][1], c["box_br"][0], c["box_br"][1]]
        self.data[os.path.basename(self.files[self.i])] = entry
        json.dump(self.data, open(self.out, "w"), indent=1)
        self.i += 1
        self.load()

    def key(self, k):
        if k in (ord("s"), ord("S")) and len(self.cur) < len(self.steps):
            self.cur[self.steps[len(self.cur)]] = [0, 0, 0]
            self.after_point()
        elif k in (ord("u"), ord("U")) and self.cur:
            self.cur.pop(list(self.cur)[-1])
            if len(self.cur) < len(BASE): self.steps = list(BASE)
        elif k in (ord("b"), ord("B")) and self.i > 0:
            self.i -= 1
            self.data.pop(os.path.basename(self.files[self.i]), None)
            self.load()
        elif k in (ord("q"), ord("Q"), 27):
            return False
        return True

    def draw(self):
        s = self.scale
        view = cv2.resize(self.img, None, fx=s, fy=s)
        for name, p in self.cur.items():
            if p[2] > 0 and name in BASE:
                c = (int(p[0] * s), int(p[1] * s))
                cv2.circle(view, c, 6, STEPS[name][1], -1 if p[2] == 2 else 2)
                cv2.circle(view, c, 8, (0, 0, 0), 1)
        if len(self.cur) < len(self.steps):
            msg, col = STEPS[self.steps[len(self.cur)]]
            txt = f"{self.i + 1}/{len(self.files)}   {msg}   (right-click=estimated, S=not visible, U=undo)"
            cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(view, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        # magnifier: 40x40 px of the original image around the cursor, shown at 4x in the top-right corner
        mx, my = int(self.mouse[0] / s), int(self.mouse[1] / s)
        h, w = self.img.shape[:2]
        pad = cv2.copyMakeBorder(self.img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = pad[max(0, min(my, h)):max(0, min(my, h)) + 40, max(0, min(mx, w)):max(0, min(mx, w)) + 40]
        if crop.shape[0] == 40 and crop.shape[1] == 40:
            z = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            cv2.drawMarker(z, (80, 80), (0, 0, 255), cv2.MARKER_CROSS, 22, 1)
            view[34:194, view.shape[1] - 164:view.shape[1] - 4] = z
            cv2.rectangle(view, (view.shape[1] - 164, 34), (view.shape[1] - 4, 194), (255, 255, 255), 1)
        return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join("dataset", "raw"))
    ap.add_argument("--out", default=os.path.join("dataset", "labels.json"))
    ap.add_argument("--scale", type=float, default=1.5)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    if not files:
        raise SystemExit(f"No .jpg files in {a.images} - run this from the folder that contains dataset/raw")
    lab = Labeler(files, a.out, a.scale)
    if lab.i >= len(files):
        print("All images are already labelled:", a.out); return
    print(f"{len(lab.data)} already labelled, {len(files) - len(lab.data)} to go. Window opens now.")
    cv2.namedWindow("label")
    cv2.setMouseCallback("label", lab.click)
    while lab.i < len(files):
        cv2.imshow("label", lab.draw())
        k = cv2.waitKey(20) & 0xFF
        if k != 255 and not lab.key(k):
            break
    cv2.destroyAllWindows()
    print(f"Saved {len(lab.data)} labelled images to {a.out}")


if __name__ == "__main__":
    main()