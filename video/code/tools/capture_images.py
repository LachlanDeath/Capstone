import os
import time
import cv2

OUT_DIR = os.path.join("dataset", "raw")
AUTO_EVERY_SEC = 0.5   # seconds between saves while auto-record is on
os.makedirs(OUT_DIR, exist_ok=True)

# Continue numbering if the folder already has images
count = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".jpg")])

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())
print("SPACE = save one frame | R = toggle auto-record | Q = quit")

auto = False
last_save = 0.0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    # NOTE: frames are saved UNMIRRORED (raw camera view) on purpose.
    now = time.time()
    save = False
    if auto and now - last_save >= AUTO_EVERY_SEC:
        save = True

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        save = True
    elif key == ord("r"):
        auto = not auto
        print("Auto-record:", "ON" if auto else "OFF")
    elif key == ord("q"):
        break

    if save:
        path = os.path.join(OUT_DIR, f"img_{count:05d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
        last_save = now

    # Draw HUD on a copy so saved images stay clean
    display = frame.copy()
    status = f"saved: {count}   auto: {'ON' if auto else 'OFF'}"
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if auto else (0, 255, 0), 2)
    cv2.imshow("Capture", display)

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} images in {OUT_DIR}")