
import cv2

# Open default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access webcam")
    exit()

print("Webcam connected. Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
