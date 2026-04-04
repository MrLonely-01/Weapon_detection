from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/weapon_detector/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Weapon Detection", annotated_frame)

    if len(results[0].boxes) > 0:
        print("⚠ Weapon Detected")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
