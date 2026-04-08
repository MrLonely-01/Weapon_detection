import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# -------------------------------
# LOAD MODELS
# -------------------------------
mask_model = load_model("mask_detector.model")
weapon_model = YOLO("best.pt")

# Load face detector (OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# START VIDEO
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize frame (optional for speed)
    frame_resized = cv2.resize(frame, (640, 480))

    # -------------------------------
    # WEAPON DETECTION (FULL FRAME)
    # -------------------------------
    weapon_results = weapon_model(frame_resized)

    annotated_frame = weapon_results[0].plot()

    weapon_detected = False

    for box in weapon_results[0].boxes:
        conf = float(box.conf)
        if conf > 0.6:
            weapon_detected = True

    # -------------------------------
    # FACE DETECTION
    # -------------------------------
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # -------------------------------
    # MASK DETECTION (ON FACES)
    # -------------------------------
    for (x, y, w, h) in faces:
        face = frame_resized[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (224, 224))
        except:
            continue

        face = np.array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, no_mask) = mask_model.predict(face)[0]

        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw face box
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated_frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # -------------------------------
    # ALERT SYSTEM
    # -------------------------------
    if weapon_detected:
        cv2.putText(annotated_frame, "⚠ WEAPON DETECTED!",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # -------------------------------
    # DISPLAY OUTPUT
    # -------------------------------
    cv2.imshow("Mask + Weapon Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
