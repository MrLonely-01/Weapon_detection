from ultralytics import YOLO

# load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# train model
model.train(
    data="weapon.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="weapon_detector"
)
