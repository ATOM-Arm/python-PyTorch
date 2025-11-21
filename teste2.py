from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("https://ultralytics.com/images/bus.jpg").show()
