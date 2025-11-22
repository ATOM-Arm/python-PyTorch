# gpu_version.py - Forçar GPU
import os, time, torch, cv2
from ultralytics import YOLO

WEIGHT_PHONE = os.path.join("models","cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE): WEIGHT_PHONE = "yolov8n.pt"
WEIGHT_HAND = os.path.join("models","hand_yolov8s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_phone = YOLO(WEIGHT_PHONE)
model_hand = YOLO(WEIGHT_HAND) if os.path.exists(WEIGHT_HAND) else None

# ultralytics models can be moved to device
try:
    model_phone.to(device)
    if model_hand: model_hand.to(device)
except Exception as e:
    print("Warning moving models to device failed:", e)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret: break

    with torch.no_grad():
        rph = model_phone(frame, conf=0.35, device=device)[0]
        rh = model_hand(frame, conf=0.35, device=device)[0] if model_hand else None

    # desenho simples
    if rh and rh.boxes is not None:
        for b in rh.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,165,0),2)
    if rph and rph.boxes is not None:
        for b in rph.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,220,0),2)

    cv2.imshow("gpu_version", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
