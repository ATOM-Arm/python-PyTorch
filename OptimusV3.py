# lightweight.py - Mais leve
import os, time
import cv2
from ultralytics import YOLO

WEIGHT_PHONE = os.path.join("models","cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE): WEIGHT_PHONE = "yolov8n.pt"
WEIGHT_HAND = os.path.join("models","hand_yolov8s.pt")
CONF = 0.4
FRAME_DELAY = 0.02

model_phone = YOLO(WEIGHT_PHONE)
model_hand = YOLO(WEIGHT_HAND) if os.path.exists(WEIGHT_HAND) else None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret: break

    if model_hand:
        rh = model_hand(frame, conf=CONF, verbose=False)[0]
        hands = rh.boxes if (rh and rh.boxes is not None) else []
    else:
        hands = []

    rp = model_phone(frame, conf=CONF, verbose=False)[0]
    phones = rp.boxes if (rp and rp.boxes is not None) else []

    for b in hands:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,165,0),2)
    for b in phones:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,220,0),2)

    cv2.imshow("lightweight", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    time.sleep(FRAME_DELAY)

cap.release(); cv2.destroyAllWindows()
