# super_fast_skip.py - Super rápida
import os, time
import cv2
from ultralytics import YOLO

WEIGHT_PHONE = os.path.join("models","cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE): WEIGHT_PHONE = "yolov8n.pt"
WEIGHT_HAND = os.path.join("models","hand_yolov8s.pt")
SKIP_FRAMES = 4

phone = YOLO(WEIGHT_PHONE)
hand = YOLO(WEIGHT_HAND) if os.path.exists(WEIGHT_HAND) else None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_hands = []
last_phones = []
cnt = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    if cnt % SKIP_FRAMES == 0:
        rph = phone(frame, conf=0.4)[0]
        last_phones = rph.boxes if (rph and rph.boxes is not None) else []
        if hand:
            rh = hand(frame, conf=0.4)[0]
            last_hands = rh.boxes if (rh and rh.boxes is not None) else []

    # desenha últimos
    for b in last_hands:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,165,0),2)
    for b in last_phones:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)

    cv2.imshow("super_fast_skip", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    cnt += 1

cap.release(); cv2.destroyAllWindows()
