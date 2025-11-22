# threaded_fast.py - Melhor fluidez
import os, time, math, threading
import cv2, torch
from ultralytics import YOLO

WEIGHT_PHONE = os.path.join("models","cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE): WEIGHT_PHONE = "yolov8n.pt"
WEIGHT_HAND = os.path.join("models","hand_yolov8s.pt")
CONF = 0.35
SKIP = 3
SLEEP = 0.008

model_phone = YOLO(WEIGHT_PHONE)
model_hand = YOLO(WEIGHT_HAND) if os.path.exists(WEIGHT_HAND) else None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cv2.setUseOptimized(True)

frame_buf = None
lock = threading.Lock()
last_hands = []
last_phones = []
frame_idx = 0

def worker(frame):
    global last_hands, last_phones
    with torch.no_grad():
        r_hand = model_hand(frame, conf=CONF, verbose=False)[0] if model_hand else None
        r_phone = model_phone(frame, conf=CONF, verbose=False)[0]
    with lock:
        last_hands = r_hand.boxes if (r_hand and r_hand.boxes is not None) else []
        last_phones = r_phone.boxes if (r_phone and r_phone.boxes is not None) else []

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_display = frame.copy()

    if frame_idx % SKIP == 0:
        # spawn worker thread
        t = threading.Thread(target=worker, args=(frame.copy(),))
        t.daemon = True
        t.start()

    with lock:
        for box in last_hands:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cv2.rectangle(frame_display, (int(x1),int(y1)), (int(x2),int(y2)), (255,165,0),2)
        for box in last_phones:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cv2.rectangle(frame_display, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0),2)

    cv2.imshow("threaded_fast", frame_display)
    if cv2.waitKey(1) & 0xFF == 27: break
    frame_idx += 1
    time.sleep(SLEEP)

cap.release(); cv2.destroyAllWindows()
