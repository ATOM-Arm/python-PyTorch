# optimized.py - versão otimizada
import os, time, math
import cv2
import torch
from ultralytics import YOLO

# CONFIG
WEIGHT_PHONE = os.path.join("models", "cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE):
    WEIGHT_PHONE = "yolov8n.pt"   # fallback (auto-download)

WEIGHT_HAND = os.path.join("models", "hand_yolov8s.pt")
CONF = 0.35
DOWNSCALE = 0.75
FRAME_SLEEP = 0.01

def center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2, (y1+y2)/2)

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def main():
    print("Phone weight:", WEIGHT_PHONE)
    print("Hand weight:", WEIGHT_HAND if os.path.exists(WEIGHT_HAND) else "NOT FOUND")
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

    model_phone = YOLO(WEIGHT_PHONE)
    if os.path.exists(WEIGHT_HAND):
        model_hand = YOLO(WEIGHT_HAND)
    else:
        model_hand = None
        print("Warning: hand model not found. Hand detection disabled.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        if DOWNSCALE != 1.0:
            frame = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)

        with torch.no_grad():
            res_phone = model_phone(frame, conf=CONF, verbose=False)[0]
            res_hand = model_hand(frame, conf=CONF, verbose=False)[0] if model_hand else None

        phones = []
        hands = []
        persons = []

        if res_phone.boxes is not None:
            for box in res_phone.boxes:
                cls = int(box.cls)
                x1,y1,x2,y2 = map(float, box.xyxy[0])
                if cls == 0:  # person
                    persons.append({"bbox":(int(x1),int(y1),int(x2),int(y2)), "center":center((x1,y1,x2,y2)), "using":False})
                else:
                    # try to detect phone using name string fallback
                    name = model_phone.names.get(cls, "")
                    if "phone" in name.lower() or "cell" in name.lower():
                        phones.append({"bbox":(int(x1),int(y1),int(x2),int(y2)), "center":center((x1,y1,x2,y2))})

        if res_hand is not None and res_hand.boxes is not None:
            for box in res_hand.boxes:
                x1,y1,x2,y2 = map(float, box.xyxy[0])
                hands.append({"bbox":(int(x1),int(y1),int(x2),int(y2)), "center":center((x1,y1,x2,y2))})

        # associação simples: mão -> phone -> pessoa
        for h in hands:
            bestp, bestd = None, 1e9
            for i,p in enumerate(phones):
                d = dist(h["center"], p["center"])
                if d < bestd:
                    bestd = d; bestp = i
            if bestp is not None and bestd < 80:
                best_person, best_pd = None, 1e9
                for pi,per in enumerate(persons):
                    d2 = dist(phones[bestp]["center"], per["center"])
                    if d2 < best_pd: best_pd = d2; best_person = pi
                if best_person is not None and best_pd < 200:
                    persons[best_person]["using"] = True

        # desenho
        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            color = (0,0,255) if p["using"] else (0,180,255)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, "Using Phone" if p["using"] else "Person", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for ph in phones:
            x1,y1,x2,y2 = ph["bbox"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
            cv2.putText(frame,"Phone",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0),2)
        for h in hands:
            x1,y1,x2,y2 = h["bbox"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)

        fps = 1.0/(time.time()-prev) if prev else 0.0
        prev = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200),2)
        cv2.imshow("optimized", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        time.sleep(FRAME_SLEEP)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
