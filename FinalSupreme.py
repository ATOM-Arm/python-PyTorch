import os, time, math
import cv2
import torch
import winsound
from ultralytics import YOLO

# ========================
# CONFIGURAÇÕES
# ========================
WEIGHT_PHONE = os.path.join("models", "cell_yolov8s.pt")
if not os.path.exists(WEIGHT_PHONE):
    WEIGHT_PHONE = "yolov8n.pt"  # fallback automático

WEIGHT_HAND = os.path.join("models", "hand_yolov8s.pt")

CONF = 0.35
DOWNSCALE = 0.75         # reduz tamanho → aumenta FPS
FRAME_SLEEP = 0.01       # pequena pausa → reduz travamento
ALERT_COOLDOWN = 1.5     # segundos entre alertas sonoros

# alerta sonoro (Windows)
def play_alert():
    winsound.Beep(1000, 250)  # frequência, duração(ms)

# cálculo geometria
def center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2, (y1+y2)/2)

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ========================
# INICIALIZAÇÃO
# ========================
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Phone model:", WEIGHT_PHONE)
print("Hand model:", WEIGHT_HAND)

model_phone = YOLO(WEIGHT_PHONE)
model_hand = YOLO(WEIGHT_HAND) if os.path.exists(WEIGHT_HAND) else None
if model_hand is None:
    print("⚠ AVISO: Modelo de mãos não encontrado. Hand detection desativada.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()
last_alert = 0

# ========================
# LOOP PRINCIPAL
# ========================
while True:
    ret, frame = cap.read()
    if not ret: break

    if DOWNSCALE != 1.0:
        frame = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)

    with torch.no_grad():
        r_phone = model_phone(frame, conf=CONF, verbose=False)[0]
        r_hand = model_hand(frame, conf=CONF, verbose=False)[0] if model_hand else None

    persons = []
    phones = []
    hands = []

    # ----------------------
    # DETECÇÃO DE PESSOAS & CELULARES
    # ----------------------
    if r_phone.boxes is not None:
        for box in r_phone.boxes:
            cls = int(box.cls)
            x1,y1,x2,y2 = map(float, box.xyxy[0])

            if cls == 0:
                persons.append({
                    "bbox": (int(x1),int(y1),int(x2),int(y2)),
                    "center": center((x1,y1,x2,y2)),
                    "using": False
                })
            else:
                name = model_phone.names.get(cls, "")
                if "phone" in name.lower() or "cell" in name.lower():
                    phones.append({
                        "bbox": (int(x1),int(y1),int(x2),int(y2)),
                        "center": center((x1,y1,x2,y2))
                    })

    # ----------------------
    # DETECÇÃO DE MÃOS
    # ----------------------
    if r_hand and r_hand.boxes is not None:
        for box in r_hand.boxes:
            x1,y1,x2,y2 = map(float, box.xyxy[0])
            hands.append({
                "bbox": (int(x1),int(y1),int(x2),int(y2)),
                "center": center((x1,y1,x2,y2))
            })

    # ----------------------
    # ASSOCIAÇÃO MÃO → CELULAR → PESSOA
    # ----------------------
    someone_using_phone = False

    for hand in hands:
        # mão -> celular
        best_phone = None
        best_dist = 9999

        for i,p in enumerate(phones):
            d = dist(hand["center"], p["center"])
            if d < best_dist:
                best_dist = d
                best_phone = i

        if best_phone is not None and best_dist < 80:
            # celular → pessoa
            best_person = None
            best_pd = 9999
            for j,per in enumerate(persons):
                d2 = dist(phones[best_phone]["center"], per["center"])
                if d2 < best_pd:
                    best_pd = d2
                    best_person = j

            if best_person is not None and best_pd < 200:
                persons[best_person]["using"] = True
                someone_using_phone = True

    # ----------------------
    # ALERTA SONORO
    # ----------------------
    if someone_using_phone and (time.time() - last_alert) > ALERT_COOLDOWN:
        play_alert()
        last_alert = time.time()

    # ----------------------
    # DESENHAR NA TELA
    # ----------------------
    for p in persons:
        x1,y1,x2,y2 = p["bbox"]
        color = (0,0,255) if p["using"] else (0,200,255)
        label = "Using Phone" if p["using"] else "Person"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    for ph in phones:
        x1,y1,x2,y2 = ph["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,"Phone",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    for h in hands:
        x1,y1,x2,y2 = h["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,165,0),2)
        cv2.putText(frame,"Hand",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,165,0),2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

    cv2.imshow("Optimized + Alert", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

    time.sleep(FRAME_SLEEP)

cap.release()
cv2.destroyAllWindows()
