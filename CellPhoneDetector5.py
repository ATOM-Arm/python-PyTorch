import os
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ================================
# CONFIGURAÇÕES DE OTIMIZAÇÃO
# ================================
DOWNSCALE = 0.75             # fatores bons: 1.0, 0.75, 0.5
FRAME_SLEEP = 0.01           # pequeno atraso de 10ms para suavizar travamentos
CONF_THRESHOLD = 0.35
HAND_PHONE_DIST = 80
PERSON_PHONE_DIST = 200

# MODELOS
WEIGHT_GENERAL = "yolov8n.pt"
WEIGHT_HAND = os.path.join("models", "hand_yolov8s.pt")


def center_of(b):
    x1, y1, x2, y2 = b
    return ((x1+x2)/2, (y1+y2)/2)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    # Carrega YOLO otimizado (lazy load)
    model_gen = YOLO(WEIGHT_GENERAL)
    model_hand = YOLO(WEIGHT_HAND)

    # Webcam com buffer rápido
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    prev = time.time()

    while True:
        # ----- CAPTURA OTIMIZADA -----
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)

        # ----- INFERÊNCIA (SEM GRADIENTES) -----
        with torch.no_grad():
            res_gen = model_gen(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            res_hand = model_hand(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        persons = []
        phones = []
        hands = []

        # ----- PROCESSA RESULTADOS -----
        if res_gen.boxes is not None:
            for box in res_gen.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]

                name = model_gen.names[cls_id]

                if name == "person":
                    persons.append({
                        "center": center_of([x1, y1, x2, y2]),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "using": False
                    })

                elif "phone" in name.lower():
                    phones.append({
                        "center": center_of([x1, y1, x2, y2]),
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })

        if res_hand.boxes is not None:
            for box in res_hand.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                hands.append({
                    "center": center_of([x1, y1, x2, y2]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })

        # ----- ASSOCIAÇÃO MÃO → CELULAR → PESSOA -----
        for h in hands:
            # Para cada mão, encontre celular mais próximo
            nearest_phone = None
            best_dp = 9999

            for p in phones:
                d = dist(h["center"], p["center"])
                if d < best_dp:
                    best_dp = d
                    nearest_phone = p

            if nearest_phone and best_dp < HAND_PHONE_DIST:
                # Encontrar pessoa mais próxima do celular
                nearest_person = None
                best_dperson = 9999

                for person in persons:
                    dpp = dist(nearest_phone["center"], person["center"])
                    if dpp < best_dperson:
                        best_dperson = dpp
                        nearest_person = person

                if nearest_person and best_dperson < PERSON_PHONE_DIST:
                    nearest_person["using"] = True

        # ----- DESENHO -----
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            color = (0, 0, 255) if p["using"] else (0, 255, 0)
            label = "USING PHONE" if p["using"] else "Person"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for ph in phones:
            x1, y1, x2, y2 = ph["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, "Phone", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,128,0), 2)

        for h in hands:
            x1, y1, x2, y2 = h["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Hand", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # ----- EXIBIÇÃO -----
        fps = 1 / (time.time() - prev)
        prev = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Detector de Celular + Mao + Pessoa (Optimized)", frame)

        # Pequeno delay para evitar travamentos
        time.sleep(FRAME_SLEEP)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
