"""
detect_phone_hand_yolo.py
Detecção em tempo real:
 - pessoas (person)
 - celulares (cell phone)
 - mãos (hand model)
Associação: se mão próxima do celular e próxima da pessoa -> pessoa marcada como "using phone".
"""

import os
import math
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# -----------------------
# Configurações
# -----------------------
WEIGHT_GENERAL = "yolov8n.pt"            # detecta person, cell phone (ultralytics automático baixa se faltar)
WEIGHT_HAND = os.path.join("models", "hand_yolov8s.pt")  # BAIXE e coloque aqui (link no README)
CONF_THRESHOLD = 0.25     # limiar de confiança para boxes
HAND_PHONE_DIST = 80      # distância em pixels entre centro da mão e centro do phone para considerar "próximo"
PERSON_PHONE_DIST = 200   # distância máxima entre pessoa e celular para associação (ajuste conforme sua webcam)
DOWNSCALE = 1.0           # use <1.0 para reduzir tamanho do frame e acelerar

# -----------------------
# Utilitários
# -----------------------
def center_of_bbox_xyxy(bbox):
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clamp(v, mi, ma):
    return max(mi, min(ma, v))

# -----------------------
# Main
# -----------------------
def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Carrega modelos
    print("Carregando modelo geral (person, cell phone)...")
    model_gen = YOLO(WEIGHT_GENERAL)  # detecta classes COCO
    print("Carregando modelo de mãos...", WEIGHT_HAND)
    if not os.path.exists(WEIGHT_HAND):
        print(f"ERRO: modelo de mãos não encontrado em {WEIGHT_HAND}")
        print("Baixe um modelo de hands (ex: hand_yolov8s.pt) e coloque em models/hand_yolov8s.pt")
        return
    model_hand = YOLO(WEIGHT_HAND)

    # Map de nomes de classes do modelo geral
    names_gen = model_gen.names  # geralmente COCO names
    # model_hand.names pode variar dependendo do arquivo; assumimos que classes representam "hand" ou "left_hand"/"right_hand"
    names_hand = model_hand.names

    # Inicializa webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE != 1.0:
            frame = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)

        H, W, _ = frame.shape

        # ---- Inferência mão ----
        # usamos model_hand.predict com conf threshold para obter boxes de mãos
        hand_results = model_hand.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        # model_hand.predict retorna lista de Results; pegamos primeiro
        hands = []
        if len(hand_results) > 0:
            r_hand = hand_results[0]
            if r_hand.boxes is not None and len(r_hand.boxes) > 0:
                boxes_h = r_hand.boxes.xyxy.cpu().numpy()
                confs_h = r_hand.boxes.conf.cpu().numpy()
                cls_h = r_hand.boxes.cls.cpu().numpy().astype(int) if r_hand.boxes.cls is not None else None
                for idx, box in enumerate(boxes_h):
                    conf = float(confs_h[idx]) if confs_h is not None else 1.0
                    if conf < CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    cx, cy = center_of_bbox_xyxy([x1, y1, x2, y2])
                    hands.append({
                        "bbox": [x1, y1, x2, y2],
                        "bbox_i": list(map(int, [x1, y1, x2, y2])),
                        "center": (cx, cy),
                        "conf": conf,
                        "cls": int(cls_h[idx]) if cls_h is not None else 0
                    })

        # ---- Inferência geral (person + cell phone) ----
        gen_results = model_gen.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        persons = []
        phones = []
        if len(gen_results) > 0:
            r = gen_results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls_id in zip(boxes, confs, classes):
                    if conf < CONF_THRESHOLD:
                        continue
                    name = names_gen.get(int(cls_id), str(cls_id))
                    x1, y1, x2, y2 = map(float, box)
                    bx_i = list(map(int, [x1, y1, x2, y2]))
                    center = center_of_bbox_xyxy([x1, y1, x2, y2])
                    if name == "person":
                        persons.append({
                            "bbox": [x1, y1, x2, y2],
                            "bbox_i": bx_i,
                            "center": center,
                            "conf": float(conf),
                            "using": False
                        })
                    elif name == "cell phone" or name.lower().replace(" ", "") == "cellphone" or "phone" in name.lower():
                        phones.append({
                            "bbox": [x1, y1, x2, y2],
                            "bbox_i": bx_i,
                            "center": center,
                            "conf": float(conf)
                        })

        # ---- Associação hands -> phones -> persons ----
        associations = []  # tuples (hand_idx, phone_idx, person_idx)
        for hi, hand in enumerate(hands):
            hcenter = hand["center"]
            # buscar phones próximos
            best_phone_idx = None
            best_pdist = float("inf")
            for pi, phone in enumerate(phones):
                pd = euclidean(hcenter, phone["center"])
                if pd < best_pdist:
                    best_pdist = pd
                    best_phone_idx = pi
            if best_phone_idx is None:
                continue
            # ver limiar mão-phone
            if best_pdist > HAND_PHONE_DIST:
                continue
            # agora achar pessoa mais próxima ao phone (ou à mão)
            best_person_idx = None
            best_person_dist = float("inf")
            phone_center = phones[best_phone_idx]["center"]
            for per_i, per in enumerate(persons):
                d = euclidean(phone_center, per["center"])
                if d < best_person_dist:
                    best_person_dist = d
                    best_person_idx = per_i
            if best_person_idx is not None and best_person_dist <= PERSON_PHONE_DIST:
                # associação válida -> marcar pessoa como usando
                persons[best_person_idx]["using"] = True
                associations.append((hi, best_phone_idx, best_person_idx))

        # ---- Desenho visual ----
        # desenhar pessoas
        for p in persons:
            x1i, y1i, x2i, y2i = p["bbox_i"]
            if p["using"]:
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)  # vermelho
                cv2.putText(frame, "Using Phone", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (200,120,0), 2)  # laranja
                cv2.putText(frame, "Person", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,120,0), 2)

        # desenhar phones
        for ph in phones:
            x1i, y1i, x2i, y2i = ph["bbox_i"]
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 220, 0), 2)  # verde
            cx, cy = map(int, ph["center"])
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
            cv2.putText(frame, "Phone", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 2)

        # desenhar hands
        for h in hands:
            x1i, y1i, x2i, y2i = h["bbox_i"]
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 255, 0), 2)  # amarelo
            cx, cy = map(int, h["center"])
            cv2.circle(frame, (cx, cy), 3, (255,255,0), -1)

        # linhas de associação
        for (hi, pi, pei) in associations:
            hand_c = tuple(map(int, hands[hi]["center"]))
            phone_c = tuple(map(int, phones[pi]["center"]))
            person_c = tuple(map(int, persons[pei]["center"]))
            cv2.line(frame, hand_c, phone_c, (180,180,0), 2)
            cv2.line(frame, phone_c, person_c, (180,180,0), 2)

        # painel info
        total_phones = len(phones)
        total_using = sum(1 for p in persons if p["using"])
        cv2.rectangle(frame, (0,0), (360,90), (0,0,0), -1)
        cv2.putText(frame, f"Phones: {total_phones}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"People using: {total_using}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2)

        # fps
        now = time.time()
        cur_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        fps = 0.9 * fps + 0.1 * cur_fps
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # exibir
        cv2.imshow("YOLO Phone+Hand Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
