import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

def center_of_bbox(bbox):
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def is_point_inside_bbox_expanded(point, bbox, expand_ratio=0.2):
    # point: (cx, cy), bbox: [x1, y1, x2, y2]
    cx, cy = point
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    # expand bbox by expand_ratio em ambas as direções
    ex = w * expand_ratio
    ey = h * expand_ratio
    x1e, y1e = x1 - ex, y1 - ey
    x2e, y2e = x2 + ex, y2 + ey
    return (cx >= x1e) and (cx <= x2e) and (cy >= y1e) and (cy <= y2e)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def main():
    # Carrega modelo (será baixado automaticamente se necessário)
    model = YOLO("yolov8n.pt")  # modelo leve; troque para yolov8s/m para mais precisão

    class_names = model.names  # dicionário id -> nome
    # verificar se 'person' e 'cell phone' estão nas classes
    # COCO: 'person' existe e 'cell phone' normalmente existe (nome exato: "cell phone")
    print("Classes carregadas:", len(class_names))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Faz a inferência (retornará um objeto Results)
        # usamos model(frame)[0] (sem stream) para facilitar a manipulação
        results = model(frame)[0]

        # Inicializações
        phones = []    # lista de bboxes de celulares: each -> [x1,y1,x2,y2,conf]
        persons = []   # lista de dicts: {'bbox': [x1,y1,x2,y2], 'center':(cx,cy), 'used': False}

        if results.boxes is not None and len(results.boxes) > 0:
            # boxes_xyxy: (N,4) numpy
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for bbox, cls_id, conf in zip(boxes_xyxy, classes, confs):
                x1, y1, x2, y2 = bbox
                # garantia de inteiros para desenhar
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                name = class_names.get(cls_id, str(cls_id))

                if name == "cell phone" or name == "cellphone" or name.lower().replace(" ", "") == "cellphone":
                    phones.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_int': [x1i, y1i, x2i, y2i],
                        'conf': float(conf)
                    })
                elif name == "person":
                    center = center_of_bbox([x1, y1, x2, y2])
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_int': [x1i, y1i, x2i, y2i],
                        'center': center,
                        'used': False
                    })
                # caso queira, pode coletar outras classes

        # Associação celulares -> pessoas
        # Para cada phone, encontra a pessoa mais próxima (centro) e verifica sobreposição/contato
        associations = []  # lista de tuples (phone_idx, person_idx)
        for i, phone in enumerate(phones):
            phone_center = center_of_bbox(phone['bbox'])
            best_idx = None
            best_dist = float('inf')
            for j, person in enumerate(persons):
                # distância entre centros
                dist = euclidean(phone_center, person['center'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            # se encontrou pessoa candidata, valida se centro do phone está dentro da bbox expandida da pessoa
            if best_idx is not None:
                person_bbox = persons[best_idx]['bbox']
                if is_point_inside_bbox_expanded(phone_center, person_bbox, expand_ratio=0.20):
                    persons[best_idx]['used'] = True
                    associations.append((i, best_idx))

        # Desenhar todas as caixas:
        # Pessoas primeiro (azul), se usando celular -> vermelho
        for p in persons:
            x1i, y1i, x2i, y2i = p['bbox_int']
            if p['used']:
                # pessoa usando celular: caixa vermelha
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
                cv2.putText(frame, "Using Phone", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                # pessoa normal: azul
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 140, 0), 2)  # cor azul-arrojada
                cv2.putText(frame, "Person", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,140,0), 2)

        # Desenhar celulares (verde) e linhas de associação (se houver)
        for i, phone in enumerate(phones):
            x1i, y1i, x2i, y2i = phone['bbox_int']
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 220, 0), 2)
            cv2.putText(frame, "Cell Phone", (x1i, y1i - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 2)

        for (ph_idx, per_idx) in associations:
            phone_center = tuple(map(int, center_of_bbox(phones[ph_idx]['bbox'])))
            person_center = tuple(map(int, persons[per_idx]['center']))
            # desenha uma linha entre o phone e a pessoa associada
            cv2.line(frame, phone_center, person_center, (180, 180, 0), 2)

        # Contadores
        total_phones = len(phones)
        people_using = sum(1 for p in persons if p['used'])

        # FPS cálculo simples
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur_time - prev_time)) if prev_time else 0.0
        prev_time = cur_time

        # Mostrar informações no frame
        cv2.rectangle(frame, (0,0), (360,80), (0,0,0), -1)  # painel preto de fundo
        cv2.putText(frame, f"Phones (frame): {total_phones}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"People using phone: {people_using}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2)
        cv2.putText(frame, f"FPS: {int(1/(cur_time - prev_time)) if cur_time!=prev_time else 0}", (260,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Exibe a janela
        cv2.imshow("Detecção: Celulares + Pessoas usando celular", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
