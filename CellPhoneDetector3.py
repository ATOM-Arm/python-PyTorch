import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import math
import time

# --------------------------
# Utils
# --------------------------
def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_near_hands(phone_center, hands_points, threshold=80):
    """
    phone_center: (cx, cy)
    hands_points: lista de pontos [(x,y), ...]
    threshold: pixel distance
    """
    if not hands_points:
        return False

    for p in hands_points:
        if euclidean(phone_center, p) < threshold:
            return True
    return False


# --------------------------
# Main
# --------------------------
def main():
    # YOLO
    model = YOLO("yolov8n.pt")

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    class_names = model.names

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar webcam")
        return

    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # --------------------------
        # MediaPipe HANDS primeiros
        # --------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)

        hands_points = []  # lista de pontos de todas as mãos

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    hands_points.append((px, py))
                # desenhar mãos
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # --------------------------
        # YOLO Detection
        # --------------------------
        results = model(frame)[0]

        phones = []
        persons = []

        if results.boxes is not None and len(results.boxes) > 0:

            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for bbox, cls_id, conf in zip(boxes_xyxy, classes, confs):
                x1, y1, x2, y2 = bbox
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                cls_name = class_names.get(cls_id, "")

                if cls_name == "cell phone":
                    phones.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_int': [x1i, y1i, x2i, y2i],
                        'center': center_of_bbox([x1, y1, x2, y2]),
                        'conf': conf
                    })

                if cls_name == "person":
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_int': [x1i, y1i, x2i, y2i],
                        'center': center_of_bbox([x1, y1, x2, y2]),
                        'using': False
                    })

        # --------------------------
        # Associação Telefones ↔ Pessoas ↔ Mãos
        # --------------------------
        associations = []

        for i, phone in enumerate(phones):
            phone_center = phone["center"]

            # 1. ver se celular está perto de alguma mão
            near_hand = is_near_hands(phone_center, hands_points, threshold=70)

            if not near_hand:
                continue  # se não estiver perto da mão, ignora

            # 2. encontrar pessoa mais próxima
            best_idx = None
            best_dist = float('inf')
            for j, per in enumerate(persons):
                dist = euclidean(phone_center, per['center'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx is not None:
                persons[best_idx]['using'] = True
                associations.append((i, best_idx))

        # --------------------------
        # DESENHO: Pessoas e Telefones
        # --------------------------

        # pessoas
        for p in persons:
            x1, y1, x2, y2 = p['bbox_int']
            if p['using']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "Using Phone", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 120, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 120, 0), 2)

        # telefones
        for ph in phones:
            x1, y1, x2, y2 = ph['bbox_int']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.circle(frame, tuple(map(int, ph['center'])), 5, (0, 255, 0), -1)
            cv2.putText(frame, "Phone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        # linhas de associação
        for (p_i, per_i) in associations:
            c1 = tuple(map(int, phones[p_i]["center"]))
            c2 = tuple(map(int, persons[per_i]["center"]))
            cv2.line(frame, c1, c2, (255, 255, 0), 2)

        # contadores
        total_phones = len(phones)
        total_using = sum(1 for p in persons if p["using"])

        cv2.putText(frame, f"Phones: {total_phones}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Using Phone: {total_using}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # fps
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # mostrar frame
        cv2.imshow("Detector de Celulares + Mãos", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
