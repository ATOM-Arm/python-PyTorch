import cv2
from ultralytics import YOLO

def main():
    # Carrega o modelo YOLO (pré-treinado no COCO)
    model = YOLO("yolov8n.pt")

    # Lista de classes COCO do modelo
    class_names = model.names

    # Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objetos no frame
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                # Filtrar: mostrar apenas CELULAR
                if class_name == "cell phone":

                    # coordenadas da caixa
                    x1, y1, x2, y2 = box.xyxy[0]

                    # converter para inteiros
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # desenhar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # escrever "cell phone"
                    cv2.putText(frame, "Cell Phone", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar o vídeo
        cv2.imshow("Detecção de Celular - YOLO + PyTorch", frame)

        # Pressionar ESC para sair
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
