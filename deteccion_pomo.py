import cv2
import numpy as np

POMO_ID = 67    #position in coco.names -1 position

# Cargar la red YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Cargar las clases
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Caputura de video (puedes cambiar esto por la carga de una imagen)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Crear un blob desde la imagen
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pasar el blob a través de la red
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Información de detección
    class_ids = []
    confidences = []
    boxes = []

    # Analizar las salidas de la red
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == POMO_ID and confidence > 0.5:  # Umbral de confianza
                # Dimensiones del objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                print("pomo coords: ", center_x, center_y)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas del rectángulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supresión no máxima para eliminar duplicados
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar los resultados en la imagen
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Color verde
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar la imagen resultante
    cv2.imshow("Door Detection", frame)

    # Romper el bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
