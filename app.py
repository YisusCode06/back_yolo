from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import math
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes entre frontend y backend

# Cargar el modelo YOLO en la GPU si está disponible
model = YOLO("./best.pt")

# Definir las clases según tu archivo data.yaml
classNames = ['Mal estado', 'Apto para chifles', 'No apto para chifles']

# Ruta para procesar las imágenes recibidas desde el frontend
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Obtener los datos de la imagen desde el frontend
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Reducir la resolución de la imagen para acelerar el procesamiento
        target_size = (640, 480)  # Ajustar el tamaño de la imagen según lo que usabas en la cámara
        img_resized = cv2.resize(img, target_size)

        # Detección con YOLO
        results = model(img_resized, stream=True)  # Ajustar el umbral de confianza si es necesario

        # Lista para almacenar los contornos de las detecciones
        all_contours = []
        banana_counts = {name: 0 for name in classNames}  # Inicializar el contador de plátanos

        # Procesar los resultados de YOLO
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if cls < len(classNames):
                    class_name = classNames[cls]
                    banana_counts[class_name] += 1  # Contar el plátano por su estado
                    color = (0, 0, 255) if class_name == "Mal estado" else (0, 255, 0) if class_name == "Apto para chifles" else (0, 255, 255)
                    cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img_resized, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Si el modelo devuelve máscaras, extraer los contornos
                if hasattr(r, 'masks') and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()  # Convertir las máscaras a numpy

                    for mask in masks:
                        mask_resized = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        # Encontrar los contornos
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Dibujar los contornos en la imagen
                        cv2.drawContours(img_resized, contours, -1, color, 2)

                        # Almacenar los contornos encontrados
                        for contour in contours:
                            # Simplificar los contornos
                            epsilon = 0.01 * cv2.arcLength(contour, True)
                            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                            contour_points = approx_contour[:, 0, :].tolist()
                            all_contours.append(contour_points)

        # Codificar la imagen procesada para devolverla al frontend
        _, buffer = cv2.imencode('.jpg', img_resized)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': img_str, 'contours': all_contours, 'banana_counts': banana_counts})

    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return jsonify({'error': str(e)})

# Nueva ruta para contar los plátanos
@app.route('/count_bananas', methods=['POST'])
def count_bananas():
    try:
        # Obtener los datos de la imagen desde el frontend
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Reducir la resolución de la imagen para acelerar el procesamiento
        target_size = (640, 480)
        img_resized = cv2.resize(img, target_size)

        # Detección con YOLO
        results = model(img_resized, stream=True)

        banana_counts = {name: 0 for name in classNames}  # Inicializar el contador de plátanos

        # Procesar los resultados de YOLO
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])

                if cls < len(classNames):
                    class_name = classNames[cls]
                    banana_counts[class_name] += 1  # Contar el plátano por su estado

        return jsonify({'banana_counts': banana_counts})

    except Exception as e:
        print(f"Error contando plátanos: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
