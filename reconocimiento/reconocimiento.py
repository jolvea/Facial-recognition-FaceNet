import os
from os import listdir
from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
#from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
import pickle
import cv2

myfile = open("data10.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

# Inicializar FaceNet y Haar Cascade
MyFaceNet = FaceNet()
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la base de datos
import pickle
with open("data10.pkl", "rb") as myfile:
    database = pickle.load(myfile)

# Definir el umbral de reconocimiento
UMBRAL = 0.8

# Función para procesar cuadros y reconocer rostros
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = HaarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x1, y1, w, h) in wajah:
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        # Extraer y preprocesar el rostro
        face = frame[y1:y2, x1:x2]
        face = Img.fromarray(face).resize((160, 160))
        face = expand_dims(asarray(face), axis=0)
        # Obtener embedding
        signature = MyFaceNet.embeddings(face)
        # Buscar identidad en la base de datos
        min_dist = 100
        identity = "Desconocido"
        for folder_name, folder_data in database.items():
            for image_name, embedding in folder_data.items():
                dist = np.linalg.norm(embedding - signature)
                if dist < min_dist:
                    min_dist = dist
                    identity = folder_name
        # Verificar umbral
        if min_dist > UMBRAL:
            identity = "Desconocido"
        # Dibujar cuadro delimitador y etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame
# Captura en tiempo real con la cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el video. Cerrando...")
        break
    # Procesar el cuadro y mostrarlo
    frame = process_frame(frame)
    cv2.imshow('Reconocimiento Facial en Vivo', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar recursos
cap.release()
cv2.destroyAllWindows()