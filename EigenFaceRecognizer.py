import cv2 as cv
import numpy as np
import os

# Función para cargar imágenes en rutas con caracteres especiales (ó, °, espacios)
def cargar_imagen_unicode(ruta):
    buf = np.fromfile(ruta, dtype=np.uint8)
    return cv.imdecode(buf, cv.IMREAD_GRAYSCALE)

data_set = r'C:\Users\hikaf\OneDrive - Instituto Tecnológico de Morelia\Documentos\Tec\9°\Inteligencia Artificial\Pruebas\Caras'
faces = sorted(
    d for d in os.listdir(data_set)
    if os.path.isdir(os.path.join(data_set, d))
)
print("Personas/Clases encontradas:", faces)

labels = []
faces_data = []
label = 0

for person in faces:
    person_path = os.path.join(data_set, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cargar_imagen_unicode(img_path)
        if img is None:
            continue
        labels.append(label)
        faces_data.append(img)
    print(f"  Clase '{person}' (label={label}): {labels.count(label)} imágenes")
    label += 1

face_recognizer = cv.face.EigenFaceRecognizer_create()
face_recognizer.train(faces_data, np.array(labels))
face_recognizer.write("Eigenface.xml")
print("Modelo escrito: Eigenface.xml")