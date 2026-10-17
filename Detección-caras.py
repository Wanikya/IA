import os
import cv2 as cv
import numpy as np

# 1. Definir la ruta base principal
base_path = r'C:\Users\hikaf\OneDrive - Instituto Tecnológico de Morelia\Documentos\Tec\9°\Inteligencia Artificial\Pruebas\Caras'

# 2. Pedir al usuario el número de subcarpeta (clase/persona)
persona_id = input("Ingresa el número de la subcarpeta (ejemplo: 1, 2, 3): ").strip()

# 3. Crear la subcarpeta
folder_path = os.path.join(base_path, persona_id)
os.makedirs(folder_path, exist_ok=True)

# 4. Cargar detector y cámara
rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)

count = 0
MAX_IMAGENES = 1000  # Límite configurado

print(f"\nCapturando datos en: {folder_path}")
print("Presiona ESC para cancelar antes de llegar a 1000.\n")

# Función para guardar imágenes en rutas con caracteres especiales (ó, °, espacios)
def guardar_imagen_unicode(ruta, img):
    exito, buf = cv.imencode('.jpg', img)
    if exito:
        buf.tofile(ruta)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        # Dibujar rectángulo guía
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Recortar y redimensionar el rostro
        img = frame[y:y + h, x:x + w]
        img_resized = cv.resize(img, (100, 100))
        
        count += 1
        
        # Guardar imagen soportando acentos y caracteres especiales
        file_name = f'{persona_id}_{count}.jpg'
        full_file_path = os.path.join(folder_path, file_name)
        guardar_imagen_unicode(full_file_path, img_resized)
        
        cv.imshow('cara', img_resized)

        # Detener la captura al alcanzar las 1000 fotos
        if count >= MAX_IMAGENES:
            break

    # Dibujar contador en pantalla
    cv.putText(frame, f"Capturas: {count}/{MAX_IMAGENES}", (20, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la toma de la cámara
    cv.imshow('rostros', frame)

    # Detener con tecla ESC o si se alcanzó el límite
    k = cv.waitKey(1)
    if k == 27 or count >= MAX_IMAGENES:
        break

cap.release()
cv.destroyAllWindows()

# Mensaje final
if count >= MAX_IMAGENES:
    print(f"¡ÉXITO! Se alcanzó el límite de {MAX_IMAGENES} imágenes en la carpeta '{persona_id}'.")
else:
    print(f"Proceso detenido manualmente. Se guardaron {count} imágenes en la carpeta '{persona_id}'.")