import cv2
import numpy as np
import os

carpeta = "fuente"
output_dir = "resultados"
extensiones = (".jpg")

for imagen in os.listdir(carpeta):
    # Cargar la imagen
    ruta_imagen = os.path.join(carpeta, imagen)
    img = cv2.imread(ruta_imagen)

    # 1. Preprocesamiento Mejorado
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Usar CLAHE para mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Filtrado adaptativo para diferentes tamaños de colonias
    blur_small = cv2.medianBlur(gray, 13)
    blur_large = cv2.GaussianBlur(gray, (25, 25), 0)

    # 2. Umbralización Adaptativa Mejorada
    # Para colonias pequeñas
    thresh_small = cv2.adaptiveThreshold(blur_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 5)

    # Para colonias grandes (usar parámetros diferentes)
    thresh_large = cv2.adaptiveThreshold(blur_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 101, 10)

    # Combinar ambos umbrales
    thresh_combined = cv2.bitwise_or(thresh_small, thresh_large)

    # 3. Operaciones Morfológicas Mejoradas
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    # Limpieza para colonias pequeñas
    opening = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # Preservar colonias grandes
    dilated = cv2.dilate(closing, kernel_large, iterations=1)

    # 4. Detección de Contornos Mejorada
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filtrado Inteligente de Contornos
    min_area = 10
    max_area = 30000  # Aumentar considerablemente para colonias grandes
    count = 0

    # Ordenar contornos por área (de mayor a menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Ajustar umbrales dinámicamente para las colonias más grandes
        if i < 5:  # Las 5 colonias más grandes
            current_min_area = 1000  # Umbral más alto para las grandes
        else:
            current_min_area = min_area

        if current_min_area < area < max_area:
            # Calcular circularidad con ajuste para formas irregulares
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                # Umbral de circularidad más flexible para colonias grandes
                if area > 1000 and circularity > 0.2 or area <= 1000 and circularity > 0.4:
                    count += 1
                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 7)

                    # Dibujar el número de la colonia
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(img, str(count), (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

    nombre = imagen.title() + "_resultado"
    # Mostrar resultados
    print(f"Número total de colonias detectadas en {imagen.title()}: {count}")

    # Guardar resultados en la carpeta "resultados"
    output_filename = os.path.join(output_dir, f"{os.path.splitext(imagen)[0]}_resultado.jpg")
    cv2.imwrite(output_filename, img)
