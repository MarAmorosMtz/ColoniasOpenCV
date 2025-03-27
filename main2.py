import cv2
import numpy as np

# Cargar y redimensionar la imagen
img = cv2.imread('HCL001R.jpg')
#img = cv2.resize(img, (int(3072 / 4), int(4080 / 4)))
original = img.copy()

# Convertir a escala de grises y aplicar filtrado
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 13)  # Filtro mediano para preservar bordes

# Aplicar umbral adaptativo para manejar variaciones de iluminación
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 5)

# Operaciones morfológicas para limpiar la imagen
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Agregar después del cierre morfológico
sure_bg = cv2.dilate(closing, kernel, iterations=3)
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Encontrar contornos
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
min_area = 10  # Área mínima para colonias muy pequeñas
max_area = 1000  # Área máxima para colonias grandes
count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Solo filtrar por área inicialmente
    if min_area < area < max_area:
        # Calcular circularidad pero no filtrar estrictamente
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
        else:
            circularity = 0

        # Aceptar colonias aunque tengan baja circularidad
        count += 1
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 7)

        # Dibujar el número de la colonia
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, str(count), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

# Mostrar resultados
print(f"Número total de colonias detectadas: {count}")
#cv2.imshow('Umbral', thresh)
#cv2.imshow('Limpieza morfológica', closing)
#cv2.imshow('Resultado', img)

# Al final del código, antes de cv2.destroyAllWindows(), añade:

# Guardar la imagen resultante
output_filename = 'resultado.jpg'
cv2.imwrite(output_filename, img)
print(f"Imagen guardada como: {output_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()
