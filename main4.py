import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('HCL001R.jpg')
imgR = img.copy()
imgY = img.copy()
imgB = img.copy()

# 1. Preprocesamiento - Mejorar contraste en zonas oscuras
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))  # Más agresivo
l = clahe.apply(l)
lab = cv2.merge((l,a,b))
img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Convertir a HSV (mejor para detección de color)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir rangos para tonos rojizos en HSV (corregido)
lower_red1 = np.array([0, 50, 50])       # Rojo bajo (tono 0-10)
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])      # Rojo alto (tono 160-180)
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([30, 255, 255])

lower_blue1 = np.array([200, 30, 30])
upper_blue1 = np.array([200, 255, 255])
lower_blue2 = np.array([85, 40, 30])
upper_blue2 = np.array([115, 120, 80])

# Crear máscaras
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask1, mask2)

mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

maskB1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
maskB2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
mask_blue = cv2.bitwise_or(maskB1, maskB2)

# Mejorar la máscara con morfología
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

kernel = np.ones((5, 5), np.uint8)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=1)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=2)

kernel = np.ones((5, 5), np.uint8)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)

#Suavizado
mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)
mask_red = cv2.GaussianBlur(mask_red, (5, 5), 0)
mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)

# Encontrar contornos
contoursRed, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contoursYellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contoursBlue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar por área y contar colonias
min_area = 20  # Ajustar según el tamaño mínimo de tus colonias
countR = 0

for cnt in contoursRed:
    area = cv2.contourArea(cnt)
    if area > min_area:
        countR += 1
        # Dibujar contorno y número
        cv2.drawContours(imgR, [cnt], -1, (0, 255, 0), 4)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, str(countR), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

countY = 0
for cnt in contoursYellow:
    area = cv2.contourArea(cnt)
    if area > min_area:
        countY += 1
        # Dibujar contorno y número
        cv2.drawContours(imgY, [cnt], -1, (0, 255, 0), 4)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, str(countR), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

countB = 0
for cnt in contoursBlue:
    area = cv2.contourArea(cnt)
    if area > min_area:
        countB += 1
        # Dibujar contorno y número
        cv2.drawContours(imgB, [cnt], -1, (0, 255, 0), 4)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, str(countR), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

# Resultados
print(f"Número total de colonias rojas detectadas: {countR}")
print(f"Número total de zonas amarillas detectadas: {countY}")
print(f"Número total de colonias azules detectadas: {countB}")
cv2.imwrite('coloniasRojas.jpg', imgR)
cv2.imwrite('mascara_rojo.jpg', mask_red)

cv2.imwrite('zonasAmarillas.jpg', imgY)
cv2.imwrite('mascara_amarillo.jpg', mask_yellow)

cv2.imwrite('coloniasAzules.jpg', imgB)
cv2.imwrite('mascara_azul.jpg', mask_blue)

# Mostrar resultados
#cv2.imshow('Colonias Detectadas', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()