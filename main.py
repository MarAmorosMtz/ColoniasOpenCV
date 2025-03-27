import cv2
import numpy as np

img = cv2.imread('HCL001R.jpg')
img = cv2.resize(img, ((int(3072/5), int(4080/5))))
cv2.imshow('img', img)

borde = (0, 0, 255)
color_texto = (0, 0, 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(gray, kernel, iterations=3)
dilation = cv2.dilate(erosion, kernel, iterations=3)
gauss = cv2.GaussianBlur(dilation, (3, 3), 2)
canny = cv2.Canny(gauss, 100, 100)

contornos, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contornos, -1, borde, 2)

cv2.imshow('Bordes', img)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()