from contextlib import nullcontext
import numpy as np
import cv2 as cv

#variables 
srcPoints = []

#functions
def circuitPoints (event,x,y,flags, param):
  if event == cv.EVENT_LBUTTONDOWN:
    srcPoints.append((x,y))

#video capture
cap = cv.VideoCapture(1)

#Events
cv.namedWindow('Original')
cv.setMouseCallback('Original', circuitPoints)

#stage capture
while (cap.isOpened()):
  ret, frame = cap.read()
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  if not ret:   #si no retorna imagen se rompe el ciclo
    break

  if len(srcPoints) == 4: # hasta que no seleccione los 4 srcPoints no entra al ciclo
    #perspective transform
    dstPoints = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
    homography, mask = cv.findHomography(np.float32(srcPoints), dstPoints, cv.RANSAC, 5.0)

    # Applies a perspective transformation 
    cutImage = cv.warpPerspective(frame, homography, (frame.shape[1], frame.shape[0]))

    gray = cv.cvtColor(cutImage,cv.COLOR_BGR2GRAY)  #convertir imagen a escala de grises

    #Gaussian blur
    blur = cv.GaussianBlur(gray, (7,7),0) 

    #threshold and binary
    threshold = cv.adaptiveThreshold (blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,5)  

    #Canny detection
    canny = cv.Canny(threshold, 100 ,200) 

    #Show
    cv.imshow('Circuito', frame)
    cv.imshow('Escala de grises', canny)
    cv.imshow('Imagen recortada', cutImage)
  else:
    for corner in srcPoints: #mostrar los puntos
      cv.circle(frame, corner, 5, (0,0,255),-1)
    cv.imshow('Original', frame)

  #exit
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()