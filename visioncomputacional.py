import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

#variables 
corners = []

#functions
def circuitCorners (event,x,y,flags, param):
  if event == cv.EVENT_LBUTTONDOWN:
    corners.append((x,y))

#video capture
cap = cv.VideoCapture(0)

#Events
cv.namedWindow('Original')
cv.setMouseCallback('Original', circuitCorners)

#stage capture
while (cap.isOpened()):
  ret, frame = cap.read()
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  if not ret:   #si no retorna imagen se rompe el ciclo
    break

  if len(corners) == 4: # hasta que no seleccione las 4 corners no entra al ciclo
    #perspective transform
    finalDimension = np.float32([[100,100], [400,100], [400,300] ,[100,300]])
    perspective = cv.getPerspectiveTransform(np.float32(corners), finalDimension)

    # Applies a perspective transformation 
    cutImage = cv.warpPerspective(frame, perspective, (frame.shape[1], frame.shape[0]))

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #convertir imagen a escala de grises
    
    #Mostrar
    cv.imshow('Original', frame)
    cv.imshow('Escala de grises', gray)
    cv.imshow('Imagen recortada', cutImage)
  else:
    for corner in corners: #mostrar los puntos
      cv.circle(frame, corner, 5, (0,0,255),-1)
    cv.imshow('Original', frame)

  #exit
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

#Difuninar el ruido
#blur = cv.GaussianBlur(gray, (7,7),0) 
  
#limite y binarizacion 
#threshold = cv.adaptiveThreshold (blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,5)  
  
#Deteccion de bordes
#canny = cv.Canny(threshold, 100 ,200) 
cap.release()
cv.destroyAllWindows()