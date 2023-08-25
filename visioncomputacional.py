import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

#variables 
esquinas = []

#funciones
def esquinasCircuito (event,x,y,flags, param):
  if event == cv.EVENT_LBUTTONDOWN:
    esquinas.append((x,y))
    cv.imshow('Original', frame)

#captura de video
captura = cv.VideoCapture(0)

#Eventos
cv.namedWindow('Original')
cv.setMouseCallback('Original', esquinasCircuito)

while (captura.isOpened()):
  ret, frame = captura.read()
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  if not ret:   #si no retorna imagen se rompe el ciclo
    break

  if len(esquinas) == 4: # hasta que no seleccione las 4 esquinas no entra al ciclo
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #convertir imagen a escala de grises
    #Mostrar
    cv.imshow('Original', frame)
    cv.imshow('Escala de grises', gray)
  else:
    for esquina in esquinas: #mostrar los puntos
      cv.circle(frame, esquina, 5, (0,0,255),-1)
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
captura.release()
cv.destroyAllWindows()