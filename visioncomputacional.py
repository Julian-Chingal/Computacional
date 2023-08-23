import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

#captura de video
captura = cv.VideoCapture(0)

while (captura.isOpened()):
  ret, frame = captura.read()
 
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  #convertir imagen a escala de grises
  gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

  #Filtrado de ruido
  blur = cv.GaussianBlur(gray, (7,7),0) 
  
  #threshold
  threshold = cv.adaptiveThreshold (blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,5)  
  
  #Deteccion de bordes
  edge = cv.Canny(threshold, 50 ,100) 

  if ret == True:
    # bordes
    cv.imshow('Bordes', edge)

    #desenfoqur
    cv.imshow('Desenfoque', blur)

    #original
    cv.imshow('Origial', frame)

    # si se preciona la tecla e se detiene
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  else: break

captura.release()
cv.destroyAllWindows()