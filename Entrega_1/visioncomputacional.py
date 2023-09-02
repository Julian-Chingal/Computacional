import numpy as np
import cv2 as cv

#variables 
srcPoints = []

#functions ------------------------------------------------------------------------------
def getPoints (event,x,y,flags, param):
  if event == cv.EVENT_LBUTTONDOWN:
    srcPoints.append((x,y))
    print("Punto agregado: ", x, y)

def DrawContours(canny):
  #Contours
    contours, _  = cv.findContours(canny,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contorno in contours:
      area = cv.contourArea(contorno)
      areamin = cv.getTrackbarPos("Area", "Parameters")
      if area > areamin:
        peri = cv.arcLength(contorno, True)
        approx = cv.approxPolyDP(contorno, 0.02 * peri ,True)
        x, y, w, h = cv.boundingRect(approx)
        cv.rectangle(cutImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(cutImage, 'Ancho: ' + str(int(w)), (x + w +20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

def empty(a):
  pass

#video capture
cap = cv.VideoCapture(0)

#Events ----------------------------------------------------------------------------------------------------
cv.namedWindow('Original')
cv.setMouseCallback('Original', getPoints)
cv.namedWindow('Parameters')
cv.resizeWindow('Parameters', 400,150)
cv.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
cv.createTrackbar("Area", "Parameters",5000,40000, empty)

#stage capture -------------------------------------------------------------------------------------------
while (cap.isOpened()):
  ret, frame = cap.read()
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  if not ret:   #si no retorna imagen se rompe el ciclo
    break

  if len(srcPoints) == 4: # hasta que no seleccione los 4 

    #Points
    srcPoints = np.array(srcPoints)
    dstPoints = np.array([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]], dtype=np.float32)

    #perspective transform
    homography, _ = cv.findHomography(np.float32(srcPoints), dstPoints)

    img_undistorted = cv.undistort(frame, np.eye(3), np.zeros(5)) # Corregir distorsión no lineal

    cutImage = cv.warpPerspective(img_undistorted, homography, (frame.shape[1], frame.shape[0]))

    #grayscale
    gray = cv.cvtColor(cutImage,cv.COLOR_BGR2GRAY)  

    #threshold and binary
    _ , binary = cv.threshold(gray,0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  

    #Gaussian blur
    blur = cv.GaussianBlur(binary, (7,7),1) 

    #Canny detection
    threshold1  = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2  = cv.getTrackbarPos("Threshold2", "Parameters")
    canny = cv.Canny(blur, threshold1 ,threshold2)

    #contours
    DrawContours(canny)

    #Show
    cv.imshow('Original', frame)
    cv.imshow('Cut', cutImage)
    cv.imshow('canny', canny)
  else:
    for corner in srcPoints: #mostrar los puntos
      cv.circle(frame, corner, 2, (0,0,255),-1)
    cv.imshow('Original', frame)

  #exit
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()