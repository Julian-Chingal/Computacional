import numpy as np
import cv2 as cv

#variables 
srcPoints = []
srcStart = None
srcFinish = None
cutImage = None
wewidth_cut = None
High_cut = None

#functions ------------------------------------------------------------------------------
def getPoints (event,x,y,flags, param):
  if event == cv.EVENT_LBUTTONDOWN:
    srcPoints.append((x,y))
    print("Punto agregado: ", x, y)

def refPoints():
  global cutImage, srcStart, srcFinish

  #saturar la imagen 
  hsv_frame = cv.cvtColor(cutImage, cv.COLOR_BGR2HSV)

  #Definir rangos de color
  lower_red = np.array([0, 100, 100])
  upper_red = np.array([10, 255, 255])

  lower_blue = np.array([100, 100, 100])
  upper_blue = np.array([130, 255, 255])
  
  #Filtra los píxeles de color rojo y azul
  mask_red = cv.inRange(hsv_frame, lower_red, upper_red)
  mask_blue = cv.inRange(hsv_frame, lower_blue, upper_blue)

  # Encuentra círculos en las máscaras de color rojo y azul
  circles_red = cv.HoughCircles(mask_red, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
  circles_blue = cv.HoughCircles(mask_blue, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

  if circles_red is not None:
    # Dibuja los círculos rojos encontrados
    circles_red = np.uint16(np.around(circles_red))
    for circle in circles_red[0, :]:
      srcStart = (circle[0], circle[1])
      cv.circle(cutImage, srcStart, circle[2], (0, 0, 255), 2)

  if circles_blue is not None:
    # Dibuja los círculos azules encontrados
    circles_blue = np.uint16(np.around(circles_blue))
    for circle in circles_blue[0, :]:
      srcFinish = (circle[0], circle[1])
      cv.circle(cutImage, srcFinish, circle[2], (255, 0, 0), 2)

def DrawContours(canny):
    global cutImage
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
        cv.putText(cutImage, 'Alto: ' + str(int(h)), (x + w +20, y + 40), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
  
    return canny

def Preprocess(frame):
  global srcPoints, cutImage
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
  
  return blur 

def empty(a):
  pass
#---------------------------------------------------------------------------------------------------------------------

#video capture
cap = cv.VideoCapture(2)

#Events ----------------------------------------------------------------------------------------------------
cv.namedWindow('Original')
cv.setMouseCallback('Original', getPoints)
cv.namedWindow('Parameters')
cv.resizeWindow('Parameters', 400,150)
cv.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
cv.createTrackbar("Area", "Parameters",5000,40000, empty)

#start  -------------------------------------------------------------------------------------------
while (cap.isOpened()):
  ret, frame = cap.read()
  frame = cv.flip(frame, 1)     #eliminar efecto espejo

  if not ret:   #si no retorna imagen se rompe el ciclo
    break

  if len(srcPoints) == 4: # hasta que no seleccione los 4 
    #Preproces
    blur  = Preprocess(frame)

    #Canny detection
    threshold1  = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2  = cv.getTrackbarPos("Threshold2", "Parameters")
    canny = cv.Canny(blur, threshold1 ,threshold2)
    
    #contours
    refPoints()
    DrawContours(canny)

    #Show
    cv.imshow('Cut', cutImage)
    cv.imshow('canny', canny)
    cv.imshow('Blur', blur)
  else:
    for corner in srcPoints: #mostrar los puntos
      cv.circle(frame, corner, 2, (0,0,255),-1)
    cv.imshow('Original', frame)
  #exit
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()