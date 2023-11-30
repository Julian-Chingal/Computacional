import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# archivos
from GAV3 import AlgoritmoGenetico
# from Api import *

# variables
srcPoints = []
srcStart = (0, 0)
srcFinish = (400, 400)

route_update = True
trajectory = None
# Image dimension crop
ancho = 93  # 93
alto = 93   # 93
anchopx = 700 # Ancho fijo en pixeles 800
pxporcm = round(anchopx/ancho)
altopx = pxporcm*alto

#? functions Process------------------------------------------------------------------------------
def getPoints(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        srcPoints.append((x, y))
        print("Punto agregado: ", x, y)

def center(mask):  # print point start and finish
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:  
      largest_contour = max(contours, key=cv.contourArea)
      # Calcula el centroide del contorno
      M = cv.moments(largest_contour)

      if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    
    return None

def refPoints(img):  # detected start and finish
    global srcStart, srcFinish
    # saturar la img
    hsv_frame = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # limite color Amarillo para detectar carro
    lower_color_1 = np.array([15, 100, 20]) 
    upper_color_1 = np.array([45, 255, 255]) 

    # Final rojo
    lower_color_2 = np.array([0, 100, 100])
    upper_color_2 = np.array([10, 255, 255])

    # Filtra los píxeles de color rojo y azul
    mask_red = cv.inRange(hsv_frame, lower_color_1, upper_color_1)
    mask_blue = cv.inRange(hsv_frame, lower_color_2, upper_color_2)

    # Comprobar si hay algún píxel rojo
    start_center = center(mask_red)
    if start_center:
        srcStart = start_center
        cv.circle(img, srcStart, 0, (0,0,255), thickness=15)
        cv.putText(img, ' Inicio', (srcStart[0], srcStart[1] + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

    else:
        srcStart = (0,0)

    # Comprobar si hay algún píxel azul
    finish_center = center(mask_blue)
    if finish_center:
        srcFinish = finish_center
        cv.circle(img, srcFinish, 0, (255,0,0), thickness=15)
        cv.putText(img, ' Meta', (srcFinish[0],srcFinish[1] + 20 ), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0),1)

    else:
        srcFinish = (40, 40)
    
    # print(f'Incio: {start_center} | Final: {finish_center}')

def DrawContours(matriz, cutImage):  # delimits the objects it detects
    # Contours
    contours, _ = cv.findContours(matriz, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contorno in contours:
        area = cv.contourArea(contorno)
        areamin = cv.getTrackbarPos("Area", "Parameters")
        if area > areamin:
            peri = cv.arcLength(contorno, True)
            approx = cv.approxPolyDP(contorno, 0.02 * peri, True)
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(cutImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv.putText(cutImage, 'Ancho: ' + str(int(w)), (x + w +10, y + 10), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0),1)
            # cv.putText(cutImage, 'Alto: ' + str(int(h)), (x + w +10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0),1)

def Preprocess(frame, width, height):
    global srcPoints, aa
    
    # Points
    srcPoints = np.float32(srcPoints)
    dstPoints = np.array([[0, 0], [width, 0], [width, height],[0, height]],dtype=np.float32,)  # img definir tamaño
    
    # perspective transform
    homography = cv.getPerspectiveTransform(srcPoints, dstPoints)
    cutImage= cv.warpPerspective(frame, homography, (width,height))

    # Detect object color
    detColor = object_color(cutImage)

    # Process
    _, thresh = cv.threshold(detColor, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
     
    # Invertir la máscara
    # thresh = cv.bitwise_not(thresh)

    # Canny detection
    canny = cv.Canny(thresh, 100, 150)
     
    
    return cutImage, canny, thresh

def object_color(cutImage): # Detectar objetos de un color determinado
    # Rangos color, rojo
    # lower_color = np.array([0, 100, 100])
    # upper_color = np.array([10, 255, 255])

    # Rangos color, verde
    lower_color = np.array([71, 92, 0])
    upper_color = np.array([101, 255, 255])  

    # Convertir a espacio de color HSV
    hsv = cv.cvtColor(cutImage, cv.COLOR_BGR2HSV)

    # Crea una máscara usando el rango de colores definido
    mask = cv.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((3, 3), np.uint8) # reduce el ruido 5x5
    mask_filtered = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return mask_filtered

def empty(a):
    pass

#? Graph and generate -----------------------------------------------------------------------------------------------------
def drawCircuit(matrix, cutImage):
    global srcStart, srcFinish, route_update, trajectory

    if route_update:
        matrix = matrix / 255.0 # Transformar valores de 255 a 1
        ag = AlgoritmoGenetico(matrix, srcStart, srcFinish)
        trajectory = ag.get_resultado(pxporcm)
        print(trajectory)
        
        # a = post_route(trajectory)
        # print(a)

        route_update = False

    # Dibujar la trayectoria
    for i in range(len(trajectory) - 1):
        x1, y1, x2, y2, g, l = trajectory[i]
        cv.line(cutImage, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
#! video capture
cap = cv.VideoCapture(1)

#? Events -----------------------------------------------------------------------------------------
cv.namedWindow("Original")
cv.setMouseCallback("Original", getPoints)
# --
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 400, 50)
cv.createTrackbar("Area", "Parameters", 400, 1000, empty)

#? start  -----------------------------------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)  # eliminar efecto espejo

    # Control FPS
    time.sleep(0.035)

    if not ret:  # si no retorna img se rompe el ciclo
        break

    if len(srcPoints) == 4:  # hasta que no seleccione los 4
        # Preproces
        cutImage,  canny, thresh = Preprocess(frame,anchopx,altopx)

        # Ref
        refPoints(cutImage)

        # contours
        DrawContours(canny, cutImage)

        # router
        drawCircuit(thresh, cutImage)
       

        # Show
        cv.imshow("Cut", cutImage)
        cv.imshow("Thres", thresh)
        cv.imshow("canny", canny)

    else:
        for corner in srcPoints:  # mostrar los puntos
            cv.circle(frame, corner, 2, (0, 0, 255), -1)
        cv.imshow("Original", frame)
    # exit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()