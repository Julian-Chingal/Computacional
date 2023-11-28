import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# archivos
from GA import AlgoritmoGenetico
from Api import *

# variables
srcPoints = []
srcStart = (0, 0)
srcFinish = (700, 700)
srcCar = (0,0)

route_update = True
# Image dimension crop
ancho = 93  # 93
alto = 93   # 93
anchopx = 800 # Ancho fijo en pixeles 800
pxporcm = round(anchopx/ancho)
altopx = pxporcm*alto

#limites para caluclar el color del objeto
limRedMax = 30
limBlueMax = 30
limGreenMax = 30

# functions Process------------------------------------------------------------------------------
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

def refCar(img): # Referencia centro del carro 
    global srcCar

    hsv_frame = cv.cvtColor(img, cv.COLOR_BGR2HLV)

    # Colores para detectar carro
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([40, 255, 255])

    mask_color = cv.inRange(hsv_frame, lower_color, upper_color)

    ref_car = center(mask_color)

    if ref_car:
        srcCar = ref_car
    else:
        srcCar = (0,0)

def refPoints(img):  # detected start and finish
    global srcStart, srcFinish
    # saturar la img
    hsv_frame = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Definir rangos de color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255]) 

    # Filtra los píxeles de color rojo y azul
    mask_red = cv.inRange(hsv_frame, lower_red, upper_red)
    mask_blue = cv.inRange(hsv_frame, lower_blue, upper_blue)

    # Comprobar si hay algún píxel rojo
    start_center = center(mask_red)
    if start_center:
        srcStart = start_center
    else:
        srcStart = (0,0)

    # Comprobar si hay algún píxel azul
    finish_center = center(mask_blue)
    if finish_center:
        srcFinish = finish_center
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
    thresh = cv.bitwise_not(detColor)

    # Canny detection
    canny = cv.Canny(detColor, 100, 150)
     
    
    return cutImage, canny, thresh

def object_color(cutImage):
    # Rangos color, rojo
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])
    
    # Convertir a espacio de color HSV
    hsv = cv.cvtColor(cutImage, cv.COLOR_BGR2HSV)

    # Crea una máscara usando el rango de colores definido
    mask = cv.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask_filtered = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return mask_filtered

def empty(a):
    pass

# PLT ----------------------------------------------------------------------------------------
def drawCircuit(matriz):
    global srcStart, srcFinish
    matriz = np.array(matriz)
    ag = AlgoritmoGenetico(matriz, srcStart , srcFinish)

    trajectory = ag.AG()

    print(trajectory)

    # Dibujar las trayectorias
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colores para las trayectorias
    for i, trajectory in enumerate(ag.last_trajectories):
        color = colors[i % len(colors)]
        for j in range(len(trajectory) - 1):
            x1, y1 = trajectory[j]
            x2, y2 = trajectory[j + 1]

# PLT ----------------------------------------------------------------------------------------
def drawCircuit1(matrix, cutImage):
    global srcStart, srcFinish

    matrix = np.array(matrix)
    ag = AlgoritmoGenetico(matrix, srcStart, srcFinish)

    trajectory = ag.AG()

    # Iterar a través de la matriz y dibujar los rectángulos
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if matrix[row][column] == 1:
                cv.rectangle(cutImage, (column, row), (column + 1, row + 1), (0, 0, 0), 1)
            else:
                cv.rectangle(cutImage, (column, row), (column + 1, row + 1), (255, 255, 255), 1)

    # Dibujar la trayectoria
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        cv.line(cutImage, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv.imshow("Cut", cutImage)
    
# video capture
cap = cv.VideoCapture(1)

# Events -----------------------------------------------------------------------------------------
cv.namedWindow("Original")
cv.setMouseCallback("Original", getPoints)
# --
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 400, 50)
cv.createTrackbar("Area", "Parameters", 400, 1000, empty)

# start  -----------------------------------------------------------------------------------------
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

        # contours
        refPoints(cutImage)
        DrawContours(canny, cutImage)

        # router
        if route_update:
            drawCircuit1(thresh, cutImage)
            route_update = False

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