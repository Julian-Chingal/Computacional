import time
import numpy as np
import cv2 as cv
# archivos
from GA import AlgoritmoGenetico
from Api import *

# variables
srcPoints = []
srcStart = (0, 0)
srcFinish = (300, 300)

route_update = True
trajectory = []
trajectory_original_size = []
# Image dimension crop
ancho =  93   # 175
alto =  93  # 142
anchopx = 800 # Ancho fijo en pixeles 800
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
    lower_color_1 = np.array([10, 50, 50])
    upper_color_1 = np.array([50, 255, 255])

    # Final rojo
    lower_color_2 = np.array([0, 100, 100])
    upper_color_2 = np.array([10, 255, 255])

    # Filtra los píxeles de color rojo y azul
    mask_red = cv.inRange(hsv_frame, lower_color_1, upper_color_1)
    mask_blue = cv.inRange(hsv_frame, lower_color_2, upper_color_2)

    # Comprobar si hay algún píxel rojo
    start_center = center(mask_blue)
    if start_center:
        srcStart = start_center
        cv.circle(img, srcStart, 0, (0,0,255), thickness=15)
        cv.putText(img, ' Inicio', (srcStart[0], srcStart[1] + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

    else:
        srcStart = (0,0)

    # Comprobar si hay algún píxel azul
    finish_center = center(mask_red)
    if finish_center:
        srcFinish = finish_center
        cv.circle(img, srcFinish, 0, (255,0,0), thickness=15)
        cv.putText(img, ' Meta', (srcFinish[0],srcFinish[1] + 20 ), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0),1)

    else:
        srcFinish = (40, 40)
    
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
    global srcPoints
    
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
     
    # Canny detection
    canny = cv.Canny(thresh, 100, 150)
     
    return cutImage, canny, thresh

def object_color(cutImage): # Detectar objetos de un color determinado
    low_1 = cv.getTrackbarPos("low_1", "Parameters")
    low_2 = cv.getTrackbarPos("low_2", "Parameters")
    low_3 = cv.getTrackbarPos("low_3", "Parameters")

    upp_1 = cv.getTrackbarPos("upp_1", "Parameters")
    upp_2 = cv.getTrackbarPos("upp_2", "Parameters")
    upp_3 = cv.getTrackbarPos("upp_3", "Parameters")
    
    # Rangos color, verde
    lower_color = np.array([low_1, low_2, low_3])
    upper_color = np.array([upp_1, upp_2, upp_3])

    # lower_color = np.array([36, 0, 0]) # Verde oscuro
    # upper_color = np.array([70, 255,255]) # Verde claro

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
def drawCircuit(matrix):
    global route_update, trajectory, trajectory_original_size

    if route_update:
        matrix = matrix / 255.0 # Transformar valores de 255 a 1
        o_x, o_y = matrix.shape

        #! Escalar
        new_x = o_x//10
        new_y = o_y//10

        x1, y1 = srcStart
        x2, y2 = srcFinish

        # Aplicar la misma escala a los puntos de inicio y fin
        re_x1 = int(x1 * (new_x / o_x) )
        re_y1 = int(y1 * (new_y / o_y))
        re_x2 = int(x2 * (new_x / o_x))
        re_y2 = int(y2 * (new_y / o_y))

        srcStart_re = (re_x1,re_y1)
        srcFinish_re = (re_x2,re_y2)

        resize_matrix = cv.resize(matrix,(new_x,new_y), interpolation=cv.INTER_AREA)
        resize_matrix = (resize_matrix != 0).astype(int)

        ag = AlgoritmoGenetico(resize_matrix, srcStart_re, srcFinish_re)
        trajectory = ag.get_resultado()
        
        a = post_route(trajectory)
        print(a)
        
        route_update = False

        trajectory_original_size = [(int(x * (o_x / new_x)), int(y * (o_y / new_y))) for x, y, _, _ in trajectory]

    
#! video capture
cap = cv.VideoCapture(0)

#? Events -----------------------------------------------------------------------------------------
cv.namedWindow("Original")
cv.setMouseCallback("Original", getPoints)
# --
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 400, 300)
cv.createTrackbar("Area", "Parameters", 400, 1000, empty)
cv.createTrackbar("low_1", "Parameters", 52, 255, empty) #36
cv.createTrackbar("low_2", "Parameters", 17, 255, empty) #73
cv.createTrackbar("low_3", "Parameters", 0, 255, empty)
cv.createTrackbar("upp_1", "Parameters", 98, 255, empty) #101
cv.createTrackbar("upp_2", "Parameters", 255, 255, empty)
cv.createTrackbar("upp_3", "Parameters", 255, 255, empty)

#? start  -----------------------------------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv.flip(frame, 1)  # eliminar efecto espejo

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

        if cv.waitKey(1) & 0xFF == ord("u"):
            drawCircuit(thresh, cutImage)

        if route_update == False:
            #Dibujar la trayectoria
            for i in range(len(trajectory_original_size) - 1):
                x1, y1 = trajectory_original_size[i]
                x2, y2 = trajectory_original_size[i+1]

                cv.line(cutImage, (x1, y1), (x2, y2), (255, 255, 255), 2)

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