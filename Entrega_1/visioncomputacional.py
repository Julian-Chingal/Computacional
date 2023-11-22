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
srcFinish = (40, 40)
# Image dimension crop
weidth_cut = 100
height_cut = 100
# object color
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

def refPoints(cutImage):  # detected start and finish
    global srcStart, srcFinish
    # saturar la img
    hsv_frame = cv.cvtColor(cutImage, cv.COLOR_BGR2HSV)

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
    
    print(f'Incio: {start_center} | Final: {finish_center}')

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
            cv.rectangle(cutImage, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv.putText(cutImage, 'Ancho: ' + str(int(w)), (x + w +10, y + 10), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0),1)
            # cv.putText(cutImage, 'Alto: ' + str(int(h)), (x + w +10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0),1)

def Preprocess(frame):
    global srcPoints, height_cut, weidth_cut
    
    # Points
    srcPoints = np.float32(srcPoints)
    dstPoints = np.array([[0, 0], [weidth_cut, 0], [weidth_cut, height_cut], [0, height_cut]],dtype=np.float32,)  # img definir tamaño

    # perspective transform
    homography = cv.getPerspectiveTransform(srcPoints, dstPoints)
    img_undistorted = cv.undistort(frame, np.eye(3), np.zeros(5))

    # Corregir distorsión no lineal
    cutImage= cv.warpPerspective(img_undistorted, homography, (weidth_cut,height_cut))

    #separacion matriz de colores, 
    B=cutImage[:,:,0]
    G=cutImage[:,:,1]
    R=cutImage[:,:,2]

    #detectar pixel por pixel si encuentra el color en este caso limite para detectar color negro
    frameCircuit = []
    auxX = 0
    for i in cutImage:
        auxY = 0
        nrow = []
        for j in i:
            if (B[auxX][auxY] < limBlueMax) and (G[auxX][auxY] < limGreenMax) and (R[auxX][auxY] < limRedMax):
              nrow.append(1)
            else:
              nrow.append(0)
            auxY += 1
        frameCircuit.append(nrow)
        auxX +=1

    # Crear la matriz binaria en función de la máscara
    checkup = [] #guarda la matriz con los colores verificados
    for i in range(len(frameCircuit)):
        auxrow = [] #guarda temporalemente la informacion de cada fila
        for j in range(len(frameCircuit[0])):
            if frameCircuit[i][j] == 0:
                auxrow.append(255)
            else:
                auxrow.append(0)
        checkup.append(auxrow)

    MatrizResult = np.array((checkup), dtype= np.uint8)

    # threshold and binary
    _, binary = cv.threshold(MatrizResult, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Gaussian blur
    blur = cv.GaussianBlur(binary, (7, 7), 1)

    # Canny detection
    canny = cv.Canny(blur, 100, 150)

    return cutImage, blur, canny, binary

def empty(a):
    pass

# PLT ----------------------------------------------------------------------------------------
def drawCircuit(matriz):
    global srcStart, srcFinish
    matriz = np.array(matriz)
    ag = AlgoritmoGenetico(matriz, srcStart , srcFinish)

    trajectory = ag.AG()

    rows, columns = matriz.shape
    
    # Crear una figura y un eje
    fig, ax = plt.subplots()
    
    # Iterar a través de la matriz y dibujar los rectángulos
    for row in range(rows):
        for column in range(columns):
          if matriz[row][column] == 1:
            ax.add_patch(plt.Rectangle((column, row), 1, 1, color='black'))
          else:
            ax.add_patch(plt.Rectangle((column,row), 1, 1, edgecolor='gray', facecolor='none'))

    # Dibujar las trayectorias
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colores para las trayectorias
    for i, trajectory in enumerate(ag.last_trajectories):
        color = colors[i % len(colors)]
        for j in range(len(trajectory) - 1):
            x1, y1 = trajectory[j]
            x2, y2 = trajectory[j + 1]
            ax.plot([y1 + 0.5, y2 + 0.5], [x1 + 0.5, x2 + 0.5], color=color, linewidth=2)
    
    # Configurar límites y mostrar el gráfico
    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')  # Para que los cuadrados tengan el mismo tamaño en x y y
    plt.gca().invert_yaxis()  # Invertir el eje y para que la parte superior sea el principio
    plt.axis('off')  # Ocultar ejes
    plt.show()

# PLT ----------------------------------------------------------------------------------------
def drawCircuit1(matriz):
    matriz = np.array(matriz)
    ag = AlgoritmoGenetico(matriz, srcStart, srcFinish)

    trajectory = ag.AG()

    # Crear una imagen vacía
    img = np.zeros((matriz.shape[0], matriz.shape[1], 3), dtype=np.uint8)

    # Iterar a través de la matriz y dibujar los rectángulos
    for row in range(matriz.shape[0]):
        for column in range(matriz.shape[1]):
            if matriz[row][column] == 1:
                cv.rectangle(img, (column, row), (column + 1, row + 1), (0, 0, 0), 1)
            else:
                cv.rectangle(img, (column, row), (column + 1, row + 1), (255, 255, 255), 1)

    # Dibujar la trayectoria
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Crear una nueva ventana
    cv.namedWindow("Circuit", cv.WINDOW_NORMAL)

    # Ajustar el tamaño de la ventana
    cv.resizeWindow("Circuit", 800, 600)

    # Dibujar las líneas verticales
    for i in range(0, img.shape[1], 20):
        cv.line(img, (i, 0), (i, img.shape[0]), (255, 255, 255), 1)

    # Dibujar las líneas horizontales
    for i in range(0, img.shape[0], 20):
        cv.line(img, (0, i), (img.shape[1], i), (255, 255, 255), 1)

    # Mostrar la imagen en la nueva ventana
    cv.imshow("Circuit", img)

    return img

# video capture
cap = cv.VideoCapture(0)

# Events -----------------------------------------------------------------------------------------
cv.namedWindow("Original")
cv.setMouseCallback("Original", getPoints)
# --
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 400, 50)
cv.createTrackbar("Area", "Parameters", 50, 200, empty)

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
        cutImage, blur, canny, binary = Preprocess(frame)

        # contours
        refPoints(cutImage)
        DrawContours(canny, cutImage)

        # router
        # drawCircuit1(blur)

        # Show
        cutImage = cv.resize(cutImage, None, fx=7, fy=7, interpolation=cv.INTER_LINEAR)  # esto es para escalar la img recortada
        blur = cv.resize(blur, None, fx=7, fy=7, interpolation=cv.INTER_LINEAR)  # esto es para escalar la img recortada
        cv.imshow("Cut", cutImage)
        cv.imshow("blur", blur)
    else:
        for corner in srcPoints:  # mostrar los puntos
            cv.circle(frame, corner, 2, (0, 0, 255), -1)
        cv.imshow("Original", frame)
    # exit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()