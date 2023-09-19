import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from GA import AlgoritmoGenetico

# variables
srcPoints = []
srcStart = (0, 0)
srcFinish = (40, 40)
# preprocess variables
cutImage = None
gray = None
blur = None
canny = None
binary = None
# Image dimension crop
weidth_cut = 50
height_cut = 50

# functions Process------------------------------------------------------------------------------
def getPoints(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        srcPoints.append((x, y))
        print("Punto agregado: ", x, y)

def centerCircle(contorno, etiqueta, color_etiqueta):  # print point start and finish
    M = cv.moments(contorno)
    if M["m00"] != 0:
        centro_x = int(M["m10"] / M["m00"])
        centro_y = int(M["m01"] / M["m00"])
        centro = (centro_x, centro_y)
        cv.circle(
            cutImage, centro, 4, color_etiqueta, -1
        )  # Dibuja un círculo en el centro
        cv.putText(
            cutImage,
            etiqueta,
            (centro[0] - 30, centro[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_etiqueta,
            2,
        )  # Agrega la etiqueta
        return centro
    else:
        return None

def refPoints():  # detected start and finish
    global cutImage, srcStart, srcFinish
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

    # mascara rojo y azul
    redContour, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blueContour, _ = cv.findContours(
        mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    for contorno in redContour:
        srcStart = centerCircle(contorno, "inicio", (0, 0, 255))

    for contorno in blueContour:
        srcFinish = centerCircle(contorno, "final", (255, 0, 0))

def DrawContours():  # delimits the objects it detects
    global cutImage, canny
    # Contours
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
    global srcPoints, cutImage, gray, blur, height_cut, weidth_cut, canny, binary

    # Points
    srcPoints = np.array(srcPoints)
    dstPoints = np.array(
        [[0, 0], [weidth_cut, 0], [weidth_cut, height_cut], [0, height_cut]],
        dtype=np.float32,
    )  # img definir tamaño

    # perspective transform
    homography, _ = cv.findHomography(np.float32(srcPoints), dstPoints)
    img_undistorted = cv.undistort(
        frame, np.eye(3), np.zeros(5)
    )  # Corregir distorsión no lineal
    cutImage = cv.warpPerspective(img_undistorted, homography, (height_cut, weidth_cut))

    #detectar el color del obstaculo. Crear una variable en la cual se almacene esa informacion 
    #no binarizar la imagen, extraer los 3 canales del color.

    # grayscale
    gray = cv.cvtColor(cutImage, cv.COLOR_BGR2GRAY)

    # threshold and binary
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Gaussian blur
    blur = cv.GaussianBlur(binary, (7, 7), 1)

    # Canny detection
    canny = cv.Canny(blur, 100, 150)

def empty(a):
    pass

# PLT ----------------------------------------------------------------------------------------
def drawCircuit(matriz):
    global srcStart, srcFinish
    matriz = np.array(matriz)
    ag = AlgoritmoGenetico(matriz, srcStart , srcFinish)

    ag.AG()

    filas, columnas = matriz.shape
    
    # Crear una figura y un eje
    fig, ax = plt.subplots()
    
    # Iterar a través de la matriz y dibujar los rectángulos
    for fila in range(filas):
        for columna in range(columnas):
            if matriz[fila][columna] == 1:
                ax.add_patch(plt.Rectangle((columna, fila), 1, 1, color='black'))
            else:
                ax.add_patch(plt.Rectangle((columna, fila), 1, 1, edgecolor='gray', facecolor='none'))
    
    # Configurar límites y mostrar el gráfico
    
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.set_aspect('equal')  # Para que los cuadrados tengan el mismo tamaño en x y y
    plt.gca().invert_yaxis()  # Invertir el eje y para que la parte superior sea el principio
    plt.axis('off')  # Ocultar ejes
    plt.show()

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

    if not ret:  # si no retorna img se rompe el ciclo
        break

    if len(srcPoints) == 4:  # hasta que no seleccione los 4
        # Preproces
        Preprocess(frame)

        # contours
        # refPoints()
        DrawContours()

        # router
        drawCircuit(blur)

        # Show
        cutImage = cv.resize(cutImage, None, fx=10, fy=10, interpolation=cv.INTER_LINEAR)  # esto es para escalar la img recortada
        contac = cv.hconcat([gray, blur, canny])
        cv.imshow("Cut", cutImage)
        cv.imshow("Concat", contac)
    else:
        for corner in srcPoints:  # mostrar los puntos
            cv.circle(frame, corner, 2, (0, 0, 255), -1)
        cv.imshow("Original", frame)
    # exit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
