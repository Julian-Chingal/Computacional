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
weidth_cut = 60
height_cut = 70

# functions Process------------------------------------------------------------------------------
def getPoints(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        srcPoints.append((x, y))
        print("Punto agregado: ", x, y)

def center(mask):  # print point start and finish
    global cutImage
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

def refPoints():  # detected start and finish
    global cutImage, srcStart, srcFinish
    # saturar la img
    hsv_frame = cv.cvtColor(cutImage, cv.COLOR_BGR2HSV)

    # Definir rangos de color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 100, 100])  
    upper_green = np.array([80, 255, 255])  

    # Filtra los píxeles de color rojo y azul
    mask_red = cv.inRange(hsv_frame, lower_red, upper_red)
    mask_green = cv.inRange(hsv_frame, lower_green, upper_green)

    # Comprobar si hay algún píxel rojo
    start_center = center(mask_red)
    if start_center:
        srcStart = start_center
        print("Inicio: ", start_center)
    else:
        srcStart = (0, 0)

    # Comprobar si hay algún píxel azul
    finish_center = center(mask_green)
    if finish_center:
        srcFinish = finish_center
        print("Final: ", finish_center)
    else:
        srcFinish = (40, 40)

def DrawContours(matriz):  # delimits the objects it detects
    global cutImage
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
    global srcPoints, cutImage, gray, blur, height_cut, weidth_cut, canny, binary
    # Points
    srcPoints = np.array(srcPoints)
    dstPoints = np.array([[0, 0], [weidth_cut, 0], [weidth_cut, height_cut], [0, height_cut]],dtype=np.float32,)  # img definir tamaño

    # perspective transform
    homography, _ = cv.findHomography(np.float32(srcPoints), dstPoints)
    m = cv.getPerspectiveTransform(np.float32(srcPoints), dstPoints)
    img_undistorted = cv.undistort(
        frame, np.eye(3), np.zeros(5)
    )  # Corregir distorsión no lineal
    cutImage = cv.warpPerspective(img_undistorted, m, (height_cut, weidth_cut))

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

# video capture
cap = cv.VideoCapture(1)

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
        refPoints()
        #DrawContours(canny)

        # router
        drawCircuit(blur)

        # Show
        cutImage = cv.resize(cutImage, None, fx=7, fy=7, interpolation=cv.INTER_LINEAR)  # esto es para escalar la img recortada
        cv.imshow("Cut", cutImage)
    else:
        for corner in srcPoints:  # mostrar los puntos
            cv.circle(frame, corner, 2, (0, 0, 255), -1)
        cv.imshow("Original", frame)
    # exit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
