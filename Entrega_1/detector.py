from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv

#Variables
window = None
canvas = None
srcPoints = []
srcCircle = []
cutImage = None

#video capture
cap = cv.VideoCapture(2)

#functions
def interface():
    global window, canvas

    #Config Window
    window = Tk()
    window.geometry("720x500")
    window.title("Captura")

    canvas = Canvas(window, width= "500" ,height= "400")
    canvas.pack()
    
    capture()

    canvas.bind("<Button-1>", getPoints)    
    
    window.mainloop()
    
    

    cap.release()

def showCapture(capt,canvas):
    #Convertir frame a un formato compatible
    capt = cv.cvtColor(capt, cv.COLOR_BGR2RGB)
    capt = Image.fromarray(capt)
    capt = capt.resize((canvas.winfo_width(), canvas.winfo_height())) #ajustar al tamaño del canvas
    captTk = ImageTk.PhotoImage(image = capt)

    #actualizar el contenido del canvas
    canvas.create_image(0,0, anchor = "nw", image = captTk)
    canvas.image = captTk

def capture():
    global canvas, window
    ret,frame = cap.read()
    if ret:   
        showCapture(frame,canvas)
        if len(srcPoints) == 4:

            blur = Preprocess(frame)
            showCapture(blur,canvas)
    window.after(10, capture)

def getPoints(event):
    global canvas,srcPoints, srcCircle

    if len(srcPoints) < 4:
        x = event.x
        y = event.y

        srcPoints.append((x,y))
        print("Punto agregado: ", x, y)
        
        circle_id = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="points")
        srcCircle.append(circle_id)

def Preprocess(frame):
  global srcPoints
  #Points
  srcPoints = np.array(srcPoints)
  dstPoints = np.array([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]], dtype=np.float32)
  #perspective transform
  homography, _ = cv.findHomography(np.float32(srcPoints), dstPoints)

  img_undistorted = cv.undistort(frame, np.eye(3), np.zeros(5)) # Corregir distorsión no lineal

  img = cv.warpPerspective(img_undistorted, homography, (frame.shape[1], frame.shape[0]))

  #grayscale
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  

  #threshold and binary
  _ , binary = cv.threshold(gray,0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  

  #Gaussian blur
  blur = cv.GaussianBlur(binary, (7,7),1) 
  
  return img 

interface()