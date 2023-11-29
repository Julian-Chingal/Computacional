import random
import numpy as np
import tkinter as tk 
from PIL import ImageGrab,Image
import math

poblacion = []
ultimas_trayectorias = []
resultado = []

img = Image.open('imgAG.jpg').convert('RGB')
img.save('flattened.png')
imagen = Image.open('flattened.png')

print("Imagen Cargada...")

#capturador = CapturadorPuntos('flattened.png')
# capturador = CapturadorPuntos('selection.png')
# puntos = capturador.obtener_puntos()
# punto_inicial = puntos[0]
# punto_final = puntos[1]

# print("Inicio: ",punto_inicial," Fin: ",punto_final)

ancho_original, alto_original = imagen.size

filas, columnas = round(alto_original/10), round(ancho_original/10)
#filas, columnas = round(ancho_original/10), round(alto_original/10)
#filas, columnas = 100,100

ancho_celda = ancho_original // columnas
alto_celda = alto_original // filas

matriz = np.zeros((filas, columnas), dtype=np.uint8)

def get_resultado(pxcms):
    for j in range(len(mejor_trayectoria_suavizada) - 1):
        x1, y1 = mejor_trayectoria_suavizada[j]
        x2, y2 = mejor_trayectoria_suavizada[j + 1]
        xr1,yr1,xr2,yr2 = y1*10, x1*10, y2*10, x2*10
        grados = calcular_grados(xr1, yr1, xr2, yr2)
        longitudPx = calcular_segmento(xr1, yr1, xr2, yr2)
        #print("AG: ",x1,",",y1,",",x2,",",y2)
        print("AG-X10: ",xr1,",",yr1,",",xr2,",",yr2,", G: ",grados,", L: ",(longitudPx/pxcms))
        resultado.append((round(xr1),round(yr1),round(xr2),round(yr2),(longitudPx/pxcms),round(grados))) # Multiplicar longitud por el factor de conversion (Esta en pixeles)
    return resultado
    '''
    if len(resultado) > 0:
        return resultado
    else:
        return 0
    '''
def actualizar_matriz():
    for fila in range(filas):
        for columna in range(columnas):
            color = imagen.getpixel((columna * ancho_celda, fila * alto_celda))
            #print(columna," ",color)
            if isinstance(color, (tuple, list)):
                valor_pixel = 1 if sum(color) < 384 else 0
                matriz[fila][columna] = valor_pixel
                #print(fila,",",columna,", V: ",valor_pixel)
            else:
                matriz[fila][columna] = 0
            
actualizar_matriz()

#inicio = (1,1)
#final = (70, 80)
inicio = (round(punto_inicial[1] / 10), round(punto_inicial[0] / 10))
final = (round(punto_final[1] / 10), round(punto_final[0] / 10))

def calcular_fitness(trayectoria):
    longitud = sum(
        abs(trayectoria[i][0] - trayectoria[i + 1][0]) + abs(trayectoria[i][1] - trayectoria[i + 1][1])
        for i in range(len(trayectoria) - 1)
    )
    return longitud

def calcular_grados(x1, y1, x2, y2):
    ang_radianes = math.atan2(y2 - y1, x2 - x1)
    ang_grados = math.degrees(ang_radianes)
    return ang_grados

def calcular_segmento(x1, y1, x2, y2):
    distancia = math.sqrt((x2 - x1)*2 + (y2 - y1)*2)
    return distancia

def inicializar_poblacion(num_individuos, inicio, final):
    poblacion = []
    for _ in range(num_individuos):
        trayectoria = [inicio]
        x, y = inicio
        coordenadas_visitadas = set([(x, y)])
        while (x, y) != final:
            opciones = []
            if x < final[0] and (x + 1, y) not in coordenadas_visitadas and matriz[x + 1, y] != 1:
                opciones.append((x + 1, y))
            if y < final[1] and (x, y + 1) not in coordenadas_visitadas and matriz[x, y + 1] != 1:
                opciones.append((x, y + 1))
            if x > final[0] and (x - 1, y) not in coordenadas_visitadas and matriz[x - 1, y] != 1:
                opciones.append((x - 1, y))
            if y > final[1] and (x, y - 1) not in coordenadas_visitadas and matriz[x, y - 1] != 1:
                opciones.append((x, y - 1))
            if opciones:
                x, y = random.choice(opciones)
                trayectoria.append((x, y))
                coordenadas_visitadas.add((x, y))
            else:
                if len(trayectoria) == 1:
                    break
                trayectoria.pop()
                x, y = trayectoria[-1]
        poblacion.append(trayectoria)
    return poblacion

def seleccionar_padres(poblacion, fitness_poblacion, num_padres):
    padres = []
    for _ in range(num_padres):
        competidores = random.sample(poblacion, 2)
        competidores_fitness = [calcular_fitness(individuo) for individuo in competidores]
        mejor_individuo = competidores[np.argmin(competidores_fitness)]
        padres.append(mejor_individuo)
    return padres

def cruzar(padre1, padre2):
    if len(padre1) <= 1 or len(padre2) <= 1:
        if len(padre1) > len(padre2):
            hijo = padre1[:]
        else:
            hijo = padre2[:]
    else:
        punto_cruce = random.randint(1, min(len(padre1), len(padre2)) - 1)
        hijo = padre1[:punto_cruce] + padre2[punto_cruce:]
    return hijo

def mutar(individuo, probabilidad_mutacion):
    for i in range(1, len(individuo) - 1):
        if random.random() < probabilidad_mutacion:
            x, y = individuo[i]
            opciones = []
            if x > 0 and matriz[x - 1, y] != 1:
                opciones.append((x - 1, y))
            if y > 0 and matriz[x, y - 1] != 1:
                opciones.append((x, y - 1))
            if x < final[1] and matriz[x + 1, y] != 1:
                opciones.append((x + 1, y))
            if y < final[0] and matriz[x, y + 1] != 1:
                opciones.append((x, y + 1))
            if opciones:
                individuo[i] = random.choice(opciones)
    return individuo

def algoritmo_genetico(num_generaciones, tam_poblacion, num_padres, inicio, final, probabilidad_mutacion):
    poblacion = inicializar_poblacion(tam_poblacion, inicio, final)

    for generacion in range(num_generaciones):
        fitness_poblacion = [calcular_fitness(individuo) for individuo in poblacion]

        mejor_individuo = poblacion[np.argmin(fitness_poblacion)]
        #mejor_fitness = min(fitness_poblacion)

        if len(ultimas_trayectorias) < 5:
            ultimas_trayectorias.append(mejor_individuo[:])
        else:
            ultimas_trayectorias.pop(0)
            ultimas_trayectorias.append(mejor_individuo[:])

        padres = seleccionar_padres(poblacion, fitness_poblacion, num_padres)

        nueva_generacion = []

        while len(nueva_generacion) < tam_poblacion:
            padre1, padre2 = random.sample(padres, 2)
            hijo = cruzar(padre1, padre2)
            hijo_mutado = mutar(hijo, probabilidad_mutacion)
            nueva_generacion.append(hijo_mutado)

        poblacion = nueva_generacion

    return mejor_individuo


num_generaciones = 5
tam_poblacion = 10
num_padres = 2
probabilidad_mutacion = 0

mejor_trayectoria = algoritmo_genetico(num_generaciones, tam_poblacion, num_padres, inicio, final, probabilidad_mutacion)

def suavizar_trayectoria(trayectoria):
    if len(trayectoria) < 7:
        return trayectoria

    puntos_suavizados = [trayectoria[0]]

    for i in range(1, len(trayectoria) - 1):
        p0 = trayectoria[i - 1]
        p1 = trayectoria[i]
        p2 = trayectoria[i + 1]

        control1 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        control2 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        puntos_suavizados.append(control1)
        puntos_suavizados.append(p1)

    puntos_suavizados.append(trayectoria[-1])

    return puntos_suavizados

def aplicar_filtro_media_movil(trayectoria, ventana):
    trayectoria_suavizada = []
    for i in range(len(trayectoria)):
        x_sum = 0
        y_sum = 0
        count = 0
        for j in range(-ventana, ventana + 1):
            if 0 <= i + j < len(trayectoria):
                x, y = trayectoria[i + j]
                x_sum += x
                y_sum += y
                count += 1
        if count > 0:
            x_promedio = x_sum / count
            y_promedio = y_sum / count
            trayectoria_suavizada.append((x_promedio, y_promedio))
    return trayectoria_suavizada

def cerrar_ventana(event):
    ventana.destroy()
    
mejor_trayectoria_suavizada = suavizar_trayectoria(mejor_trayectoria)

ventana = tk.Tk()
ventana.title("Matriz")

canvas = tk.Canvas(ventana, width=(filas*10), height=(columnas*10), bg="white")
canvas.pack()

r = 7
for fila in range(len(matriz)):
    for columna in range(len(matriz[0])):
        if matriz[fila][columna] == 1:
            canvas.create_rectangle(columna*r, fila*r, columna*r+r, fila*r+r, fill="black", outline="black")
        else:
            canvas.create_rectangle(columna*r, fila*r, columna*r+r, fila*r+r, outline="gray")

canvas.create_rectangle(inicio[1]*r, inicio[0]*r, inicio[1]*r+r, inicio[0]*r+r, fill="red")
canvas.create_rectangle(final[1]*r, final[0]*r, final[1]*r+r, final[0]*r+r, fill="green")

num_iteraciones_suavizado = 3
for _ in range(num_iteraciones_suavizado):
    mejor_trayectoria_suavizada = aplicar_filtro_media_movil(mejor_trayectoria_suavizada, ventana=5)

valorX = 3.5
xr1,yr1,xr2,yr2 = 0,0,0,0
grados,longitudPx = 0,0

for j in range(len(mejor_trayectoria_suavizada) - 1):
    x1, y1 = mejor_trayectoria_suavizada[j]
    x2, y2 = mejor_trayectoria_suavizada[j + 1]
    #xr1,yr1,xr2,yr2 = y1*10, x1*10, y2*10, x2*10
    #grados = calcular_grados(xr1, yr1, xr2, yr2)
    #longitudPx = calcular_segmento(xr1, yr1, xr2, yr2)
    #print("AG: ",x1,",",y1,",",x2,",",y2)
    #print("AG-X10: ",xr1,",",yr1,",",xr2,",",yr2,", G: ",grados,", L: ",longitudPx)
    #resultado.append((round(xr1),round(yr1),round(xr2),round(yr2),longitudPx,round(grados))) # Multiplicar longitud por el factor de conversion (Esta en pixeles)
    canvas.create_line(y1 * r + valorX, x1 * r + valorX, y2 * r + valorX, x2 * r + valorX, fill="green", width=2)

canvas.configure(width=len(matriz[0])*r, height=len(matriz)*r)
#ventana.focus()
ventana.bind("<Key>", cerrar_ventana)

ventana.mainloop()