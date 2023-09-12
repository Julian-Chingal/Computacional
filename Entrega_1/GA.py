import numpy as np
import random

class AlgoritmoGenetico:
    def __init__(self, imagen, punto_inicial, punto_final, tamano_poblacion = 50, probabilidad_mutacion = 0.001, max_generaciones = 100):
        self.imagen = imagen
        self.punto_inicial = punto_inicial
        self.punto_final = punto_final
        self.tamano_poblacion = tamano_poblacion
        self.probabilidad_mutacion = probabilidad_mutacion
        self.max_generaciones = max_generaciones
        self.mejor_individuo = None
        self.generacion_actual = 0
    
    def inicializar_poblacion(self):
        poblacion = []
        for _ in range(self.tamano_poblacion):
            individuo = self.generar_individuo()
            poblacion.append(individuo)
        return poblacion
    
    def generar_individuo(self):
        individuo = []
        for _ in range(len(self.imagen)):
            fila = []
            for _ in range(len(self.imagen[0])):
                if random.random() < 0.5:
                    fila.append(1)  # Movimiento permitido
                else:
                    fila.append(0)  # Movimiento bloqueado
            individuo.append(fila)
        return individuo
    
    def evaluar_individuo(self, individuo):
        x, y = self.punto_inicial
        for i in range(len(individuo)):
            fila = individuo[i]
            for j in range(len(fila)):
                movimiento = fila[j]
                dx, dy = self.obtener_direccion(movimiento)
                x += dx
                y += dy
                if not self.es_movimiento_valido(x, y):
                    break
            if not self.es_movimiento_valido(x, y):
                break
        return self.distancia_euclidiana((x, y), self.punto_final)
    
    def distancia_euclidiana(self, punto1, punto2):
        x1, y1 = punto1
        x2, y2 = punto2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def es_movimiento_valido(self, x, y):
        filas, columnas = self.imagen.shape
        return 0 <= x < filas and 0 <= y < columnas and self.imagen[x][y] == 1
    
    def obtener_direccion(self, movimiento):
        if movimiento == 0:
            return -1, 0  # Movimiento hacia arriba
        elif movimiento == 1:
            return 1, 0  # Movimiento hacia abajo
        elif movimiento == 2:
            return 0, -1  # Movimiento hacia la izquierda
        else:
            return 0, 1  # Movimiento hacia la derecha
    
    def seleccionar_padres(self, poblacion):
        padres = random.choices(poblacion, k=2, weights=[1 / (individuo['fitness'] + 1) for individuo in poblacion])
        return padres
    
    def cruzar_padres(self, padre1, padre2):
        hijo = []
        for i in range(len(padre1)):
            fila_hijo = []
            for j in range(len(padre1[0])):
                if random.random() < 0.5:
                    fila_hijo.append(padre1[i][j])
                else:
                    fila_hijo.append(padre2[i][j])
            hijo.append(fila_hijo)
        return hijo
    
    def mutar_individuo(self, individuo):
        for i in range(len(individuo)):
            for j in range(len(individuo[0])):
                if random.random() < self.probabilidad_mutacion:
                    individuo[i][j] = 1 - individuo[i][j]  # Cambiar el valor del bit (0 a 1 o 1 a 0)
        return individuo
    
    def ejecutar(self):
        poblacion = self.inicializar_poblacion()
        
        while self.generacion_actual < self.max_generaciones:
            nueva_poblacion = []
            
            for individuo in poblacion:
                fitness = self.evaluar_individuo(individuo)
                individuo_actualizado = {'individuo': individuo, 'fitness': fitness}
                
                if self.mejor_individuo is None or fitness < self.mejor_individuo['fitness']:
                    self.mejor_individuo = individuo_actualizado
                
                nueva_poblacion.append(individuo_actualizado)
            
            poblacion = []
            
            while len(poblacion) < self.tamano_poblacion:
                padre1, padre2 = self.seleccionar_padres(nueva_poblacion)
                hijo = self.cruzar_padres(padre1['individuo'], padre2['individuo'])
                hijo_mutado = self.mutar_individuo(hijo)
                poblacion.append({'individuo': hijo_mutado, 'fitness': None})
            
            self.generacion_actual += 1
        
        return self.mejor_individuo['individuo']