import numpy as np
import random

class AlgoritmoGenetico:
    def __init__(self, imageBin, srcStart, srcFinish,num_restrictions, tam_population = 50, pro_mutation = 0.001, max_generations = 100):
        self.imageBin = imageBin
        self.srcStart = srcStart
        self.srcFinish = srcFinish
        self.num_restrictions = num_restrictions
        self.tam_population = tam_population
        self.pro_mutation = pro_mutation
        self.max_generations = max_generations
        self.max_size = 2
        self.min_size = 1
    
    def restriction(self):
        cell_size = self.max_size + 1  # Tamaño de la celda basado en el tamaño máximo de restricción
        cells_x = (self.imageBin.shape[0] - 1) // cell_size + 1
        cells_y = (self.imageBin.shape[1] - 1) // cell_size + 1

        for _ in range(self.num_restrictions):
            size = random.randint(self.min_size, self.max_size)
            cell_x = random.randint(0, cells_x - 1)
            cell_y = random.randint(0, cells_y - 1)

            x = cell_x * cell_size
            y = cell_y * cell_size

            if (
                (x + size <= self.srcStart[0] or x >= self.srcStart[0] + 1 or y + size <= self.srcStart[1] or y >= self.srcStart[1] + 1)
                and (x + size <= self.srcFinish[0] or x >= self.srcFinish[0] + 1 or y + size <= self.srcFinish[1] or y >= self.srcFinish[1] + 1)
            ):
                self.imageBin[x:x+size, y:y+size] = 1

        return self.imageBin
    
    def initialize_population(self):
        poblacion = []
        for _ in range(self.tam_population):
            individuo = self.generar_individuo()
            poblacion.append(individuo)
        return poblacion
    
    def generar_individuo(self):
        individuo = []
        for _ in range(len(self.imageBin)):
            fila = []
            for _ in range(len(self.imageBin[0])):
                if random.random() < 0.5:
                    fila.append(1)  # Movimiento permitido
                else:
                    fila.append(0)  # Movimiento bloqueado
            individuo.append(fila)
        return individuo
    
    def fitness(self, individuo):
        x, y = self.srcStart
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
        return self.distancia_euclidiana((x, y), self.srcFinish)
    

    
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
                if random.random() < self.pro_mutation:
                    individuo[i][j] = 1 - individuo[i][j]  # Cambiar el valor del bit (0 a 1 o 1 a 0)
        return individuo
    
    def ejecutar(self):
        poblacion = self.initialize_population()
        
        while self.generacion_actual < self.max_generations:
            nueva_poblacion = []
            
            for individuo in poblacion:
                fitness = self.evaluar_individuo(individuo)
                individuo_actualizado = {'individuo': individuo, 'fitness': fitness}
                
                if self.mejor_individuo is None or fitness < self.mejor_individuo['fitness']:
                    self.mejor_individuo = individuo_actualizado
                
                nueva_poblacion.append(individuo_actualizado)
            
            poblacion = []
            
            while len(poblacion) < self.tam_population:
                padre1, padre2 = self.seleccionar_padres(nueva_poblacion)
                hijo = self.cruzar_padres(padre1['individuo'], padre2['individuo'])
                hijo_mutado = self.mutar_individuo(hijo)
                poblacion.append({'individuo': hijo_mutado, 'fitness': None})
            
            self.generacion_actual += 1
        
        return self.mejor_individuo['individuo']