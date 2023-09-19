import numpy as np
import random

class AlgoritmoGenetico:
    def __init__(self, matrizImg, srcStart, srcFinish, tam_population = 20, num_parents = 50, pro_mutation = 0.001, max_generations = 100):
        self.matrizImg = matrizImg
        self.srcStart = srcStart
        self.srcFinish = srcFinish
        self.tam_population = tam_population
        self.num_parents = num_parents
        self.pro_mutation = pro_mutation
        self.max_generations = max_generations
    
    #Inciar la poblacion 
    def initialize_population(self):
        population = []
        for _ in range(self.tam_population):
            trajectory = [self.srcStart]
            x, y = self.srcStart
            routeVisited = set([self.srcStart]) #coordenadas visitadas
            
            while (x, y) != self.srcFinish:
                options = []

                # Movimiento hacia la derecha
                if x < self.srcFinish[0] and self.matrizImg[x + 1][y] != 1 and (x + 1, y) not in routeVisited:
                    options.append((x + 1, y))
                
                # Movimiento hacia arriba
                if y < self.srcFinish[1] and self.matrizImg[x][y + 1] != 1 and (x, y + 1) not in routeVisited:
                    options.append((x, y + 1))
                
                # Movimiento hacia la izquierda
                if x > self.srcFinish[0] and self.matrizImg[x - 1][y] != 1 and (x - 1, y) not in routeVisited:
                    options.append((x - 1, y))
            
                # Movimiento hacia abajo
                if y > self.srcFinish[1] and self.matrizImg[x][y - 1] != 1 and (x, y - 1) not in routeVisited:
                    options.append((x, y - 1))

                if options:
                    x, y = random.choice(options)
                    trajectory.append((x, y))
                    routeVisited.add((x, y))
                
                else:
                    if len(trajectory) == 1:
                        break
                    trajectory.pop()
                    x, y = trajectory[-1]
            
            population.append(trajectory)

        return population
    
    #funcion de fitness
    def fitness(self, individuo):
        longitud = 0
        for i in range(len(individuo) - 1):
            x1, y1 = individuo[i]
            x2, y2 = individuo[i + 1]
        
            distancia_x = abs(x1 - x2)
            distancia_y = abs(y1 - y2)
        
            longitud += distancia_x + distancia_y
        return longitud
    
    #seleccionar los padres
    def select_parents(self, population):
        parents = []
        for _ in range(self.num_parents):
            participants = random.sample(population, 2)
            fitness_competitors = [self.fitness(guy) for guy in participants]
            best_guy = participants[np.argmin(fitness_competitors)]
            parents.append(best_guy)

        return parents
    
    #funcion que permite cruzar en un punto para dos individuos 
    def cross(self, padre1, padre2):
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