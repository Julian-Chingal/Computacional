import numpy as np
import random

class AlgoritmoGenetico:
    def __init__(self, imgMatrix, srcStart, srcFinish, tam_population = 20, num_parents = 50, mutation_pro = 0.001, max_generations = 100):
        self.imgMatrix = imgMatrix
        self.srcStart = srcStart
        self.srcFinish = srcFinish
        self.tam_population = tam_population
        self.num_parents = num_parents
        self.mutation_pro = mutation_pro
        self.max_generations = max_generations
        self.last_trajectories = []
    
    #Inciar la poblacion 
    def initialize_population(self):
        population = []
        for _ in range(self.tam_population):
            trajectory = [self.srcStart]
            x, y = self.srcStart
            routesVisited = set([self.srcStart]) #coordenadas visitadas
            
            while (x, y) != self.srcFinish:
                options = []
                # Movimiento hacia la derecha
                if x < self.srcFinish[0] and (x + 1, y) not in routesVisited and self.imgMatrix[x + 1, y] != 1:
                    options.append((x + 1, y))
                # Movimiento hacia arriba
                if y < self.srcFinish[1] and (x, y + 1) not in routesVisited and self.imgMatrix[x, y + 1] != 1:
                    options.append((x, y + 1))
                # Movimiento hacia la izquierda
                if x > self.srcFinish[0] and (x - 1, y) not in routesVisited and self.imgMatrix[x - 1, y] != 1:
                    options.append((x - 1, y))
                # Movimiento hacia abajo
                if y > self.srcFinish[1] and (x, y - 1) not in routesVisited and self.imgMatrix[x, y - 1] != 1:
                    options.append((x, y - 1))
                if options:
                    x, y = random.choice(options)
                    trajectory.append((x, y))
                    routesVisited.add((x, y))
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
    def cross(self, father1, father2):
        if len(father1) <= 2 or len(father2) <= 2:
            if random.random() < 0.5:
                son = father1[:]
            else:
                son = father2[:]
        else:
            crossing_point1 = random.randint(1, len(father1) - 2)   #puntos de cruce 
            crossing_point2 = random.randint(crossing_point1  + 1, len(father2) - 1)

            son = father1[:crossing_point1 ] + father2[crossing_point1 :crossing_point2] + father1[crossing_point2:]
   
        return son
    
    #funcion de mutacion 
    def mutate(self, individuo):
        for i in range(1, len(individuo) - 1):
            if random.random() < self.mutation_pro:
                x, y = individuo[i]
                options = []
                
                #posibles movimientos
                movements = [(0,1),(0,-1),(1,0),(-1,0)]

                for dx, dy in movements:
                    new_x, new_y = x + dx, y + dy
                    if(0<= new_x < len(self.imgMatrix) and 0 <= new_y < len(self.imgMatrix[0]) and self.imgMatrix[new_x, new_y]):
                        options.append((new_x, new_y))
                
                if options:
                    individuo[i] = random.choice(options)

        return individuo
    
    def AG(self):
        #inciar la poblacion
        population = self.initialize_population()
        current_generation = 0
        while current_generation < self.max_generations:
            fitness_population = [self.fitness(guy) for guy in population]
            
            # mejor trayectoria de la generación actual
            best_guy = population[np.argmin(fitness_population)]  #mejor individuo
            
             # Almacenar las últimas 5 trayectorias
            if len(self.last_trajectories) < 5:
                self.last_trajectories.append(best_guy[:])
            else:
                self.last_trajectories.pop(0)
                self.last_trajectories.append(best_guy[:])

            #seleccionar los padres
            parents = self.select_parents(population)
            
            # Creación de la nueva generación
            new_generation = []

            while len(new_generation) < self.tam_population:
                father1, father2 = random.sample(parents, 2)
                son = self.cross(father1, father2)
                mutate_son = self.mutate(son)
                new_generation.append(mutate_son)
            
            # Reemplazo de la población anterior por la nueva generación
            population = new_generation

            current_generation += 1
        
        return best_guy