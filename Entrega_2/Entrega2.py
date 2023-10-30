import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix as cm
from sklearn.preprocessing import LabelEncoder
#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC


# Variables
path = "./Entrega_2/Data/Titanic.csv"
label = LabelEncoder()
warnings.filterwarnings(action='ignore', category=FutureWarning)


if os.path.exists(os.path.join(path)):
    print(f"------ Archivo encontrado ------")
else:
    print("Archivo no encontrado") 
    sys.exit(1) 
    
# Prerpocess ----------------------------------------------------------------------------
data = pd.read_csv(path)
features =  ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

# discretize
data['Age'] = pd.qcut(data['Age'], 4, labels=['Niño', 'Joven', 'Adulto', 'Mayor'])
data['Fare'] = pd.qcut(data['Fare'], 5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

# Transformar valores
data['Sex'] = label.fit_transform(data['Sex']) # 0 = male, 1 = female
data['Embarked'] = label.fit_transform(data['Embarked']) # S = 2, Q = 1,C = 0
data['Age'] = label.fit_transform(data['Age']) 
data['Fare'] = label.fit_transform(data['Fare'])
data.dropna(subset=features + [target], inplace=True) # Eliminar valores vacios

# Training ------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size= 0.3, random_state=0)

#info
print("------------------------------------------\n",
      f'Tamaño del dataset: {len(data)}\n',
      f'Cantidad de datos Entrenamiento: {len(y_train)}\n',
      f'Cantidad de datos Prueba: {len(y_test)}\n',
      f'Cantidad de datos Clase Not Survived (0): {(data[target] == 0).sum()}\n',
      f'Cantidad de datos Clase Survived (1): {(data[target] == 1).sum()}\n',
      "------------------------------------------")

# Funcion k 
n_knn = data[target].nunique()

#manual models
models = [
    ("Desicion Tree", DecisionTreeClassifier(criterion='entropy')), # arbol de decision  
    ("Random Forest", RandomForestClassifier(n_estimators=100)), # con 100 árboles class_weight= "balanced"
    ("KNN", KNeighborsClassifier(n_neighbors=n_knn)), # vecinos cercanos
    ("Support -vector Machine-linear", SVC(kernel="linear")), # Separacion Lineal
    ("Support -vector Machine-rbf", SVC(kernel="rbf")) # 
]

#Settings
graph_model = ["Support -vector Machine-linear", "Support -vector Machine-rbf"]

#models
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")

    # Imprimir los resultados de la validación cruzada
    print(f'\n{name} Precisión media en validación cruzada: {cv_results.mean() * 100:.2f}%')

    # Entrenar el modelo
    model.fit(x_train, y_train)

    # Realizar predicciones  y Calcular la precisión del modelo en el conjunto de prueba
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print(f'{name} Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')

    # Calcular la matriz de confusión
    c_matriz = cm(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(c_matriz, annot=True,cmap="Greens", fmt="d")
    plt.xlabel("Predicción")
    plt.ylabel("Valor verdadero")
    plt.title(f'Matriz de confusion {name}')
    plt.show()

    # Graph SVC
    if(name == graph_model[0] or name == graph_model[1]):
        # Figura muestra solo los datos
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(data['Sex'], data['Pclass'],c= data[target])
        ax.set_title("Datos ESL.mixture")
        ax.set_xlabel('Sex')
        ax.set_ylabel('Pclass')
        plt.show()

        #***************************************************************
        #preguntar que se debe hacer si los datos estan en esos valores
        #***************************************************************

        # Grafico division lineal
        
# Implementación de K -Means--------------------------------------------------------------------
# Elbow Method
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x_train)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o') # Un modelo bueno es aquel con una inercia baja y un bajo número de grupos (K)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

k = int(input("\n----- Ingrese el numero de clusters para el modelo k-Means = "))

# Train model
kmeans = KMeans(n_clusters=k)
kmeans.fit(x_train) 

# Graph K-Means
labels = kmeans.labels_
centroids = kmeans.cluster_centers_ # Coordenadas de los centroides de los clústeres

# Dibujar los puntos de datos y los centroides
plt.scatter(x_train.values[:, 0], x_train.values[:, 1], c=labels, cmap='viridis', label='Datos')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroides')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('KMeans Clustering')
plt.show()