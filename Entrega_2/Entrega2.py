import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix as cm
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from imblearn.under_sampling import RandomUnderSampler
#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree


# Variables
path = "./Entrega_2/Data/Titanic.csv"
label = LabelEncoder()
scaler = StandardScaler()
underSample = RandomUnderSampler(sampling_strategy='majority')
warnings.filterwarnings(action='ignore', category=FutureWarning)


if os.path.exists(os.path.join(path)):
    print(f"------ Archivo encontrado ------")
else:
    print("Archivo no encontrado") 
    sys.exit(1) 
    
# Prerpocess ----------------------------------------------------------------------------
data = pd.read_csv(path)
features =  ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp'] # Sex
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
# data = data.sample(n= 300, random_state=42) # reducir datase a 300 instancias
print(data.dtypes)

# Balanced Data
x_under, y_under = underSample.fit_resample(data[features], data[target])

# Training ------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size= 0.3, random_state=0)

# escalar la data 
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns= x_train.columns)
x_test  = pd.DataFrame(scaler.transform(x_test), columns= x_test.columns)

# Grafica Data
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c = y_train)
ax.set_title("Datos ESL.mixture")
plt.show()

#info
print("------------------------------------------\n",
      f'Tamaño del dataset: {len(data)}\n',
      f'Cantidad de datos Entrenamiento: {len(y_train)}\n',
      f'Cantidad de datos Prueba: {len(y_test)}\n',
      f'Cantidad de datos Clase Not Survived (0): {(y_under == 0).sum()}\n',
      f'Cantidad de datos Clase Survived (1): {(y_under == 1).sum()}\n',
      "------------------------------------------")

# Funcion Numero de vecinos cercanos knn (k) -----------------------------------------------------
k_range = 20
scores = np.zeros((k_range-1))
for k in range(1,k_range):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    scores[k-1] = knn.score(x_test, y_test)
plt.plot(range(1,k_range), scores, marker='o')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title("Accuracy K")
plt.show()

print(f'la mejor precicion {scores.max()} con k =  {scores.argmax()}')
n_knn = scores.argmax()

# Modesl ----------------------------------------------------------------------------------
models = [
    ("Desicion Tree", DecisionTreeClassifier(criterion='entropy')), # arbol de decision  
    ("Random Forest", RandomForestClassifier(n_estimators=100)), # con 100 árboles class_weight= "balanced"
    ("KNN" , KNeighborsClassifier(n_neighbors= n_knn)),
    ("svc-linear", SVC(kernel="linear")), # Separacion Lineal
    ("svc-radial", SVC(kernel="rbf")),
    ("svc-polinomial", SVC(kernel="poly"))
]

#models -------------------------------------------------------------------------------------------
for name, model in models:
    # Imprimir los resultados de la validación cruzada
    print(f'\n---------------------- \nModelo {name}\n----------------------')
    kfold = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")

    print(f'{name} Precisión media en validación cruzada: {cv_results.mean() * 100:.2f}%')

    # Entrenar el modelo
    model.fit(x_train, y_train)

    # Realizar predicciones  y Calcular la precisión del modelo en el conjunto de prueba
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print(f'{name} Precisión en el conjunto de prueba: {accuracy * 100:.2f}%\n')

    # Calcular la matriz de confusión
    c_matriz = cm(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(c_matriz, annot=True,cmap="Greens", fmt="d")
    plt.xlabel("Predicción")
    plt.ylabel("Valor verdadero")
    plt.title(f'Matriz de confusion {name}')
    plt.show()

    #tree
    if(name == "Desicion Tree"):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(model)
        plt.show()

    # Obtener los features mas importantes
    if name == "Random Forest": 
        # Model
        clf = model
        clf.fit(x_train, y_train)

        feature_importances = clf.feature_importances_ # importancia de las características
        top_feature_indices = feature_importances.argsort()[-2:][::-1]
        top_features = x_train.columns[top_feature_indices] # Name
        print(top_feature_indices)
    
        # Print top features
        for feature, importance in zip(top_features, feature_importances[top_feature_indices]):
            print(f"Característica: {feature}, Importancia: {importance*100:.2f}%")

    # Graph SVC
    if "svc-" in name:  # Grafico division lineal

        # Ajustar el modelo SVM para obtener los coeficientes e intercepto
        modelo_svm = model
        modelo_svm.fit(x_train[[top_features[0], top_features[1]]], y_train)

        # Crear una figura para la visualización
        plt.figure(figsize=(6, 4))

        # Graficamos los datos en el espacio de características con colores de clase
        cmap = matplotlib.colors.ListedColormap(['k', 'g'])
        plt.scatter(x_train[top_features[0]], x_train[top_features[1]], c=y_train, s=40, cmap=cmap)

        # Creamos un mesh para evaluar la función de decisión
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
            
        # Modificar xy para que sea una matriz 2D con dos columnas
        xy = np.vstack([XX.ravel(), YY.ravel()]).T  
        Z = modelo_svm.predict(xy).reshape(XX.shape)
            
        # Graficamos el hiperplano y el margen
        ax.contourf(XX, YY, Z, alpha=0.8)

        plt.show()
    
   
# ----------------------------------------------------------------------------------------------
# Implementación de K -Means--------------------------------------------------------------------
#------------------------------------------------------------------------------------------

# Elbow Method
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x_train)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o') # Un modelo bueno es aquel con una inercia alta y un bajo número de grupos (K)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

k = int(input("\n----- Ingrese el numero de clusters para el modelo k-Means = "))

# Determinar las variables con mas peso del dataset

kmeans = KMeans(n_clusters=k) # Train model
kmeans.fit(x_train) 

# Graph K-Means
labels = kmeans.labels_
centroids = kmeans.cluster_centers_ # Coordenadas de los centroides de los clústeres

# Dibujar los puntos de datos y los centroides
plt.scatter(x_train.iloc[:, top_feature_indices[0]], x_train.iloc[:, top_feature_indices[1]], c=y_train, cmap='viridis', label='Datos')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroides')
plt.xlabel(f'Característica {top_features[0]}')
plt.ylabel(f'Característica {top_features[1]}')
plt.title('KMeans Clustering')
plt.show()

# valancear los datos y mostrar luego los datos corregidos y los no corregidos