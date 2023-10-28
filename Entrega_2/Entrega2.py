import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix as cm
from sklearn.preprocessing import LabelEncoder
#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Variables
path = "./Entrega_2/Data/Titanic.csv"
label = LabelEncoder()

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
x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size= 0.3, random_state=42)
logistic_model = LogisticRegression(max_iter=1000)

models = [
    ("Logistic Regression", logistic_model), # Regresion logistica
    ("Desicion Tree", DecisionTreeClassifier()), # arbol de decision 
    ("Random Forest", RandomForestClassifier(n_estimators=10)), # con 10 árboles
    ("KNN", KNeighborsClassifier(n_neighbors=4)), # vecinos cercanos
    ("Support -vector Machine-linear", SVC(kernel="linear")), # Separacion Lineal
    ("Support -vector Machine-rbf", SVC(kernel="rbf")) # k-Means
]

# Realizar validación cruzada k-fold (k=10) en los datos de entrenamiento
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
        
    # Imprimir los resultados de la validación cruzada
    print(f'\n{name} Precisión media en validación cruzada: {cv_results.mean() * 100:.2f}%')

    model.fit(x_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    predict = model.predict(x_test)

    # Calcular la precisión del modelo en el conjunto de prueba
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
    # print(f'{name} confussion matrix: \n', c_matriz)