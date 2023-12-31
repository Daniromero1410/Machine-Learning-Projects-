

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

"""Importamos el dataset"""

dataset_california  = fetch_california_housing()

print(dataset_california)

#Cargar el dataset a un dataframe de pandas

california_dataframe= pd.DataFrame(dataset_california.data,columns = dataset_california.feature_names)

#Imprimir las primaras 5 filas del dataframe

california_dataframe.head()

#Agregar el target columns (precio/price) del data frame
california_dataframe['Price']= dataset_california.target

california_dataframe.head()

#Observar el numero de filas y columnas en el dataframe

california_dataframe.shape

#Check for missing values

california_dataframe.isnull().sum()

#Medidas estadisticas del dataset

california_dataframe.describe()

"""Entendiendo la correlacion entre varios factores en el dataset

1. Correlacion Positiva
2. Correlacion Negativa
"""

correlation = california_dataframe.corr()

#Construyendo un Heatmap para entender la correlacion
plt.figure(figsize =(10,10))
sns.heatmap(correlation, cbar=True, square =True, fmt='.1f',annot=True, annot_kws={'size':8},cmap='Blues')

"""Muestra de datos y targets"""

X= california_dataframe.drop(['Price'],axis=1)
Y= california_dataframe['Price']

print(X)
print(Y)

"""Splitting the data into training data and test data

"""

X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

"""Entrenamiento del modelo

XGBOOST Regressor
"""

#Cargar el modelo
modelo = XGBRegressor()

#Entrenamiento del modelo con X_train
modelo.fit(X_train,Y_train)

"""Evaluacion

Prediccion del training data
"""

#Accuracy for prediction on training data
training_data_prediction = modelo.predict(X_train)

print(training_data_prediction)

#R squared error

score_1 = metrics.r2_score(Y_train, training_data_prediction)


#Mean Absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1 )
print("Mean Absolute error: ", score_2)

"""Visualizacion de los precios actuales y los precios predecidos"""

plt.scatter(Y_train, training_data_prediction)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Price vs Predicted Price')
plt.show()

"""Prediccion del data test"""

#Accuracy for prediction on test data
test_data_prediction = modelo.predict(X_test)

#R squared error

score_1 = metrics.r2_score(Y_test, test_data_prediction)


#Mean Absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1 )
print("Mean Absolute error: ", score_2)
