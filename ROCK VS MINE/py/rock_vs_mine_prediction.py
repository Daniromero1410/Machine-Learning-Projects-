# -*- coding: utf-8 -*-
"""ROCK VS MINE Prediction.ipynb


Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing"""

#loading the dataset to a pandas Dataframe

sonar_data = pd.read_csv('/content/sonar data.csv',header = None)

sonar_data.head()

#Number of rows and columns
sonar_data.shape

sonar_data.describe() #describe --> statistical mesaures of the data

sonar_data[60].value_counts()

"""M -->  Mine
R --> Rock
"""

sonar_data.groupby(60).mean()

# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

"""Training and Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

"""MODEL TRAINING --> LOGISTIC REGRESSION"""

model = LogisticRegression()

#Training the logistic regression
model.fit(X_train,Y_train)

"""Model Evaluation"""

#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ', training_data_accuracy)

#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ', test_data_accuracy)

"""Making a predictive System"""

input_data = (0.0303,	0.0353,	0.0490,	0.0608,	0.0167,	0.1354,	0.1465,	0.1123,	0.1945,	0.2354,	0.2898,	0.2812,	0.1578,	0.0273,	0.0673,	0.1444,	0.2070,	0.2645,	0.2828,	0.4293,	0.5685,	0.6990,	0.7246,	0.7622,	0.9242,	1.0000,	0.9979,	0.8297,	0.7032,	0.7141,	0.6893,	0.4961,	0.2584,	0.0969,	0.0776,	0.0364,	0.1572,	0.1823,	0.1349,	0.0849,	0.0492,	0.1367,	0.1552,	0.1548,	0.1319,	0.0985,	0.1258,	0.0954,	0.0489,	0.0241,	0.0042,	0.0086,	0.0046,	0.0126,	0.0036,	0.0035,	0.0034,	0.0079,	0.0036,	0.0048)

#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#Reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')
