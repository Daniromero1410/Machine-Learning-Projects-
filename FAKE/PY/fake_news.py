


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

#Imprimimos stopwords en ingles ya que es el lenguaje que vamos a utilizar
print(stopwords.words('english'))

"""Data pre-processing"""

# loading the dataset to a pandas DataFrame
news_dataset= pd.read_csv('/content/train.csv')

news_dataset.shape

#Imprimimos las primeras 5 lineas del dataframe

news_dataset.head()

# Contamos el numero de valores faltantes en el dataset
news_dataset.isnull().sum()

#Remplazamos los valores nulos con strings vacios
news_dataset= news_dataset.fillna('')

#Mergin the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

#Separando los datos del label
X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

print(X)
print(Y)

"""Stemming:

Stemming el proceso de reducir una palabra a su palabra ruta.


por ejemplo
actor, actress,acting --> act
"""

port_stem =  PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content']= news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#Separamos los datos del label nuevamente
X= news_dataset['content'].values
Y= news_dataset['label'].values

print(X)
print(Y)

Y.shape

#Converting the textual data to numerical Data
vectorizer= TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)

print(X)

"""Splitting the dataset to training and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2,stratify=Y,random_state=2)

"""Training the model: Logistic Regression"""

modelo = LogisticRegression()

modelo.fit(X_train, Y_train)

"""Evaluation"""

#Accuracy score on the training data
X_train_prediction= modelo.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data: ",training_data_accuracy)

#Accuracy score on the training data
X_test_prediction= modelo.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of the test data: ",test_data_accuracy)

"""Making a predictive System"""

X_new = X_test[4]

prediction = modelo.predict(X_new)
print(prediction)

if(prediction[0]==0):
  print("The news is real")
else:
  print("The news is fake")

print(Y_test[4])
