import warnings
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


#EDA

df = pd.read_csv('fertility_Diagnosis.txt', header=None)

# Variabili target e features:
v = df.iloc[:, -1].values
y, c = pd.factorize(v, sort=True)
X = df.iloc[:, :-1].values

# Suddivisione in training e testing set:
# Usiamo stratify=y per bilanciare classi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


#Scalamento dei dati

# Scalo i dati perchè la maggior parte dei metodi che ho scelto necessitano lo scalamento:
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
n_features = X_train_scaled.shape[1]  # mi serve per decidere la griglia di ricerca del numero di hidden unit dell'MLP

print('\n Numero di training sample =', X_train_scaled.shape[0])
print('\n Numero di feature =', n_features)