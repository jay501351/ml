# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:24:22 2019

@author: Pushpalatha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('J:\\Machine Learning\\Class\\Practical\\Algorithms\\All_Algorithms\\KNN\\shirtsize.csv')
print(df.columns)
df



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['T Shirt Size'] = le.fit_transform(df['T Shirt Size'])
df

X = df.iloc[:, :-1].values
Y = df.iloc[:, 2].values
X
Y

#Split traing and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
                                                    ,random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
#accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
print(knn.score(X_test,Y_test))

#Predict for single
print knn.predict([[158,58]])
print knn.predict([[170,68]])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, prediction)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction)
print("Accuracy:",result2)
