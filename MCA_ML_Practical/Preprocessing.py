# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:25:11 2019

@author: Pushpalatha
"""


import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('E:\\Latha\\LathaSKPIMCS\\Machine Learning\\Class\Practical\\Preprocessing\\Data1.csv')
print(dataset.columns)
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
X
Y

#Count missing values
dataset.isnull().sum()

#Remove missing value rows
ds_new = dataset.dropna()
ds_new
ds_new.isnull().sum()


from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

# Encode Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X
Y

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Split the data between the Training Data and Test Data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
X_train
X_test

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train
X_test


# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, Y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
print(y_pred)
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
  
from sklearn.model_selection import train_test_split 
print("Gaussian Naive Bayes model accuracy(in %):",metrics.accuracy_score(Y_test, y_pred)*100)

