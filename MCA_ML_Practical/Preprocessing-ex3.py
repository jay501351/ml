# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 09:55:44 2019

@author: Pushpalatha
"""

import pandas as pd
data = pd.read_csv('J:\\Machine Learning\\Class\Practical\\Preprocessing\\Book1.csv')

# Slice the result for first 5 rows
print (data[0:5]['Salary'])

# Use the multi-axes indexing method called .loc
print (data.loc[:,['Salary','Name']])

# Use the multi-axes indexing funtion
print (data.loc[[1,3,5],['Salary','Name']])


# Use the multi-axes indexing funtion
print (data.loc[2:6,['Salary','Name']])

print (data.loc[:,['Salary','Name']])


import pandas as pd
dataset = pd.read_csv('J:\\Machine Learning\\Class\Practical\\Preprocessing\\Data.csv')
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
X

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])
X
print(X)


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X[:, 0] = onehotencoder.fit_transform(X[:, 0])
X

dummy = pd.get_dummies(dataset['Country'])
dummy

dataset = pd.concat([dataset,dummy],axis=1)
dataset

dataset.drop(['Country'],axis=1)
dataset

# Split the data between the Training Data and Test Data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
                                                    ,random_state = 0)
X_train
X_test
Y_train
Y_test
dataset