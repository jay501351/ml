# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:34:21 2019

@author: Pushpalatha
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


play_tennis = pd.read_csv("J:\Machine Learning\Class\Practical\Algorithms\All_Algorithms\PlayTennis.csv")
play_tennis.head()

number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])
play_tennis

#define the features and the target variables
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

features_train, features_test, target_train, target_test = train_test_split(play_tennis[features],
play_tennis[target],
test_size = 0.33,
random_state = 54)


model = GaussianNB()
model.fit(features_train, target_train) 
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)

print model.predict([[1,2,0,1]])
print model.predict([[2,0,0,0]])
print model.predict([[1,0,0,0]])