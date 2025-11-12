# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:34:22 2019

@author: Pushpalatha
"""
#https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('J:\\Machine Learning\\Class\Practical\\Algorithms\\NaiveBayes\\Heart.csv')
print(df.columns)
df
df.head()

#To write in csv file
df.columns
header = ['Age','Sex','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','	Ca','Thal','AHD']
df.to_csv('J:\\Machine Learning\\Class\Practical\\Algorithms\\NaiveBayes\\data_modified.csv', columns=header)

    
#Label encoding location and salary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ChestPain1'] = le.fit_transform(df['ChestPain'])
df.head()
df['Thal1'] = le.fit_transform(df['Thal'])
df.head()
df['Result'] = le.fit_transform(df['AHD'])
df.head()
df.drop(['Result'],axis=1)
df.drop(['ChestPain'],axis=1)
df.drop(['Thal'],axis=1)
df.to_csv('J:\\Machine Learning\\Class\Practical\\Algorithms\\NaiveBayes\\Heartdata1.csv')
df.columns

header = ['Age','Sex','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca','ChestPain1','Thal1','Result']
df.to_csv('J:\\Machine Learning\\Class\Practical\\Algorithms\\NaiveBayes\\FinalHeartdata.csv', columns=header)

df = pd.read_csv('J:\\Machine Learning\\Class\Practical\\Algorithms\\NaiveBayes\\FinalHeartdata.csv')
print(df.columns)
df
df.isnull().sum()
df.Result.value_counts()


sns.countplot(x="Result", data=df, palette="bwr")
plt.show()

countNoDisease = len(df[df.Result == 0])
countHaveDisease = len(df[df.Result == 1])
total = (len(df.Result))
print(total)
print(countNoDisease)
print(countHaveDisease)
int(164)/int(303)*100
per = (float(countNoDisease) / float(total))*100
print(per)
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease // (len(df.Result))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease // (len(df.Result))*100)))


sns.countplot(x='Sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

countFemale = len(df[df.Sex == 0])
countMale = len(df[df.Sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.Sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.Sex))*100)))

df.groupby('AHD').mean()


pd.crosstab(df.Age,df.Resul).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

df.head()
df.columns

X = df.iloc[:, :-1].values
Y = df.iloc[:, 14].values
X
Y

#Split traing and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
                                                    ,random_state = 0)
X_train
X_test
Y_train
Y_test


accuracies = {}
#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
acc = nb.score(X_test,Y_test)*100
print(acc)
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, Y_train) 
# making predictions on the testing set 
Y_pred = gnb.predict(X_test) 
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred)*100)


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))

#SVM
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, Y_train)
acc = svm.score(X_test,Y_test)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
acc = dtc.score(X_test, Y_test)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(X_train, Y_train)
acc = rf.score(X_test,Y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

#Comparison of all algorithms
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()