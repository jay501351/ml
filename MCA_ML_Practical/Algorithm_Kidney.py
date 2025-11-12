# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:06:38 2019

@author: Pushpalatha
"""
'''https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy'''

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


kdata = pd.read_csv("I:\G-Drive\PhD\Post_PhD\Kidney\kidney_disease.csv")
kdata.head()
kdata.shape
kdata.info()
from scipy import stats
from scipy.stats import shapiro
p = shapiro(kdata['rbc'])
print(p)


kdata.hist(bins=50, figsize=(20, 15))
plt.show()

# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(kdata)
train_set_scaled = scaler.transform(kdata)


#Flaot to integer
kdata.age.astype(int)
kdata['age'].fillna(kdata['age'].median(), inplace=True)
kdata.rbc.astype(int)
kdata['rbc'].fillna(kdata['bp'].median(), inplace=True)


#Count missing values
kdata.isnull().sum().sort_values(ascending=False)

for dataset in kdata:    
    #complete missing age with median
    dataset['age'].fillna(dataset['age'].median(), inplace = True)

   

#Transform non-numeric columns into numerical columns
for column in kdata.columns:
        if kdata[column].dtype == np.number:
            continue
        kdata[column] = LabelEncoder().fit_transform(kdata[column])

kdata.head()

#Count missing values
kdata.isnull().sum().sort_values(ascending=False)
#Write to csv file
kdata.to_csv('I:\G-Drive\PhD\Post_PhD\Kidney\kidney_disease1.csv')
kdata.columns

#Read the new csv file
df = pd.read_csv('I:\G-Drive\PhD\Post_PhD\kidney_disease1.csv')
print(df.columns)
df
df.isnull().sum()


sns.countplot(x="Result", data=df, palette="bwr")
plt.show()

#Calculate the total number of disease / normal user
countNoDisease = len(df[df.classification == 0])
countHaveDisease = len(df[df.classification == 2])
total = (len(df.classification))
print(total)
print(countNoDisease)
print(countHaveDisease)
int(164)/int(303)*100
per = (float(countNoDisease) / float(total))*100
print(per)

#Creating Independent variable
X = df.iloc[:, :-1].values #Takes all rows of all columns except the last column

#Creating Dependent variable
Y = df.iloc[:, -1].values # Takes all rows of the last column
X
Y


#Dealing with missing values with mean imputer
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) 
imputer.fit(X[:,1:24])
X[:,1:24]=imputer.transform(X[:,1:24])
X
print(X)


#Split traing and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
                                                    ,random_state = 0)
X_train
X_test
Y_train
Y_test
print(X_train.shape)
print(X_test.shape)

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

rmse_train = mean_squared_error(Y_test, Y_pred)**(0.5)
print(rmse_train)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred,average='micro'))
print("Recall:",metrics.recall_score(Y_test, Y_pred,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
rmse_train = mean_squared_error(Y_test, Y_pred)**(0.5)
print(rmse_train)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred,average='micro'))
print("Recall:",metrics.recall_score(Y_test, Y_pred,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)




#SVM
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, Y_train)
acc = svm.score(X_test,Y_test)*100
accuracies['Bagging'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
rmse_train = mean_squared_error(Y_test, Y_pred)**(0.5)
print(rmse_train)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred,average='micro'))
print("Recall:",metrics.recall_score(Y_test, Y_pred,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
acc = dtc.score(X_test, Y_test)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred,average='micro'))
print("Recall:",metrics.recall_score(Y_test, Y_pred,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)


# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(X_train, Y_train)
acc = rf.score(X_test,Y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred)

#Comparison of all algorithms
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
#plt.figure(figsize=(12,12))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()



confusion = confusion_matrix(Y_test, acc)
print('Confusion Matrix:')
print(confusion)

print('Accuracy: %3f' % accuracy_score(y_true, lr_pred))
# Determine the false positive and true positive rates
fpr,tpr,roc_auc = auc_scorer(clf_best, X_test, y_test, 'RF')
 