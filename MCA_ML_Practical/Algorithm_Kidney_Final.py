# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:10:36 2020

@author: Pushpalatha
"""
'''https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease'''

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

dataset1 = pd.read_csv("E:\\Latha\\PhD\\ALL_REPORTS\\KidneyDisease\\kidney_disease.csv")
dataset1.shape
dataset1.info()
dataset1.head()

#Visualize
import matplotlib.pyplot as plt
dataset1.hist(bins=50, figsize=(20, 15))
plt.show()

#Mean imputation & Convert datatype as integer
dataset1['age']=dataset1['age'].replace(np.NaN,dataset1['age'].mean())
dataset1['age'] = dataset1.age.astype(int)
dataset1['bp']=dataset1['bp'].replace(np.NaN,dataset1['bp'].mean())
dataset1['bp'] = dataset1.age.astype(int)
dataset1.info()


#Transform into numeric
number = LabelEncoder()
dataset1['rbc'] = number.fit_transform(dataset1['rbc'])
dataset1['pc'] = number.fit_transform(dataset1['pc'])
dataset1['pcc'] = number.fit_transform(dataset1['pcc'])
dataset1['ba'] = number.fit_transform(dataset1['ba'])
dataset1['htn'] = number.fit_transform(dataset1['htn'])
dataset1['dm'] = number.fit_transform(dataset1['dm'])
dataset1['cad'] = number.fit_transform(dataset1['cad'])
dataset1['appet'] = number.fit_transform(dataset1['appet'])
dataset1['pe'] = number.fit_transform(dataset1['pe'])
dataset1['ane'] = number.fit_transform(dataset1['ane'])
dataset1['dm'] = number.fit_transform(dataset1['dm'])
dataset1['classification'] = number.fit_transform(dataset1['classification'])
dataset1.head()

#Write trasnformed value in new file
dataset1.to_csv('E:\\Latha\\PhD\\ALL_REPORTS\\KidneyDisease\\chronic_kidney.csv')

#Read new file
dataset = pd.read_csv("E:\\Latha\\PhD\\ALL_REPORTS\\KidneyDisease\\chronic_kidney.csv")
dataset.info()
dataset.shape()

#Count missing values
dataset.isnull().sum().sort_values(ascending=False)
dataset.info()
dataset['age']=dataset['age'].replace(np.NaN,dataset1['age'].mean())
dataset['bp']=dataset['bp'].replace(np.NaN,dataset['bp'].mean())
dataset['sg']=dataset['sg'].replace(np.NaN,dataset['sg'].mean())
dataset['al']=dataset['al'].replace(np.NaN,dataset['al'].mean())
dataset['su']=dataset['su'].replace(np.NaN,dataset['su'].mean())
dataset['bgr']=dataset['bgr'].replace(np.NaN,dataset['bgr'].mean())
dataset['bu']=dataset['bu'].replace(np.NaN,dataset['bu'].mean())
dataset['sc']=dataset['sc'].replace(np.NaN,dataset['sc'].mean())
dataset['sod']=dataset['sod'].replace(np.NaN,dataset['sod'].mean())
dataset['pot']=dataset['pot'].replace(np.NaN,dataset['pot'].mean())
dataset['hemo']=dataset['hemo'].replace(np.NaN,dataset['hemo'].mean())
dataset['dm']=dataset['dm'].replace(np.NaN,dataset['dm'].mean())
dataset.info()
dataset.isnull().sum().sort_values(ascending=False)
dataset.to_csv('E:\\Latha\\PhD\\ALL_REPORTS\\KidneyDisease\\chronic_kidney_disease3.csv')

dataset = pd.read_csv("E:\\Latha\\PhD\\ALL_REPORTS\\KidneyDisease\\chronic_kidney_disease3.csv")
dataset.shape()
#Creating Independent variable
X = dataset.iloc[:, :-1].values #Takes all rows of all columns except the last column

#Creating Dependent variable
Y = dataset.iloc[:, -1].values # Takes all rows of the last column
X
Y

#Split traing and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
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
Y_pred_NB = gnb.predict(X_test) 
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred_NB)*100)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred_NB))
print("Precision:",metrics.precision_score(Y_test, Y_pred_NB,average='micro'))
print("Recall:",metrics.recall_score(Y_test, Y_pred_NB,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, Y_pred_NB)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, Y_pred_NB)

rmse_train = mean_squared_error(Y_test, Y_pred_NB)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

y_pred_nb = gnb.predict(X_test)
def display_summary(true,pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true,pred).ravel()
    print(tn, fp, fn, tp)
    print('confusion matrix')
    print(np.array([[tp,fp],[fn,tn]]))
    print('sensitivity is %f',1.*tp/(tp+fn))
    print('specificity is %f',1.*tn/(tn+fp))
    print('accuracy is %f',1.*(tp+tn)/(tp+tn+fp+fn))
    print('balanced accuracy is %',1./2*(1.*tp/(tp+fn)+1.*tn/(tn+fp)))
 
print('Gaussian NB')
display_summary(Y_test,Y_pred_NB)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
print("Accuracy:",metrics.accuracy_score(Y_test, prediction))
print("Precision:",metrics.precision_score(Y_test, prediction,average='micro'))
print("Recall:",metrics.recall_score(Y_test, prediction,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, prediction)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, prediction)
rmse_train = mean_squared_error(Y_test, prediction)**(0.6)
print('\nRMSE on train dataset : ', rmse_train)

def display_summary(true,pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true,pred).ravel()
    print(tn, fp, fn, tp)
    print('confusion matrix')
    print(np.array([[tp,fp],[fn,tn]]))
    print('sensitivity is %f',1.*tp/(tp+fn))
    print('specificity is %f',1.*tn/(tn+fp))
    print('accuracy is %f',1.*(tp+tn)/(tp+tn+fp+fn))
    print('balanced accuracy is %',1./2*(1.*tp/(tp+fn)+1.*tn/(tn+fp)))
 
print('Gaussian NB')
display_summary(Y_test,prediction)


#SVM
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, Y_train)
pred_svm = svm.predict(X_test)
#acc = svm.score(X_test,Y_test)*100
accuracies['Bagging'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
print("Accuracy:",metrics.accuracy_score(Y_test, pred_svm))
print("Precision:",metrics.precision_score(Y_test, pred_svm,average='micro'))
print("Recall:",metrics.recall_score(Y_test, pred_svm,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, pred_svm)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, pred_svm)
rmse_train = mean_squared_error(Y_test, pred_svm)**(0.6)
print('\nRMSE on train dataset : ', rmse_train)

def display_summary(true,pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true,pred).ravel()
    print(tn, fp, fn, tp)
    print('confusion matrix')
    print(np.array([[tp,fp],[fn,tn]]))
    print('sensitivity is %f',1.*tp/(tp+fn))
    print('specificity is %f',1.*tn/(tn+fp))
    print('accuracy is %f',1.*(tp+tn)/(tp+tn+fp+fn))
    print('balanced accuracy is %',1./2*(1.*tp/(tp+fn)+1.*tn/(tn+fp)))
 
print('Gaussian NB')
display_summary(Y_test,pred_svm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
y_pred_dc = dtc.predict(X_test)
#acc_dt = dtc.score(X_test, Y_test)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_dc))
print("Precision:",metrics.precision_score(Y_test, y_pred_dc,average='micro'))
print("Recall:",metrics.recall_score(Y_test, y_pred_dc,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, y_pred_dc)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, y_pred_dc)
rmse_train = mean_squared_error(Y_test, y_pred_dc)**(0.6)
print('\nRMSE on train dataset : ', rmse_train)

def display_summary(true,pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true,pred).ravel()
    print(tn, fp, fn, tp)
    print('confusion matrix')
    print(np.array([[tp,fp],[fn,tn]]))
    print('sensitivity is %f',1.*tp/(tp+fn))
    print('specificity is %f',1.*tn/(tn+fp))
    print('accuracy is %f',1.*(tp+tn)/(tp+tn+fp+fn))
    print('balanced accuracy is %',1./2*(1.*tp/(tp+fn)+1.*tn/(tn+fp)))
 
print('Gaussian NB')
display_summary(Y_test,y_pred_dc)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
#acc = rf.score(X_test,Y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(y_pred_rf))
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_rf))
print("Precision:",metrics.precision_score(Y_test, y_pred_rf,average='micro'))
print("Recall:",metrics.recall_score(Y_test, y_pred_rf,average='micro'))
print '\n clasification report:\n', metrics.classification_report(Y_test, y_pred_rf)
print '\n confussion matrix:\n',metrics.confusion_matrix(Y_test, y_pred_rf)
rmse_train = mean_squared_error(Y_test, y_pred_rf)**(0.6)
print('\nRMSE on train dataset : ', rmse_train)

def display_summary(true,pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true,pred).ravel()
    print(tn, fp, fn, tp)
    print('confusion matrix')
    print(np.array([[tp,fp],[fn,tn]]))
    print('sensitivity is %f',1.*tp/(tp+fn))
    print('specificity is %f',1.*tn/(tn+fp))
    print('accuracy is %f',1.*(tp+tn)/(tp+tn+fp+fn))
    print('balanced accuracy is %',1./2*(1.*tp/(tp+fn)+1.*tn/(tn+fp)))
 
print('Gaussian NB')
display_summary(Y_test,y_pred_rf)

#Comparison of all algorithms
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

