# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:02:09 2019

@author: Pushpalatha
"""

import pandas as pd
dataset = pd.read_csv('E:\\Latha\\LathaSKPIMCS\\Machine Learning\\Class\Practical\\Preprocessing\\Data1.csv')
print(dataset.columns)
dataset


dataset.info()
dataset.head()

#Row and column count
dataset.shape

#Removing insufficient column
dataset_new = dataset.drop(['Age',], axis = 1)
dataset_new

#To measure the central tendency of variables
dataset_new.describe()

#To change column name
dataset.rename(index  =str, columns={'Country' : 'Countries',
                                     'Age' : 'age',
                                     'Salary' : 'Sal',
                                     'Purchased' : 'Purchased'}, inplace = True)

dataset

#Count missing values
dataset.isnull().sum()

#Print the missing value column
dataset[dataset.isnull().any(axis=1)].head()

#Remove missing value rows
ds_new = dataset.dropna()
ds_new
ds_new.isnull().sum()

#To check datatype
ds_new.dtypes

#To convert as integer
ds_new['age'] = ds_new['age'].astype('int64')

ds_new.dtypes


