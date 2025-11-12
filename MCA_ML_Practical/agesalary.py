# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:27:50 2019

@author: Pushpalatha
"""

import pandas as pd
data = pd.read_csv("J:\Machine Learning\Class\Practical\Practical_1\age_salary.csv")
print(data.columns)
X = dataset.iloc[:,:-1].values #Takes all rows of all columns except the last column
Y = dataset.iloc[:,-1].values # Takes all rows of the last column