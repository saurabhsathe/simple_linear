# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:04:40 2018

@author: therock
"""

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 1].values
Y.reshape((30,1))
#splitting the dataset in training and test sections
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

#the next part is feature scaling which might be needed sometimes
"""
learn.preprocessing import StandardScaler
scx = StandardScaler()
x_train=scx.fit_transform(x_train)
x_test=scx.transform(x_train)
"""


from sklearn.linear_model import LinearRegression
regres = LinearRegression()
regres.fit(x_train,y_train)

ypred = regres.predict(x_test)

mp.scatter(x_train,y_train,color='red')
mp.plot(x_train,regres.predict(x_train),color='blue')
mp.xlabel('xp')
mp.ylabel('salary')
mp.show()