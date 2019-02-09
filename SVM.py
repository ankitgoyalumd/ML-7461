# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:56:52 2019

@author: Ankit Goyal
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import cross_val_score
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn import svm
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
from keras.layers import Dense
import numpy as np
import os


# Downloading the data and storing it into a dataframe
os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1')
data=pd.read_csv('weather_data.csv')
data=data.dropna()



#Mapping RainTomoorow and Rain Today to 0 and 1
data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})
data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})


#Adding dummy variable for attributes with more than 2 categories
Categorical= ['WindGustDir', 'WindDir9am', 'WindDir3pm']
for each in Categorical:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
df = pd.concat([data, dummies], axis=1)
fields_to_drop = ['Date', 'Location','WindGustDir', 'WindDir9am', 'WindDir3pm']
df = df.drop(fields_to_drop, axis=1)

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1/Dataset/nba')
data=pd.read_csv('nba.csv')
data=data.dropna()
print(data.info())
fields_to_drop=['TARGET_5Yrs','Name']
X_2 = data.drop(fields_to_drop,axis=1)
y_2 = data['TARGET_5Yrs']
C=[0.01,0.1,1,2,3,5]
def optim_linear(C,X,y):
    accuracy_train=[]
    accuracy_test=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    for c in C:
        clf = svm.SVC(kernel='linear', C=c).fit(X_train,y_train)
        scores = cross_val_score(clf,X_train,y_train, cv=5)
        accu_train=clf.score(X_train,y_train)
        accu_test=clf.score(X_test,y_test)
        accuracy_train.append(accu_train)
        accuracy_test.append(accu_test)
    accuracy_train1=np.asarray(accuracy_train)
    accuracy_test1=np.asarray(accuracy_test)
    C=np.asarray(C)
    line1, = plt.plot(C,accuracy_train1,color='r',label='train_accuracy')
    line2, = plt.plot(C,accuracy_test1,color='b',label='test_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('C')
    print(accuracy_test1)
    plt.show()
    return None
gamma=[0.001,0.01,0.1,0.2,0.5,1]
def optim_rbf(X,y,gamma):
    accuracy_train=[]
    accuracy_test=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    for g in gamma:
        clf = svm.SVC(kernel='rbf', gamma=g).fit(X_train,y_train)
        scores = cross_val_score(clf,X_train,y_train, cv=5)
        accu_train=clf.score(X_train,y_train)
        accu_test=clf.score(X_test,y_test)
        accuracy_train.append(accu_train)
        accuracy_test.append(accu_test)
    accuracy_train1=np.asarray(accuracy_train)
    accuracy_test1=np.asarray(accuracy_test)
    gamma=np.asarray(gamma)
    line1, = plt.plot(gamma,accuracy_test1,color='b',label='test_accuracy')
    line2, = plt.plot(gamma,accuracy_train1,color='r',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('gamma')
    print(scores)
    plt.show()
    return None
degree=[0,1,2,3]
def optim_poly(X,y,degree):
    accuracy_train=[]
    accuracy_test=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    for d in degree:
        clf = svm.SVC(kernel='poly', degree=d,gamma='auto').fit(X_train,y_train)
        scores = cross_val_score(clf,X_train,y_train, cv=5)
        accu_train=clf.score(X_train,y_train)
        accu_test=clf.score(X_test,y_test)
        accuracy_train.append(accu_train)
        accuracy_test.append(accu_test)
        print(accuracy_test)
    accuracy_train1=np.asarray(accuracy_train)
    accuracy_test1=np.asarray(accuracy_test)
    degree=np.asarray(degree)
    line1, = plt.plot(degree,accuracy_test1,color='b',label='test_accuracy')
    line2, = plt.plot(degree,accuracy_train1,color='r',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('degree')
    plt.show()
    return None

X1=X.iloc[0:2000]
y1=y.iloc[0:2000]
#optim_linear(C,X1,y1)
#optim_rbf(X1,y1,gamma)
optim_poly(X_2,y_2,degree)
#optim_linear(C,X_2,y_2)
#optim_rbf(X_2,y_2,gamma)
optim_poly(X1,y1,degree)
#optim_linear(C,X_2,y_2)
#optim_rbf(X,y,gamma)
#optim_poly(X,y,degree)