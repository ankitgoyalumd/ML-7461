# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:14:10 2019

@author: Ankit Goyal
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
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
import time


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

'''Dataset#2'''

os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1/Dataset/nba')
data=pd.read_csv('nba.csv')
data=data.dropna()
print(data.info())
fields_to_drop=['TARGET_5Yrs','Name']
X_2 = data.drop(fields_to_drop,axis=1)
y_2 = data['TARGET_5Yrs']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
#scaler = StandardScaler()  
#scaler.fit(X_train)
#scaler.fit(X_test)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(13,12,12,12,12,12,12,13,12,12,12,12,12,12,12,2), random_state=1)
#clf.fit(X_train,y_train)
#y_predict = clf.predict(X_test)
#conf_mat=pd.DataFrame(
#    confusion_matrix(y_test, y_predict),
#    columns=['Predicted Not RainTomorrow', 'Predicted RainTomorrow'],
#    index=['True Not RainTomorrow', 'True RainTomorrow']
#)
#print(conf_mat)
#accu_test=clf.score(X_test,y_test)
#print(accu_test)
# Test number of nodes keeping number of hidden layers 2
def cross_val(layers,X,y,nodes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()  
    scaler.fit(X_train)
    scaler.fit(X_test)
    accuracy_test=[]
    accuracy_train=[]
    for i in range(6,nodes):
        layer=tuple([i]*layers)
        print(layer)
        clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=layer,random_state=1)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        accu_test=clf.score(X_test,y_test)
        #print(accu_test)
        accu_train=clf.score(X_train,y_train)
        #accu_test_kfold=cross_val_score(clf,X_train,y_train,cv=5)
        accuracy_test.append(accu_test)
        accuracy_train.append(accu_train)
    accu_test=np.asarray(accuracy_test)
    print(accu_test)
    accu_train=np.asarray(accuracy_train)
    line1, = plt.plot(range(6,nodes),accu_test,color='r',label='test_accuracy')
    line2, = plt.plot(range(6,nodes),accu_train,color='b',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of nodes')
    plt.figure(figsize=(20,20))
    return accuracy_test,accuracy_train
#l=cross_val(6,X_train,y_train,X_test,y_test,20)
#print(l[0])
#accu_test=np.asarray(l[0])
#accu_train=np.asarray(l[1])
#line1, = plt.plot(range(6,20),accu_test,color='r',label='test_accuracy')
#line2, = plt.plot(range(6,20),accu_train,color='b',label='train_accuracy',)
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#plt.ylabel('Accuracy_score')
#plt.xlabel('Number of nodes')
#plt.figure(figsize=(20,20))
#plt.savefig('knn-CV.png')
#plt.show()
#layer=[3]*5
#print(tuple(layer))
    
def cross_val_layer(layers,X,y,nodes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    scaler.fit(X_test)
    accuracy_test=[]
    accuracy_train=[]
    for i in range(3,layers):
        layer=tuple([nodes]*i)
        print(layer)
        clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=layer,random_state=1)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        accu_test=clf.score(X_test,y_test)
        #print(accu_test)
        accu_train=clf.score(X_train,y_train)
        accuracy_test.append(accu_test)
        accuracy_train.append(accu_train)
    accu_test=np.asarray(accuracy_test)
    print(accu_test)
    accu_train=np.asarray(accuracy_train)
    line1, = plt.plot(range(3,layers),accu_test,color='r',label='test_accuracy')
    line2, = plt.plot(range(3,layers),accu_train,color='b',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of layers')
    plt.figure(figsize=(20,20))
    return accuracy_test,accuracy_train

#cross_val(7,X_2,y_2,20)

def iterations(X,y,n):
    X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.20) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    scaler.fit(X_test)
    accuracy_test=[]
    accuracy_train=[]
    for iter in range(100,n,100):
        layer=tuple([7]*7)
        clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=layer,random_state=1,max_iter=iter)
        clf.fit(X_train,y_train)
        accu_test=clf.score(X_test,y_test)
        #print(accu_test)
        accu_train=clf.score(X_train,y_train)
        accuracy_test.append(accu_test)
        accuracy_train.append(accu_train)
        print(accuracy_test)
    accu_test=np.asarray(accuracy_test)
    print(accu_test)
    accu_train=np.asarray(accuracy_train)
    line1, = plt.plot(range(100,n,100),accu_test,color='r',label='test_accuracy')
    line2, = plt.plot(range(100,n,100),accu_train,color='b',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of iterations')
    plt.figure(figsize=(20,20))
    return accuracy_test,accuracy_train

#iterations(X_2,y_2,1000)
lr=[0.0001,0.0005,0.001,0.005,0.01,0.02,0.04,0.05,0.07,0.1,0.2,0.5,1,2]
def learn_rate(learn):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    time2=[]
    scaler = StandardScaler()  
    scaler.fit(X_train)        
    scaler.fit(X_test)
    accuracy_test=[]
    accuracy_train=[]
    for l in learn:
        clf=MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=tuple([8]*5),random_state=1,learning_rate_init=l)
        start_time=time.time()
        clf.fit(X_train,y_train)
        t=time.time()-start_time
        time2.append(t)
        print(time2)
        #print(accu_test)
    time1=np.asarray(time2)
    lr1=np.asarray(lr)
    line1, = plt.plot(lr1,time1,color='r',label='learning_rate vs time')
    #line2, = plt.plot(range(100,n,100),accu_train,color='b',label='train_accuracy',)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Time(Seconds)')
    plt.xlabel('Learning Rate')
    plt.figure(figsize=(20,20))
    return accuracy_test,accuracy_train

learn_rate(lr)