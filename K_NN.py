# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 23:47:21 2019

@author: Ankit Goyal
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
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
 
X_1=X.iloc[0:10000]
#y_1=y.loc[0:10000]
#conf_mat=pd.DataFrame(
 #   confusion_matrix(y_test, y_pred),
  #  columns=['Predicted Not RainTomorrow', 'Predicted RainTomorrow'],
   # index=['True Not RainTomorrow', 'True RainTomorrow']
#)
#score=accuracy_score(y_pred,y_test)
#print(score)
#print(conf_mat)
#print(classification_report(y_test, y_pred))  
#print(classifier.score(X_test,y_test))
# Varying number of neighbors 

'''
Dataset#2
'''
os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1/Dataset/nba')
data=pd.read_csv('nba.csv')
data=data.dropna()
print(data.info())
fields_to_drop=['TARGET_5Yrs','Name']
X_2 = data.drop(fields_to_drop,axis=1)
y_2 = data['TARGET_5Yrs']


def neighbors(k,X,y,p):
    train_accu1=[]
    test_accu1=[]
    train_accu2=[]
    test_accu2=[]
    k_array=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train1=X_train.iloc[0:40000]
    y_train1=y_train.iloc[0:40000]
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train1 = scaler.transform(X_train1)  
    X_test = scaler.transform(X_test) 
    for k in range(3,k):
        classifier1 = KNeighborsClassifier(n_neighbors=k,weights='uniform',algorithm='auto',p=p)
        classifier2=KNeighborsClassifier(n_neighbors=k,weights='distance',algorithm='auto',p=p)
        start_time=time.time()
        classifier1.fit(X_train1, y_train1)  
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time=time.time()
        classifier2.fit(X_train,y_train)
        print("--- %s seconds ---" % (time.time() - start_time))
        accu_train1=classifier1.score(X_train1,y_train1)
        accu_test1=classifier1.score(X_test,y_test)
        accu_train2=classifier2.score(X_train1,y_train1)
        accu_test2=classifier2.score(X_test,y_test)
        train_accu1.append(accu_train1)
        test_accu1.append(accu_test1)
        train_accu2.append(accu_train2)
        test_accu2.append(accu_test2)
        k_array.append(k)
    train_accuracy=np.asarray(train_accu1)
    test_accuracy=np.asarray(test_accu1)
    train_accuracy2=np.asarray(train_accu2)
    test_accuracy2=np.asarray(test_accu2)
    k_array=np.asarray(k_array)
    #print(k_arrayprint()
    print(test_accuracy)
    print(train_accuracy)
    print(test_accuracy2)
    print(train_accuracy2)
    line1, = plt.plot(k_array,train_accuracy,color='r',label='train_accuracy_uniform')
    line2, = plt.plot(k_array,test_accuracy,color='b',label='test_accuracy_uniform')
    line3, = plt.plot(k_array,train_accuracy2,color='g',label='train_accuracy_distance')
    line4, = plt.plot(k_array,test_accuracy2,color='k',label='test_accuracy_distance')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of nearest neighbors')
    plt.show()
    return None

#neighbors(15,X,y,1)
def best_fit(X,y,p,algo,k,weights):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()  
    scaler.fit(X_train)
    classifier1 = KNeighborsClassifier(n_neighbors=k,weights=weights,algorithm=algo,p=p)
    start_time=time.time()
    classifier1.fit(X,y)
    accu_train1=classifier1.score(X_train,y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    return accu_train1

def train_frac(X,y,p,k):
    train_accu1=[]
    test_accu1=[]
    train_accu2=[]
    test_accu2=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test) 
    train_size=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for size in train_size:
        X1_train,X2_test,y1_train,y1_test=train_test_split(X_train,y_train,train_size=size,test_size=0.2,random_state=1)
        classifier1 = KNeighborsClassifier(n_neighbors=k,weights='uniform',algorithm='auto',p=p)
        classifier2=KNeighborsClassifier(n_neighbors=k,weights='distance',algorithm='auto',p=p)
        start_time=time.time()
        classifier1.fit(X1_train, y1_train)  
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time=time.time()
        classifier2.fit(X1_train,y1_train)
        print("--- %s seconds ---" % (time.time() - start_time))
        accu_train1=classifier1.score(X1_train,y1_train)
        accu_test1=classifier1.score(X_test,y_test)
        accu_train2=classifier2.score(X1_train,y1_train)
        accu_test2=classifier2.score(X_test,y_test)
        train_accu1.append(accu_train1)
        test_accu1.append(accu_test1)
        train_accu2.append(accu_train2)
        test_accu2.append(accu_test2)
    train_accuracy=np.asarray(train_accu1)
    test_accuracy=np.asarray(test_accu1)
    train_accuracy2=np.asarray(train_accu2)
    test_accuracy2=np.asarray(test_accu2)
    train_size=np.asarray(train_size)
    #print(k_arrayprint()
    print(test_accuracy)
    print(train_accuracy)
    print(test_accuracy2)
    print(train_accuracy2)
    line1, = plt.plot(train_size,train_accuracy,color='r',label='train_accuracy_uniform')
    line2, = plt.plot(train_size,test_accuracy,color='b',label='test_accuracy_uniform')
    line3, = plt.plot(train_size,train_accuracy2,color='g',label='train_accuracy_distance')
    line4, = plt.plot(train_size,test_accuracy2,color='k',label='test_accuracy_distance')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Train size fraction')
    plt.show()
    return None
            
train_frac(X,y,1,6)

#print(best_fit(X,y,2,'auto',3,'uniform'))



    