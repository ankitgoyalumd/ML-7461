# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:19:18 2019

@author: Ankit Goyal
"""
#
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.externals.six import StringIO  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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


#Split the data into test and training sets
def rand_forest(num_trees,depth_max,min_split,X,y,features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()  
    scaler.fit(X_train)
    scaler.fit(X_test)
    model = RandomForestClassifier(n_estimators=num_trees,max_depth=depth_max,min_samples_split=min_split,random_state=0,max_features=features)
    clf=model.fit(X_train,y_train)
    scores=cross_val_score(model,X_train,y_train,cv=5)
    accu_train=clf.score(X_train,y_train)
    accu_test=clf.score(X_test,y_test)
    return scores,accu_train,accu_test

def vary_estimators(num_trees_max,X,y,depth_max,min_split,features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()  
    scaler.fit(X_train)
    scaler.fit(X_test)
    k_score=[]
    accu_test=[]
    accu_train=[]
    for cnt in range(1,num_trees_max):
        l=rand_forest(cnt,depth_max,min_split,X,y,features)
        k_array=np.asarray(l[0])
        k_score.append(np.mean(k_array))
        accu_test.append(l[2])
        accu_train.append(l[1])
    accuracy_test=np.asarray(accu_test)
    accuracy_train=np.asarray(accu_train)
    print(accuracy_test)
    k_fold=np.asarray(k_score)
    line1, = plt.plot(range(1,num_trees_max),accuracy_test,color='r',label='test_accuracy')
    line2, = plt.plot(range(1,num_trees_max),accuracy_train,color='b',label='train_accuracy')
    line3,=plt.plot(range(1,num_trees_max),k_fold,color='g',label='5-fold_score')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of trees')
    plt.figure(figsize=(20,20))
    plt.show()
    return k_fold
#for i in range(6,12):
vary_estimators(10,X_2,y_2,15,100,15)

#for i in range(6,12):
#vary_estimators(10,X,y,10,100,15)

#print(vary_estimators(30))
