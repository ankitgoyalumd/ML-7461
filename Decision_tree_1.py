# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import all the required libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
#from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

#Split the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
clf=model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(clf.score(X_train, y_train))
print(clf.score(X_test,y_test))

# Checking the confusion matrix
conf_mat=pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not RainTomorrow', 'Predicted RainTomorrow'],
    index=['True Not RainTomorrow', 'True RainTomorrow']
)

print(conf_mat)

"""
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,feature_names=X.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
"""
## Pruning the maximum depth
def depth_prune(max_depth,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    maximum_depth = np.linspace(1, max_depth, max_depth, endpoint=True)
    train_accuracy = []
    test_accuracy = []
    k_fold=[]
    for max in maximum_depth:
        model1= tree.DecisionTreeClassifier(max_depth=max)
        mod=model1.fit(X_train, y_train)
        train_pred = model1.predict(X_train)
        accu_train=mod.score(X_train,y_train)
        accu_test=mod.score(X_test,y_test)
        accu_test_kfold=cross_val_score(model1,X_train,y_train,cv=5)
        k_fold.append(np.mean(accu_test_kfold))
        train_accuracy.append(accu_train)
        test_accuracy.append(accu_test)
    train_accuracy=np.asarray(train_accuracy)
    test_accuracy=np.asarray(test_accuracy)
    k_fold=np.asarray(k_fold)
    print(k_fold)
    line1, = plt.plot(maximum_depth,train_accuracy,color='r',label='train_accuracy')
    line2, = plt.plot(maximum_depth,test_accuracy,color='b',label='test_accuracy')
    line3, = plt.plot(maximum_depth,k_fold,color='g',label='10-fold_test_accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Depth of the tree')
    plt.show()
    return None
#depth_prune(20,X,y)
#depth_prune(20,X_2,y_2)
#Vary the test size and check the performance
    
def train_size(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    train_accuracy1=[]
    test_accuracy1=[]
    k_fold=[]
    train_size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for size in train_size:
        X1_train,X2_test,y1_train,y1_test=train_test_split(X_train,y_train,train_size=size,test_size=0.1,random_state=1)
        model2=tree.DecisionTreeClassifier(max_depth=6)
        mod=model2.fit(X1_train,y1_train)
        train1_pred=model2.predict(X1_train)
        accu_test_kfold=cross_val_score(model2,X_train,y_train,cv=5)
        accu_train1=mod.score(X1_train,y1_train)
        accu_test1=mod.score(X_test,y_test)
        train_accuracy1.append(accu_train1)
        k_fold.append(np.mean(accu_test_kfold))
        test_accuracy1.append(accu_test1)
    train_accuracy1=np.asarray(train_accuracy1)
    test_accuracy1=np.asarray(test_accuracy1)
    train_size=np.asarray(train_size)
    k_fold=np.asarray(k_fold)
#print(train_size.shape)
    line1, = plt.plot(train_size,train_accuracy1,color='r',label='train_accuracy')
    line2, = plt.plot(train_size,test_accuracy1,color='b',label='test_accuracy')
    line3, = plt.plot(train_size,k_fold,color='g',label='5-fold_test_accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Train Size fraction')
    plt.show()
    return None
#train_size(X_2,y_2)
#train_size(X,y)
#Maximum number of features to consider when looking for best split, randomly selects specified features to make a split
def features(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    features=list(range(1,X.shape[1]))
    train_accuracy2=[]
    test_accuracy2=[]
    k_fold=[]
    for feat in features:
        model3=tree.DecisionTreeClassifier(max_features=feat,max_depth=6)
        mod=model3.fit(X_train,y_train)
        train2_pred=model3.fit(X_train,y_train)
        accu_test_kfold=cross_val_score(model3,X_train,y_train,cv=5)
        accu_train2=mod.score(X_train,y_train)
        accu_test2=mod.score(X_test,y_test)
        train_accuracy2.append(accu_train2)
        k_fold.append(np.mean(accu_test_kfold))
        test_accuracy2.append(accu_test2)
    train_accuracy2=np.asarray(train_accuracy2)
    test_accuracy2=np.asarray(test_accuracy2)
    features=np.asarray(features)
    k_fold=np.asarray(k_fold)
    line1, = plt.plot(features,train_accuracy2,color='r',label='train_accuracy')
    line2, = plt.plot(features,test_accuracy2,color='b',label='test_accuracy')
    line3, = plt.plot(features,k_fold,color='g',label='5-fold_test_accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('Number of features')
    plt.show()
    return None
features(X,y)

features(X_2,y_2)
test = SelectKBest(score_func=chi2, k=6)
#fit = test.fit(X,y)
np.set_printoptions(precision=3)
#score=pd.DataFrame(fit.scores_)
#features_1 = fit.transform(X)

#X_3=X_2.loc[:,['GP','MIN','PTS','FGM','FGA','FTA','REB']]
#print(X_3.head())
#depth_prune(20,X_3,y_2)
#print(X_2['GP','MIN'].head())
#model=tree.DecisionTreeClassifier(max_features=10,max_depth=10)
#parameters={'min_samples_split' : range(10,1000,20),
#sample_split_range = list(range(1, 50))
#param_grid = dict(min_samples_split=sample_split_range)
def min_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    train_accuracy2=[]
    test_accuracy2=[]
    k_fold=[]
    sample_split_range = list(range(100,1000))
    for split in sample_split_range:
        model3=tree.DecisionTreeClassifier(max_features=10,max_depth=7,min_samples_split=split)
        mod=model3.fit(X_train,y_train)
        train2_pred=model3.fit(X_train,y_train)
        accu_test_kfold=cross_val_score(model3,X_train,y_train,cv=5)
        accu_train2=mod.score(X_train,y_train)
        accu_test2=mod.score(X_test,y_test)
        train_accuracy2.append(accu_train2)
        k_fold.append(np.mean(accu_test_kfold))
        test_accuracy2.append(accu_test2)
    train_accuracy2=np.asarray(train_accuracy2)
    test_accuracy2=np.asarray(test_accuracy2)
    splits=np.asarray(sample_split_range)
    k_fold=np.asarray(k_fold)
    line1, = plt.plot(splits,train_accuracy2,color='r',label='train_accuracy')
    line2, = plt.plot(splits,test_accuracy2,color='b',label='test_accuracy')
    line3, = plt.plot(splits,k_fold,color='g',label='5-fold_test_accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('min_sample_split')
    plt.show()
    return None
   

#min_split(X,y)
#grid(X,y)
#y_predict = model.predict(X_test)

##
 #param_grid = {"criterion": ["gini", "entropy"],
  #            "min_samples_split": [2, 10, 20],
   #           "max_depth": [None, 2, 5, 10],
    #          "min_samples_leaf": [1, 5, 10],
     #         "max_leaf_nodes": [None, 5, 10, 20],
      #        }
    
#grid_search= run_gridsearch(X, y, model, param_grid, cv=10)

    
