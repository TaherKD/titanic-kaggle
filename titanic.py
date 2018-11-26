#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:02:09 2018

@author: taherkd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['Name'], axis = 1)  #remove Name
l1 = preprocessing.LabelEncoder()
dataset.iloc[:, 3] = l1.fit_transform(dataset.iloc[:, 3])   #transform Sex
dataset['Family'] = dataset['SibSp'] + dataset['Parch']
dataset = dataset.drop(['SibSp', 'Parch'], axis = 1)
dataset = dataset.drop(['Ticket', 'Fare'], axis = 1)
dataset = dataset.drop(['Cabin'], axis = 1)     #remove individual columnet
dataset.isnull().any()      #find column that has null values
dataset.iloc[:, 5] = dataset.iloc[:, 5].fillna('X')     #replace nan in Embarked with X
dataset.iloc[:, 4] = dataset.iloc[:, 4].fillna(dataset.iloc[:, 4].mean())   #replace nan in Age with mean
l2 = preprocessing.LabelEncoder()
dataset.iloc[:, 5] = l2.fit_transform(dataset.iloc[:, 5])   #encode Embarked
checkpoint = dataset

X = dataset.iloc[:, [2,3,4,5,6]].values
y = dataset.iloc[:, 1].values

#Split data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Feature Selection
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 3)
fit = rfe.fit(X_train, y_train)
fit.n_features_
fit.support_
fit.ranking_

X_train = X_train[:, [0,1,2]]
X_test = X_test[:, [0,1,2]]
# Fitting Random Forest Classification to the Training set
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Teting Phase on Kaggle data
datasetTesting = pd.read_csv('test.csv')
datasetTesting = datasetTesting.drop(['Name'], axis = 1)
l1 = preprocessing.LabelEncoder()
datasetTesting.iloc[:, 2] = l1.fit_transform(datasetTesting.iloc[:, 2])
datasetTesting['Family'] = datasetTesting['SibSp'] + datasetTesting['Parch']
datasetTesting = datasetTesting.drop(['SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis = 1)
datasetTesting.isnull().any()
l2 = preprocessing.LabelEncoder()
datasetTesting.iloc[:, 4] = l2.fit_transform(datasetTesting.iloc[:, 4])
datasetTesting.iloc[:, 3] = datasetTesting.iloc[:, 3].fillna(datasetTesting.iloc[:, 3].mean())
XK = datasetTesting.iloc[:, [1,2,3,4,5]].values
XK = sc.transform(XK)
XK = XK[:, [0,1,2]]
yk = classifier.predict(XK)
yk = pd.DataFrame(yk)
yk['PassengerId'] = datasetTesting.iloc[:, 0]
yk.to_csv('output.csv')











