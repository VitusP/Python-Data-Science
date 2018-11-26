# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:04:28 2018

@author: Putra
K nearest neighbor experiment code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from skilearn.preprocessing import LabelEncoder
from skilearn.model_selection import train_test_split

#load the data required for model creation.
dataset = pd.read_csv('iris.csv')
dataset.head(5)

#Assign X predictors and Y target for the species
X = dataset.iloc[:, 1:5].values #select the column that contain the predictors ignoring the id column on index 0
y = dataset.iloc[:,5].values #number species containing target for our model

#since our target is in word, we need to assign them a label
le = LabelEncoder()
y = le.fit_transform(y) # transform species into categorical values

#split dataset to training and evaluation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
