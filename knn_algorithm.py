# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:22:02 2018

@author: Putra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as ply
import scipy as sp

#Load Data Set
dataset = pd.read_csv('Iris.csv')
dataset.head(5)

#Assign dataset
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

#Target is currently in text, so we need to assign numerical value identity
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#split training data from testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#Calculating similarity using Euclidean plane distance measure called L2 Distance
def euclidean_distance(training_set, test_instance):
    #number of samples included in the training set
    n_samples = training_set.shape[0]
    
    #create array for distances
    distances = np.empty(n_samples, dtype=np.float64)
    
    #euclidean distance calculation
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum(np.square(test_instance - training_set[i])))
        
    return distances

class MyKNeighborsClassifier():
    '''
    sample collection
    '''
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
     
        n_samples = X.shape[0]
        #number of neighbor cannot be larger than number of samples
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors cannot be larger than number of traiing sets")
            
        #X and y need to have the same sample size
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of labels must mach with the number of predictors")
        
        #finding and savings all class labels.
        self.X = X
        self.y = y
        
    def predict_from_neighbors(self, training_set, labels, test_instance, k):
        '''
        Predictors
        '''
        distances = euclidean_distance(training_set, test_instance)
        #predict the labels from the distances
        
        #combining arrays as columns
        distances = sp.c_[distances, labels]
        #Sort array by lowest distances
        sorted_distances = distances[distances[:,0].argsort()]
        #Selecting labels from k smallest distances
        targets = sorted_distances[0:k, 1]
        
        unique, counts = np.unique(targets, return_counts=True)
        return(unique[np.argmax(counts)])
    
    def predict(self, X_test):
        #number of predictions to make and number of features inside the sample
        n_predictions, n_features = X_test.shape
        
        #allocating space for array prediction
        predictions = np.empty(n_predictions, dtype=int)
        
        #Loop over all observation
        for i in range(n_predictions):
            #calculate single prediction
            predictions[i] = self.predict_from_neighbors(self.X, self.y, X_test[i,:],self.n_neighbors)
            
        return predictions
    
my_classifier = MyKNeighborsClassifier()

#fitting the model
my_classifier.fit(X_train, y_train)

#predicting from test result
my_y_pred = my_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy:' + str(round(accuracy,2)) + '%')
            