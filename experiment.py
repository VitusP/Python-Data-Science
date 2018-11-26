# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from scipy import linalg


dataset = np.array([0,1,2,3,4,5,3,2,5,3])

def numpy_experiment():
    #Mean Calculation
    mean = np.mean(dataset)
    
    #Median Calculation
    median = np.median(dataset)
    
    print('Mean: {} || Median: {}'.format(mean, median))
    
    #Array Declaration
    x = np.array([[3,2,4],[2,3,5]])
    y = np.array([[4,6,2],[2,4,8]])
    
    print('Cross product: {}'.format(np.cross(x, y)))
    
def pandas_experiment():
    # Two structures of data: Series and Dataframe
    #Series
    s = pd.Series([1,4,6,2,5,8,4,np.nan, 10,27,23,34])
    print('Series: {}'.format(s))
    
    #dframe
    df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
    print('{}'.format(df))
    
def scipy_experiment():
    z = np.array([[1,5],[3,8]])
    print(linalg.inv(z))

def matplotlib_experiment():
    #Compute the x and y coordinate
    x = np.arange(0, 3*np.pi, 0.1)
    y = np.sin(x)
    
    #plot the point
    plt.plot(x,y)
    #call the plot
    plt.show()
    
def scikitlearn_experiment():
    #load the iris datasets
    dataset = datasets.load_iris()
    #Fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(dataset.data, dataset.target)
    print(model)
    
    #make prediction
    expected = dataset.target
    predicted = model.predict(dataset.data)
    
    #summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    
    
#numpy_experiment()
#pandas_experiment()
#scipy_experiment()
#matplotlib_experiment()
scikitlearn_experiment()


