#!/usr/bin/env python
# coding: utf-8

# ## Exercise 3: Fitting a Linear Regression Model (10 Points)

# #### Task 1: Implement a loss function which measures the average squared difference between the true data and the model prediction, i.e the mean squared error (MSE).

# In[ ]:


# We will make use of numpy to vectorize most of the computations
import numpy as np


# In[ ]:


def loss(y, prediction):
    """
    :param y: The true outputs
    :param prediction: The predictions of your model
    :return: The MSE between the model predictions and the true outputs
    """
    n = len(y)
    sum = 0 
    for i in range(n):
        sum += (y[i]-prediction[i])**2
    
    return (sum / n)


# #### Task 2: Implement a function which describes a linear relationship between the input and output.

# In[ ]:


def linear_model(intercept, slope, x):
    """
    :param intercept: The model intercept
    :param slope: The model slope
    :param x: The model input
    :return: The model prediction on x
    """
    n = len(x)
    model_pred = [0]*n
    for i in range(n):
        model_pred[i] = slope*x[i] + intercept
    
    return model_pred


# #### Task 3: Given different values for the slope and the intercept of your model. Implement a function which returns those that result in the best fit, i.e. minimizes the difference between the true data and the model prediction.

# In[ ]:


def grid_search(intercepts, slopes, x, y):
    """
    :param intercepts: A numpy array of different intercepts
    :param slopes: A numpy array of different slopes
    :param x: The inputs
    :param y: The true outputs
    :return (intercept, slope): The intercept and slope that result in the best fit
    """
    n,m = intercepts.size , slopes.size
    best_fit = [0,0]
    min_error = 2147483647 # max integer value to emphasize that we seek after a minimum for the readibility :)
    
    for i in range(n):
        for j in range(m) :
            model_prediction = linear_model(intercepts[i],slopes[j],x)
            error = loss(y,model_prediction)
            if min_error > error :
                min_error = error
                best_fit = [intercepts[i],slopes[j]]     
    
    return (best_fit[0],best_fit[1])


# #### Task 4: Fit a linear model over some training data and plot the resulting model using matplotlib.

# In[ ]:


# We will use the datasets functionality provided by sklearn to generate some training data
from sklearn.datasets import make_regression

# Let's create some training data to fit our model on

# We first generate 50 1-dimensional samples as our training data. Then we add Gaussian noise on them with the standard deviation 30
x_train, y_train = make_regression(n_samples=50, n_features=1, n_informative=1, noise=30.0)
y_train = y_train[:, None] #  make y a column vector


# In[ ]:


# This is the test data on which we want to evaluate our fitted model
x_test = np.linspace(start=-4, stop=4, num=20)
x_test = x_test[:, None] #  make x_test a column vector


# In[ ]:


# These are the different values for the intercept and slope on which we want to perform a gridsearch
intercepts = np.linspace(start=-10.0, stop=10.0, num=50)
intercepts = intercepts[:, None] #  make intercepts a column vector
slopes = np.linspace(start=0.0, stop=100.0, num=50)
slopes = slopes[:, None] #  make slopes a column vector


# In[ ]:


# TODO: implement
# Fit the model and evaluate the fitted model on the test data
intercept,slope = grid_search(intercepts,slopes,x_train,y_train)
y_predicted = linear_model(intercept,slope,x_test)


# **Complete the code below to plot the training data together with the fitted linear model.**

# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Create a matplotlib figure for the training data and our fitted linear regression model
fig, axes = plt.subplots(1, 1)
axes.scatter(x_train, y_train, color='blue', marker='.', alpha=.6, s=2e2, label='Training Data')
axes.plot(x_test,y_predicted,color='red',alpha= 1)
# Plot the prediction from the fitted model on the test data

