# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:42:27 2020

@author: Brad
"""
import numpy as np;
import math;
import matplotlib.pyplot as plt;

def normalizeData(data):

    numFeatures = data.shape[1];
    rows = data.shape[0];
    # normalizes the values using mean normalization
    mean = 0.0;  
    # finds mean, min and max of each feature 
    for j in range (numFeatures) :
        
        sum = 0.0;
        min = 0.0;
        max = 0.0;
        for i in range (rows) : 
            sum = data[i][j] + sum;
            if (i == 0) :
                min = data[i][j];
                max = data[i][j];
            if (data[i][j] > max) :
                max = data[i][j];
            elif (data[i][j] < min) :
                min = data[i][j];
            
           
        if (i == rows-1) :
            mean = sum/rows;
            # adjusts data by subtracting the mean and dividing by the range           
            for k in range (rows) :
         
                data[k][j] = (data[k][j] - mean)/(max - min);
                                           
    return data;

# calculates new weights using gradient descent
def calculateWeights(weights, alpha, data, jVal, iterations) :
    for a in range (iterations):
          # adds columns of 1 for calculating w0 
          bigSum = 0;
         
      
          for j in range(weights.shape[0]):
              
              for i in range(data.shape[0]) :
                  prediction = 0;
                  sum= 0;
                  # calculates exponent in logistic function
                  for k in range (data.shape[1]-1) :
                      sum = weights[k]*data[i,k] + sum;
                   
                  sum = sum + weights[0];
                  # logistic function result
                  hVal = logisticFunc(sum)
                  prediction = (hVal - data[i, data.shape[1]-1])* data[i,j];
                      
                  bigSum = bigSum + prediction;     
              print("big", bigSum)
              weights[j] = weights[j] - (alpha * (1/rows) * bigSum);
              print(weights);
      
    #  J = calculateJFunc(data, weights);
    
      #calculateWeights(weights, alpha, data, newJ, iterations);  
      
    return weights;

def logisticFunc(z) :
    logisticVal = 1/(math.exp(-z)+1)
    return logisticVal
    
def calculateJFunc(data, weights) :
    sum = 0.0;
    n = data.shape[1];
    for i in range (data.shape[0]) :
         # checks the classification to determine how cost will be calculated
            cost = costFunction(data[i], weights);    
            sum = cost + sum;
        
    jVal = (1/data.shape[0]) *sum;
    return jVal;
    
def costFunction(data, weight) :
    n = data.shape[0];
    cost = 0;
    denom = 0;
    for j in range (n) :
        denom = denom + data[j]*weight[j];
        cost = 1/(1 + math.exp(-(denom)));
    
    if (data[n-1] == 0):
        cost = np.log(1- cost)*-1
    else :
        cost = np.log(cost)*-1
        
    return cost;
        

file = input("Enter name of training file: \n");
myFile = open(file, "r");

dataString = myFile.readline();
t = dataString.split("\t");
t[0].strip();

rows = int(t[0])
cols = int(t[1])
# creates empty matrix 
data = np.zeros([rows, cols + 1]);

# fills in matrix
for k in range(rows) :
    dataString = myFile.readline();
    t = dataString.split("\t");
    for j in range(cols+1) :
        t[j].strip();
        if (j < cols+1) :
            data[k, j] = float(t[j]);
      
myFile.close();

newColumn = np.ones(data.shape[0]);
data = np.insert(data, 0, newColumn, axis = 1);

#initalize weights
alpha = 0.01;
weights = np.zeros([data.shape[1]]);
print(data.shape[1]);
print(weights)
J = calculateJFunc(data, weights);
iterations = 2;
#for i in range (iterations):
weights = calculateWeights(weights, alpha, data, J, iterations);

J = calculateJFunc(data, weights);
print(J);
print(weights);

TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i in range (rows):
    sum = 0;
    # calculate z val by multiplying the weight by the data value and
    # summing the result for each row
    for j in range (cols+1):
        sum = sum + weights[j]*data[i,j];        
    logisticResult = logisticFunc(sum)
   
    if (data[i,cols+1] == 0):
         if (logisticResult < 0.5):
             TN = TN + 1;
         else :
             FN = FN + 1;
    else :
        if (logisticResult < 0.5):
             FP = FP + 1;
        else :
             TP = TP + 1;

print("TP = ", TP);
print("FP = ", FP)
print(FN)
print(TN)
        