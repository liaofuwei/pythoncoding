#coding:utf-8
__author__ = 'Administrator'

import numpy as np
import random
from numpy import genfromtxt

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        #print loss
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta

def newbatchGradientDescent(x, y, theta, alpha, m, maxIteration):
    theta = theta.transpose()   #把theta转换成列向量
    xTrains = x.transpose()
    for i in range(0, maxIteration):
        hy = np.dot(x, theta)   #点乘以后是一个列向量
        loss = hy - y          #loss--->列向量
        gradient = np.dot(xTrains, loss) / m    #gradient列向量
        theta = theta - alpha * gradient
    print loss
    return theta
'''
def newbatchGradientDescentByEpsilon(x, y, theta, alpha, m, Epsilon):
    theta = theta.transpose()   #把theta转换成列向量
    xTrains = x.transpose()
    where()


    for i in range(0, maxIteration):
        hy = np.dot(x, theta)   #点乘以后是一个列向量
        loss = hy - y          #loss--->列向量
        gradient = np.dot(xTrains, loss) / m    #gradient列向量
        theta = theta - alpha * gradient
    return theta
'''


'''
iput x 是行向量或者矩阵
'''
def predict(testdata, theta):

    m, n = np.shape(testdata)
    xTest = np.array(testdata[:, :])
    yP = np.dot(xTest, theta)
    return yP

def readData(dataSet):
    m, n = np.shape(dataSet)
    #trainData = np.ones((m, n))
    trainData = np.array(dataSet[:, :(n-1)])
    trainLabel = dataSet[:, -1]
    return trainData, trainLabel

path = "C:\\pythoncoding\\mechine learning\\machine_learning-master\\machine learning demo\\data1.csv"
dataSet = genfromtxt(path, delimiter=',')
testdatapath = "C:\\pythoncoding\\mechine learning\\machine_learning-master\\machine learning demo\\testdata1.csv"
testdata = genfromtxt(testdatapath,delimiter=',')


trainData, trainLabel = readData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.001
maxIteration = 100000

print trainData
print trainLabel
print theta
theta = newbatchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
#x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print predict(testdata, theta)