#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:55:07 2017

@author: huangshuhui
"""  
    
##import data##
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import random
from scipy.stats import multivariate_normal
from sklearn import preprocessing


train_data = scipy.io.loadmat('/Users/huangshuhui/Google Drive/study/cs289/hw2017/project/data/train.mat')
train_y=train_data['trainX'][...,784]
train_x=train_data['trainX'][...,0:784]
test_data=scipy.io.loadmat('/Users/huangshuhui/Desktop/study/CS289/hw2017/hw3/mnist_dist/test.mat')
test_x=test_data['testX']
train_x=preprocessing.normalize(train_x,norm='l2')
test_x=preprocessing.normalize(test_x,norm='l2')

x_y = list(zip(train_x, train_y))
random.shuffle(x_y)
train_x = np.array([e[0] for e in x_y])
train_label = np.ravel([e[1] for e in x_y])

train_y = np.zeros(shape=(train_x.shape[0],10))
for i in range(len(train_y)):
    train_y[i][train_label[i]]=1

##RBF function##
from scipy import *
from scipy.linalg import norm, pinv
 
from matplotlib import pyplot as plt
 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        # calculate activations of RBFs
        G = self._calcAct(X)
         
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = dot(G, self.W)
        predicted_labels = Y.argmax(axis=1)
        return predicted_labels
        
##test RBF with MNIST data##
rbf=RBF(784,10,10)
rbf.train(train_x,train_y)
rbf.test(train_x)