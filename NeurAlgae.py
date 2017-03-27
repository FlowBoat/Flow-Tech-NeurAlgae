# FlowTech | NeurAlgae
## 2017 WWSEF Science Fair | HAB Prediction Using Machine Learning Algorithms

#Describes and trains a neural network for the analysis and prediction of algal bloom data
#Copyright (C) 2017 AH Zachary Trefler and Atif Mahmud

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

#If you have comments, questions, concerns, or you just want to say 'hi',
#email Zachary Trefler at zmct99@gmail.com or Atif Mahmud at atifmahmud101@gmail.com

import numpy as np
import random
from scipy import *

class Neural_Network(object):
    def __init__(self, inputLayerSize = 2, outputLayerSize = 1, hiddenLayerSize = 3, Lambda = 0):
        #Layers (Hyperparameters)
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        #Regularization
        self.Lambda = Lambda
    def forward(self, x):
        #Propagate inputs through network
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    def sigmoid(self, z):
        #Apply sigmoid activation function
        return 1 / (1 + np.exp(-z))
    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    def costFunction(self, x, y):
        #Compute cost for given x, y, use weights already stored in class
        self.yHat = self.forward(x)
        J = 0.5 * np.sum((y - self.yHat) ** 2) / x.shape[0] + (self.Lambda / 2) * (sum(self.W1 ** 2) + sum(self.W2 ** 2))
        return J
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to W1 and W2
        self.yHat = self.forward(x)
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3) / x.shape[0] + self.Lambda * self.W2
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(x.T, delta2) / x.shape[0] + self.Lambda * self.W1
        return dJdW1, dJdW2
    #Helper functions for interacting with other classes
    def getParams(self):
        #Get W1 and W2 rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
    def computeGradients(self, x, y):
        dJdW1, dJdW2 = self.costFunctionPrime(x, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, x, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4
    for p in range(len(paramsInitial)):
        #Set perturbation vector:
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(x, y)
        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(x, y)
        #Compute numerical gradient:
        numgrad[p] = (loss2 - loss1) / (2 * e)
        #Return the value we changed to zero:
        perturb[p] = 0
    #Return params to original value:
    N.setParams(paramsInitial)
    return numgrad

class trainer(object):
    def __init__(self, N, maxiter = 200):
        #Make local reference to network:
        self.N = N
        self.maxiter = maxiter
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.x, self.y))
        self.testJ.append(self.N.costFunction(self.testx, self.testy))
    def costFunctionWrapper(self, params, x, y):
        self.N.setParams(params)
        cost = self.N.costFunction(x, y)
        grad = self.N.computeGradients(x, y)
        return cost, grad
    def train(self, trainx, trainy, testx, testy):
        #Make an internal variable for the callback function:
        self.x = trainx
        self.y = trainy
        self.testx = testx
        self.testy = testy
        #Make empty list to store costs:
        self.J = []
        self.testJ = []
        params0 = self.N.getParams()
        options = {"maxiter" : self.maxiter, "disp" : True}
        _res = scipy.optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = "BFGS", args = (trainx, trainy), options = options, \
                callback = self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

#How to use it all:
    #import NeuralAlgae as nal
    #Load initial datasets:
        #x = nal.np.array(([], [], [], [], ...), dtype = float)
            #11 elements in each []: [watertemp, nitrogen, etc...]
            #Number of [] = number of complete sets of data
        #y = nal.np.array(([], [], [], [], ...), dtype = float)
            #1 element in each []: [algalbloom (quantified how?)]
            #Same number of [] as in x

    #Normalize:
        #x = x / nal.np.amax(x, axis = 0)
        #y = y / nal.np.amax(y, axis = 0)
            #If maximum value of any x thing or the y thing is known, instead do:
                #y = y / maxy
                #Or something to that effect

    #Divide into testing, training, and cross-validation data:
        #trainx, testx, validx = nal.np.array((), dtype = float)
        #trainy, testy, validy = nal.np.array((), dtype = float)
        #for i in range(len(x)):
            #r = nal.random.randrange(0, 3)
            #if r == 0:
                #trainx.append(x[i])
                #trainy.append(y[i])
            #elif r == 1:
                #testx.append(x[i])
                #testy.append(y[i])
            #elif r == 2:
                #validx.append(x[i])
                #validy.append(y[i])

    #Create neural network:
        #NN = nal.Neural_Network(inputLayerSize = 11, outputLayerSize = 1, hiddenLayerSize = 12, Lambda = 0.0001)

    #Check gradients:
        #numgrad = nal.computeNumericalGradient(NN, x, y)
        #grad = NN.computeGradients(x, y)
        #nal.np.linalg.norm(grad - numgrad) / nal.np.linalg.norm(grad + numgrad)
            #result should be less than 1e-8

    #Train network:
        #T = nal.trainer(NN, maxiter = 1000)
        #T.train(NN, nal.trainx, nal.trainy, nal.testx, nal.testy)

    #See results:
        #trainy
        #NN.forward(trainx)
        #testy
        #NN.forward(testx)
        #validy
        #NN.forward(validx)

import json
import random
import sys
import numpy as np
import scipy as sp
import tensorflow as tf

class Cost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
        #For mutiple ys?
        #If y = y_1, y_2, ... are desired output values, a^L = a^L_1, a^L_2, ... are actual output values (in layer L)
            #There are x data points, and j neurons in the layer
            #C = -1 / len(train) * sum((sum(y_j * ln(a^L_j) + (1 - y_j) * ln(1 - a^L_j)), j), x)
    def delta(z, a, y):
        return (a - y)

#At some point we need to normalize our data and organize it into subsets

class NeuralNet(object):
    def __init__(self, sizes):
        #Initialize everything
        self.nLayers = len(sizes)
        self.sizes = sizes
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    def sigmoid(self, z):
        #Sigmoid neuron activation function
        return 1 / (1 + np.exp(-z))
    def sigmoidPrime(self, z):
        #Derivative of sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    def forward(self, a):
        #Move values forward through network
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    #def cost()
        #Cross-entropy cost function
        #C = -1 / len(train) * sum(y * ln(a) + (1 - y) * ln(1 - a), x) - one output
        #C = -1 / len(train) * sum(y * ln(y) + (1 - y) * ln(1 - y), x) - one thing
        #If y = y_1, y_2, ... are desired output values, a^L = a^L_1, a^L_2, ... are actual output values (in layer L)
            #There are x data points, and j neurons in the layer
            #C = -1 / len(train) * sum((sum(y_j * ln(a^L_j) + (1 - y_j) * ln(1 - a^L_j)), j), x)

            
    def costPrime(self, outActivations, y):
        #Derivative of cost function
        return (outActivations - y)
    #def evaluate(self, test)
        #return the average cost of all the test data
    def backprop(self, x, y):
        #Compute gradient of the cost function
        delw = [np.zeros(w.shape) for w in self.weights]
        delb = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.costPrime(activations[-1], y) * self.sigmoidPrime(zs[-1])
        delw[-1] = np.dot(delta, activations[-2].transpose())
        delb[-1] = delta
        for i in xrange(2, self.nLayers):
            z = zs[-i]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            delw[-i] = np.dot(delta, activations[-i - 1].transpose())
            delb = delta
        return (delw, delb)
    def updateMinBat(self, minBat, learnRate):
        #Do gradient descent using backprop to a single mini-batch
        delw = [np.zeros(w.shape) for w in self.weights]
        delb = [np.zeros(b.shape) for b in self.biases]
        for x, y in minBat:
            deltadelw, deltadelb = self.backprop(x, y)
            delw = [dw + ddw for dw, ddw in zip(delw, deltadelw)]
            delb = [db + ddb for db, ddb in zip(delb, deltadelb)]
        self.weights = [w - (learnRate / len(minBat)) * dw for w, dw in zip(self.weights, delw)]
        self.biases = [b - (learnRate / len(miniBat)) * db for b, db in zip(self.biases, delb)]
    def stochGradDescent(self, train, test = None, epochs, minBatSize, learnRate, lmda = 0.0, monitorTrainCost = False, monitorTrainAccuracy = False, monitorTestCost = False, monitorTestAccuracy = False):
        #Apply gradient descent
        n = len(train)
        if test: nTest = len(test)
        for i in xrange(epochs):
            random.shuffle(train)
            minBats = [train[j:j + minBatSize] for j in xrange(0, n, minBatSize)]
            for minBat in minBats:
                self.updateMinBat(minBat, learnRate)
            if test:
                print("Epoch {0}: average test cost of {1}".format(i, self.evaluate(test)))
            else:
                print("Epoch {0} complete".format(i))

