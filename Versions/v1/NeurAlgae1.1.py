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

import json
import random
import sys
import time
import numpy as np
#For a network with inputs <a> through <c>, outputs <a> through <b>, and <n> data points
#Input data in the form [(np.array([[xa1], [xb1], [xc1]]), np.array([[ya1], [yb1]])), (np.array([[xa2], [xb2], [xc2]]), np.array([[ya2], [yb2]])), ... (np.array([[xan], [xbn], [xcn]]), np.array([[yan], [ybn]]))]
class NeuralNet(object):
    def __init__(self, sizes, cost = "crossEntropy"):
        #Initialize everything
        self.nLayers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.trainCost, self.testCost = [], []
        self.trainDelta, self.testDelta = [], []
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
    def costfn(self, a, y):
        #Implement cross-entropy cost function
        if self.cost == "meanSquareError":
            return 0.5 * np.linalg.norm(a - y) ** 2
        elif self.cost == "crossEntropy":
            return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    def deltaCost(self, z, a, y):
        #Simple difference between activation and expected output
        if self.cost == "meanSquareError":
            return (a - y) * self.sigmoidPrime(z)
        elif self.cost == "crossEntropy":
            return (a - y)
    def totalCost(self, data, lmda):
        #Sum the cost of passing a dataset forward
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            cost += self.costfn(a, y) / len(data)
        cost += 0.5 * (lmda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
    def totalDelta(self, data):
        #Sum the absolute error of passing a dataset forward
        delta = 0.0
        for x, y in data:
            a = self.forward(x)
            delta += (a - y)
        delta = delta / len(data)
        return delta
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
        delta = self.deltaCost(zs[-1], activations[-1], y)
        delw[-1] = np.dot(delta, np.transpose(activations[-2]))
        delb[-1] = delta
        for i in range(2, self.nLayers):
            z = zs[-i]
            sp = self.sigmoidPrime(z)
            delta = np.dot(np.transpose(self.weights[-i + 1]), delta) * sp
            delw[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
            delb[-i] = delta
        return (delw, delb)
    def updateMinBat(self, minBat, learnRate, lmda, n):
        #Do gradient descent using backprop to a single mini-batch
        delw = [np.zeros(w.shape) for w in self.weights]
        delb = [np.zeros(b.shape) for b in self.biases]
        for x, y in minBat:
            deltadelw, deltadelb = self.backprop(x, y)
            delw = [dw + ddw for dw, ddw in zip(delw, deltadelw)]
            delb = [db + ddb for db, ddb in zip(delb, deltadelb)]
        self.weights = [(1 - learnRate * (lmda / n)) * w - (learnRate / len(minBat)) * dw for w, dw in zip(self.weights, delw)] 
        self.biases = [b - (learnRate / len(minBat)) * db for b, db in zip(self.biases, delb)]
    def remindSGD(self):
        #Remind forgetful programmers how to train network
        print ("<network>.stochGradDescent(self, train, epochs, minBatSize, learnRate, lmda = 0.0, test = None, monitorTrain = False, monitorTest = False, monitorDelta = False, outputMeta = False, delay = 0.0)")
    def stochGradDescent(self, train, epochs, minBatSize, learnRate, lmda = 0.0, test = None, monitorTrain = False, monitorTest = False, monitorDelta = False, returnMeta = False, delay = 0.0):
        #Apply gradient descent
        n = len(train)
        if test: nTest = len(test)
        trainCost, testCost = [], []
        trainDelta, testDelta = [], []
        train2 = train
        if len(self.trainCost) == 0:
            if monitorTrain:
                cost = self.totalCost(train2, lmda)
                trainCost.append(cost)
                if monitorDelta:
                    delta = self.totalDelta(train2)
                    trainDelta.append(delta)
                    print("Initial error with training data: {}".format(delta))
                print("Initial cost with training data: {}".format(cost))
        if len(self.testCost) == 0:
            if monitorTest:
                cost = self.totalCost(test, lmda)
                testCost.append(cost)
                if monitorDelta:
                    delta = self.totalDelta(test)
                    testDelta.append(delta)
                    print("Initial error with testing data: {}".format(delta))
                print("Initial cost with testing data: {}".format(cost))
        for i in range(epochs):
            random.shuffle(train2)
            minBats = [train2[j:j + minBatSize] for j in range(0, n, minBatSize)]
            for minBat in minBats:
                self.updateMinBat(minBat, learnRate, lmda, len(train2))
            print("Epoch %s: training complete" % i)
            if monitorTrain:
                cost = self.totalCost(train2, lmda)
                trainCost.append(cost)
                if monitorDelta:
                    delta = self.totalDelta(train2)
                    trainDelta.append(delta)
                    print("Error with training data: {}".format(delta))
                print("Cost with training data: {}".format(cost))
            if monitorTest:
                cost = self.totalCost(test, lmda)
                testCost.append(cost)
                if monitorDelta:
                    delta = self.totalDelta(test)
                    testDelta.append(delta)
                    print("Error with testing data: {}".format(delta))
                print("Cost with testing data: {}".format(cost))
            print
            time.sleep(delay)
        self.trainCost += trainCost
        self.testCost += testCost
        self.trainDelta += trainDelta
        self.testDelta += testDelta
        if returnMeta:
            return trainCost, testCost, trainDelta, testDelta
    def save(self, filename):
        #Save a neural network to a file
        data = {"sizes": self.sizes,
                "cost": self.cost,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "trainCost": self.trainCost,
                "testCost": self.testCost,
                "trainDelta": [trd.tolist() for trd in self.trainDelta],
                "testDelta": [ted.tolist() for ted in self.testDelta]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
def load(filename):
    #Load a previously created neural network from a file
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = NeuralNet(data["sizes"], data["cost"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    net.trainCost = data["trainCost"]
    net.testCost = data["testCost"]
    net.trainDelta = [np.array(trd) for trd in data["trainDelta"]]
    net.testDelta = [np.array(ted) for ted in data["testDelta"]]
    return net
