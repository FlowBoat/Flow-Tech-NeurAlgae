import json
import random
import sys
import numpy as np
import scipy as sp
import tensorflow as tf
class CrossEntropy(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    @staticmethod
    def delta(z, a, y):
        return (a - y)
class NeuralNet(object):
    def __init__(self, sizes):
        #Initialize everything
        self.nLayers = len(sizes)
        self.sizes = sizes
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
        delta = CrossEntropy.delta(zs[-1], activations[-1], y)
        delw[-1] = np.dot(delta, np.transpose(activations[-2]))
        delb[-1] = delta
        for i in range(2, self.nLayers):
            z = zs[-i]
            sp = self.sigmoidPrime(z)
            print(self.weights)
            print(i)
            print(self.weights[0])
            print(self.weights[1])
            print(self.weights[i])
            delta = np.dot(np.transpose(self.weights[i + 1]), delta) * sp
            delw[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
            delb = delta
        return (delw, delb)
    def updateMinBat(self, minBat, learnRate, lmda, n):
        #Do gradient descent using backprop to a single mini-batch
        delw = [np.zeros(w.shape) for w in self.weights]
        delb = [np.zeros(b.shape) for b in self.biases]
        for x, y in minBat:
            deltadelw, deltadelb = self.backprop(x, y)
            delw = [dw + ddw for dw, ddw in zip(delw, deltadelw)]
            delb = [db + ddb for db, ddb in zip(delb, deltadelb)]
        self.weights = [(1 - learnRate * (lmda / n)) * w - (learnRate / len(minBat)) * dw for w, ddw in zip(self.weights, delw)] 
        self.biases = [b - (learnRate / len(miniBat)) * db for b, db in zip(self.biases, delb)]
    def totalCost(self, data, lmda):
        #Sum the cost of passing a dataset forward
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
    def stochGradDescent(self, train, epochs, minBatSize, learnRate, lmda = 0.0, test = None, monitorTrain = False, monitorTest = False):
        #Apply gradient descent
        n = len(train)
        if test: nTest = len(test)
        trainCost, testCost = [], []
        for i in range(epochs):
            random.shuffle(train)
            minBats = [train[j:j + minBatSize] for j in range(0, n, minBatSize)]
            for minBat in minBats:
                self.updateMinBat(minBat, learnRate, lmda, len(train))
            print("Epoch %s training complete" % i)
            if monitorTrain:
                cost = self.totalCost(train, lmda)
                trainCost.append(cost)
                print("Cost with training data: {}".format(cost))
            if monitorTest:
                cost = self.totalCost(test, lmda)
                testCost.append(cost)
                print("Cost with testing data: {}".format(cost))
            print
        return trainCost, trainAccuracy, testCost, testAccuracy
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
