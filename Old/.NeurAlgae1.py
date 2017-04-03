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
    '''WE DO NOT NEED TO VECTORIZE OUR RESULTS of course
    def vectorResult(self, j):
        e = np.zeroes((10, 1))
        e[j] = 1.0
        return e'''
    def forward(self, a):
        #Move values forward through network
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    '''def cost()
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
        #return the average cost of all the test data'''
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
        '''delta = self.costPrime(activations[-1], y) * self.sigmoidPrime(zs[-1])'''
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        delw[-1] = np.dot(delta, activations[-2].transpose())
        delb[-1] = delta
        for i in xrange(2, self.nLayers):
            z = zs[-i]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            delw[-i] = np.dot(delta, activations[-i - 1].transpose())
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
        '''self.weights = [w - (learnRate / len(minBat)) * dw for w, dw in zip(self.weights, delw)]'''
        self.weights = [(1 - learnRate * (lmda / n)) * w - (learnRate / len(minBat)) * dw for w, ddw in zip(self.weights, delw)] 
        self.biases = [b - (learnRate / len(miniBat)) * db for b, db in zip(self.biases, delb)]
    '''def accuracy(self, data, convert = False):
        if convert:
            results = [(np.argmax(self.forward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.foward(x)), y) for x, y in data]
        return sum(int(x == y) for x, y in results)'''
    def totalCost(self, data, lmda, convert = False):
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            '''if convert: y = self.vectorResult(y)'''
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
    '''def stochGradDescent(self, train, test = None, epochs, minBatSize, learnRate, lmda = 0.0, monitorTrainCost = False, monitorTrainAccuracy = False, monitorTestCost = False, monitorTestAccuracy = False):'''
    def stochGradDescent(self, train, test = None, epochs, minBatSize, learnRate, lmda = 0.0, monitorTrain = False, monitorTest = False):
        #Apply gradient descent
        n = len(train)
        if test: nTest = len(test)
        '''trainAccuracy, trainCost = [], []
        testAccuracy, testCost = [], []'''
        trainCost, testCost = [], []
        '''#Notes
            #We do not need accuracy!
            #Finish making all this!
            #Make sure that this works with our thing - check how the y-hat-values will come out!
            #Generally make this a regression thing
            #Figure out how it wants input and output data so we can format ours that way'''
        for i in xrange(epochs):
            random.shuffle(train)
            minBats = [train[j:j + minBatSize] for j in xrange(0, n, minBatSize)]
            for minBat in minBats:
                self.updateMinBat(minBat, learnRate, lmda, len(train))
            print("Epoch %s training complete" % i)
            '''if monitorTrainAccuracy:
                accuracy = self.accuracy(train, convert = True)
                trainAccuracy.append(accuracy)
                print("Accuracy with training data: {} / {}".format(accuracy, n))
            if monitorTrainCost:
                cost = self.totalCost(train, lmda)
                trainCost.append(cost)
                print("Cost with training data: {}".format(cost))
            if monitorTestAccuracy:
                accuracy = self.accuracy(test)
                testAccuracy.append(accuracy)
                print("Accuracy with testing data: {} / {}".format(accuracy, nTest))
            if monitorTestCost:
                cost = self.totalCost(test, lmda, convert = True)
                testCost.append(cost)
                print("Cost with testing data: {}".format(cost))'''
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
