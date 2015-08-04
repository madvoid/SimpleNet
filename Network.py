"""
Filename: Network.py

Author: Nipun Gunawardena

Acknowledgements: Based off of Michael Nielsen's Neural Network and Deep Learning Tutorial code found at
                  https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py.
                  Also based off of info found at http://iamtrask.github.io/2015/07/12/basic-python-network/?

Requirements: Python 2.7.6 (Tested on)
              Numpy
              Matplotlib

Notes:  The purpose of this file is to help me understand neural networks on a programming level
"""



## Imports ----------------------------------------------------------------------------------------
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt







## Miscellaneous Functions ------------------------------------------------------------------------
def sigmoid(z):
    """
    The sigmoid function
    """
    return 1.0/(1.0 + np.exp(-z))


def sigmoidPrime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


def dataSplit(inputs, outputs, trainSplit = 0.70, testSplit = 0.15, valSplit = 0.15):
    """
    - Splits data into test, train, and validation data
    Original input data needs to be an m x n array where there are n variables and m samples
    - By default randomly splits the data, but block split needs to be added. 
    - The number from valSplit isn't actually used, it's just the leftovers after testSplit and trainSplit 
    are accounted for.
    - Hasn't been tested on any dataset other than house dataset yet. Multiple outputs may cause problems

    TODO: Add blockSplit
    """

    # Check correct split
    if testSplit + trainSplit + valSplit != 1.0:
        print("ERROR: Please enter splits that add up to 1!")
        sys.exit()

    # Check correct lengths
    if len(inputs) != len(outputs):
        print("ERROR: Please ensure inputs and outputs are same length!")
        sys.exit()

    dataLen = len(inputs)
    trainLen = int(math.ceil(trainSplit*dataLen))
    testLen = int(math.ceil(testSplit*dataLen))
    valLen = dataLen - trainLen - testLen

    shuffle = np.random.permutation(dataLen)
    trainIdx = shuffle[0:trainLen]
    testIdx = shuffle[trainLen:trainLen+testLen]
    valIdx = shuffle[trainLen+testLen:trainLen+testLen+valLen]

    inputsTrain = inputs[trainIdx, :]
    inputsTest = inputs[testIdx, :]
    inputsVal = inputs[valIdx, :]
    outputsTrain = outputs[trainIdx]
    outputsTest = outputs[testIdx]
    outputsVal = outputs[valIdx]

    if __debug__:
        print("{0} elements total, {1} elements towards training, {2} elements towards testing, and {3} elements towards validation\n".format(dataLen, trainLen, testLen, valLen))

    return np.array(inputsTrain).T, np.array(inputsTest).T, np.array(inputsVal).T, np.array(outputsTrain).T, np.array(outputsTest).T, np.array(outputsVal).T














## Vectorized Functions ---------------------------------------------------------------------------
sigmoidVec = np.vectorize(sigmoid)
sigmoidPrimeVec = np.vectorize(sigmoidPrime)







## Classes ----------------------------------------------------------------------------------------
class Network():
    """Neural Network Class"""

    def __init__(self, sizes):
        """
        Initialize neural network.
        'sizes' should be a list where each element is the size
        of that layer. For example, a network with 3 input nodes,
        4 hidden nodes, and 1 output node would be [3,4,2]
        """
        self.numLayers = len(sizes)
        self.biases = [np.random.rand(i, 1) for i in sizes[1:]]
        self.weights = [np.random.rand(i,j) for i,j in zip(sizes[1:],sizes[:-1])]

        if __debug__:
            print("Biases Sizes:\n {0}\n".format([b.shape for b in self.biases]))
            print("Weights Sizes:\n {0}\n".format([w.shape for w in self.weights]))


    def regFeedForward(self, a, actFunc):
        """
        Forward propogates an input vector "a" throughout
        the neural network, assuming a linear activation function on output layer. 
        Input vector "a" should be passed in as a *2d* numpy array, 
        even though it is only mathematically
        1d. (Use a.reshape(-1,1) to do so)
        """
        for l in xrange(self.numLayers-2):
            b = self.biases[l]
            w = self.weights[l]
            a = actFunc(np.dot(w,a)+b)

        # Last layer uses linear activation instead of sigmoidal
        b = self.biases[-1]
        w = self.weights[-1]
        a = np.dot(w,a) + b

        return a


    def train(self, trainInputs, trainOutputs, miniBatchSize, epochs = 10, eta = 0.3, lmbda = 0.001, valInputs = None, valOutputs = None):
        """
        Train neural network using training data. If valInputs & valOutputs are included,
        validation will be calculated as well. "miniBatchSize" should be an even factor
        of the number of elements in the training set. "eta" and "lmbda" are the learning rate
        and regularization value respectively.
        """

        # Get important sizes
        numVar, numSamples = trainInputs.shape
        if (valInputs is not None) and (valOutputs is not None):
            valSamples = valInputs.shape[1]

        # Train over epochs
        for i in xrange(epochs):
            print "Epoch {0} || ".format(i),

            # Create Mini-batches (array of arrays that correspond to each other)
            shuffle = np.random.permutation(numSamples)
            trainInputs = trainInputs[:,shuffle]
            trainOutputs = trainOutputs[:,shuffle]
            miniBatchInputs = [ trainInputs[:,k:k+miniBatchSize] for k in xrange(0,numSamples,miniBatchSize) ]
            miniBatchOutputs = [ trainOutputs[:,k:k+miniBatchSize] for k in xrange(0,numSamples,miniBatchSize) ]

            # Update weights using mini batches
            for miniBatch in zip(miniBatchInputs, miniBatchOutputs):
                self.sgdMiniBatch(miniBatch[0], miniBatch[1], eta, lmbda)

            # Check on validation data
            if (valInputs is not None) and (valOutputs is not None):
                mse = 0.0
                for v in xrange(valSamples):
                    mse += self.squaredError(self.regFeedForward(valInputs[:,v].reshape(-1,1), sigmoidVec).T, valOutputs[:,v])
                mse /= valSamples
                print "Val Err =", mse,

            # Finish Printing
            print " "

        return 0


    def sgdMiniBatch(self, inputs, outputs, eta, lmbda):
        """
        Performs SGD on a mini-batch. More info to come.
        """
        return 0


    def squaredError(self, estimate, actual):
        """
        Calculates the standard squared error between two scalars/vectors
        The two vectors passed in need to be the same size, and this needs to be done outside
        the function
        """
        return (estimate - actual)**2









## Main -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read Data
    # inputs = np.genfromtxt("houseInputs.csv",delimiter=",");
    # outputs = np.genfromtxt("houseTargets.csv",delimiter=",");
    inputs = np.genfromtxt("buildingInputs.csv",delimiter=",");
    outputs = np.genfromtxt("buildingTargets.csv",delimiter=",");

    # Initialize Neural Network
    # testNN = Network([3,4,2])
    # houseNN = Network([13,20,1])
    buildingNN = Network([14,20,3])

    # Split data
    inputsTrain, inputsTest, inputsVal, outputsTrain, outputsTest, outputsVal, = dataSplit(inputs, outputs)


    # Plot Data
    # plt.figure()
    # plt.plot(inputsTrain)
    # plt.figure()
    # plt.plot(inputsTest)
    # plt.figure()
    # plt.plot(inputsVal)
    # plt.figure()
    # plt.plot(inputs)
    # plt.figure()
    # plt.plot(outputsTrain)
    # plt.figure()
    # plt.plot(outputsTest)
    # plt.figure()
    # plt.plot(outputsVal)
    # plt.figure()
    # plt.plot(outputs)
    # plt.show()

    # Train Network
    buildingNN.train(inputsTrain, outputsTrain, 491, valInputs=inputsVal, valOutputs=outputsVal)
