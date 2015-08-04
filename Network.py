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
    Splits data into test, train, and validation data
    Original input data needs to be an m x n array where there are n variables and m samples
    By default randomly splits the data, but block split needs to be added. 
    Outputs should be test, train and validation data where each set is an n x j array where
    there are n variables and j samples
    TODO: Add blockSplit
    The number from valSplit isn't actually used, it's just the leftovers after testSplit and trainSplit 
    are accounted for.
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
            print("Biases:\n {0}\n".format(self.biases))
            print("Weights:\n {0}\n".format(self.weights))


    # def feedForward(self, a, actFunc):
    #     """
    #     Forward propogates an input vector "a" throughout
    #     the neural network. Input vector "a" should be passed
    #     in as a *2d* numpy array, even though it is only mathematically
    #     1d
    #     """
    #     for l in xrange(self.numLayers-1):
    #         b = self.biases[l]
    #         w = self.weights[l]
    #         a = actFunc(np.dot(w,a)+b)

    #     return a


    def train(self, trainInputs, trainOutputs, miniBatchSize, epochs = 100, eta = 0.3, lmbda = 0.001, valInputs = None, valOutputs = None):
        """
        Train neural network using training data. If valInputs & valOutputs are included,
        validation will be calculated as well. "miniBatchSize" should be an even factor
        of the number of elements in the training set. "eta" and "lmbda" are the learning rate
        and regularization value respectively.
        """
        for i in xrange(epochs):
            # Create Mini-batches
            # Update weights using mini batches
            # Check on validation data
            continue
        return 0







## Main -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read Data
    inputs = np.genfromtxt("houseInputs.csv",delimiter=",");
    outputs = np.genfromtxt("houseTargets.csv",delimiter=",");

    # Initialize Neural Network
    # houseNN = Network([13,20,1])
    testNN = Network([3,4,2])

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
