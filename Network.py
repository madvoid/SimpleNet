"""
Filename: Network.py

Author: Nipun Gunawardena

Acknowledgements: Based off of Michael Nielsen's Neural Network and Deep Learning Tutorial code found at
                  https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py.
                  Also based off of info found at http://iamtrask.github.io/2015/07/12/basic-python-network/?

Requirements: Python 2.7.6 (Tested on)
              Numpy
              Matplotlib

Notes:  - The purpose of this file is to help me understand neural networks on a programming level
        - This code is specifically made to perform neural network regressions, not classifications. It won't
          work for classifications

TODO:   - Make reshape function that will reshape any vector to vertical 1-d vector
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


def dataSplit(inputs, targets, trainSplit = 0.70, testSplit = 0.15, valSplit = 0.15):
    """
    - Splits data into test, train, and validation data
    Original input data needs to be an m x n array where there are n variables and m samples
    - By default randomly splits the data, but block split needs to be added. 
    - The number from valSplit isn't actually used, it's just the leftovers after testSplit and trainSplit 
    are accounted for.
    - Hasn't been tested on any dataset other than house dataset yet. Multiple targets may cause problems

    TODO: Add blockSplit
    """

    # Check correct split
    if testSplit + trainSplit + valSplit != 1.0:
        print("ERROR: Please enter splits that add up to 1!")
        sys.exit()

    # Check correct lengths
    if len(inputs) != len(targets):
        print("ERROR: Please ensure inputs and targets are same length!")
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
    targetsTrain = targets[trainIdx]
    targetsTest = targets[testIdx]
    targetsVal = targets[valIdx]

    if __debug__:
        print("{0} elements total, {1} elements towards training, {2} elements towards testing, and {3} elements towards validation\n".format(dataLen, trainLen, testLen, valLen))

    return np.array(inputsTrain).T, np.array(inputsTest).T, np.array(inputsVal).T, np.array(targetsTrain).T, np.array(targetsTest).T, np.array(targetsVal).T














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
        4 hidden nodes, and 1 output node would be [3,4,1]
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


    def evaluateMSE(self, evInputs, evTargets):
        """
        Evaluate the network error with data. Mean squared error
        is the metric used. Multiple outputs will be summed into
        1 number
        """
        evSamples = evInputs.shape[1]
        mse = 0.0
        for t in xrange(evSamples):
            mse += self.squaredError(self.regFeedForward(evInputs[:,t].reshape(-1,1), sigmoidVec).T, evTargets[:,t])
        mse /= evSamples
        return np.sum(mse)


    def train(self, trainInputs, trainTargets, miniBatchSize, epochs = 50, eta = 0.3, lmbda = 0.001, valInputs = None, valTargets = None):
        """
        Train neural network using training data. If valInputs & valTargets are included,
        validation will be calculated as well. "miniBatchSize" should be an even factor
        of the number of elements in the training set. "eta" and "lmbda" are the learning rate
        and regularization value respectively.
        """

        # Get important sizes
        numVar, numSamples = trainInputs.shape
        if (valInputs is not None) and (valTargets is not None):
            valSamples = valInputs.shape[1]
            valCounter = 0
            mseEps = 0.001
            mseOld = 0.0

        # Train over epochs
        for i in xrange(epochs):
            print "Epoch {0} || ".format(i),

            # Create Mini-batches (array of arrays that correspond to each other)
            shuffle = np.random.permutation(numSamples)
            trainInputs = trainInputs[:,shuffle]
            trainTargets = trainTargets[:,shuffle]
            miniBatchInputs = [ trainInputs[:,k:k+miniBatchSize] for k in xrange(0,numSamples,miniBatchSize) ]
            miniBatchTargets = [ trainTargets[:,k:k+miniBatchSize] for k in xrange(0,numSamples,miniBatchSize) ]

            # Update weights using mini batches
            for miniBatch in zip(miniBatchInputs, miniBatchTargets):
                self.sgdMiniBatch(miniBatch[0], miniBatch[1], eta, lmbda)

            # Check on validation data
            if (valInputs is not None) and (valTargets is not None):

                # Calculate MSE
                mse = self.evaluateMSE(valInputs, valTargets)     

                # Check for Validation increasing accuracy
                if abs(mse - mseOld) < mseEps:
                    valCounter += 1
                else:
                    valCounter = 0
                mseOld = mse

                # Print and (maybe) break
                print "Val Mse =", mse, "|| Val Fail Count =", valCounter,
                if valCounter >= 5:
                    print " "
                    break

            # Finish Printing
            print " "

        return 0


    def sgdMiniBatch(self, inputs, targets, eta, lmbda):
        """
        Performs SGD on a mini-batch. This function is almost identical to Michael Nielson's
        The actual back-propagation is done in another function, while this handles the
        SGD
        """
        numSamples = inputs.shape[1]
        gradB = [np.zeros(b.shape) for b in self.biases]
        gradW = [np.zeros(w.shape) for w in self.weights]

        # Calculate the gradient of the weights and biases to be used in SGD
        for i in xrange(numSamples):
            deltaGradB, deltaGradW = self.backprop(inputs[:,i], targets[:,i])
            gradB = [nb + dnb for nb, dnb in zip(gradB, deltaGradB)]
            gradW = [nw + dnw for nw, dnw in zip(gradW, deltaGradW)]

        # Do gradient descent update step
        self.weights = [w - (eta/numSamples)*nw for w, nw in zip(self.weights, gradW)]
        self.biases = [b - (eta/numSamples)*nb for b, nb in zip(self.biases, gradB)]

        return 0


    def squaredError(self, estimate, actual):
        """
        Calculates the standard squared error between two scalars/vectors
        The two vectors passed in need to be the same size, and this needs to be done outside
        the function
        """
        return (estimate - actual)**2


    def backprop(self, inputVec, targetVec):
        """
        Return gradient for single example for cost function.
        Called by sgd function
        return values are layer-by-layer lists of arrays, similar to
        self.biases and self.weights
        This function is also similar to Nielson's, but modified for the 
        linear output activation function
        """
        inputVec = inputVec.reshape(-1,1)       # Reshape inputVec into vertical vector so math works
        targetVec = targetVec.reshape(-1,1)     # Same with targetVec

        # Initialize gradient
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        # Feedforward and save intermediate values
        a = inputVec
        acts = [inputVec]    # List to store activations for each layer
        zs = []                     # List to store weighted inputs for each layer
        for i in xrange(self.numLayers - 2):        # -2 because weights, biases have 1 less than numLayers, then avoid last layer
            b = self.biases[i]
            w = self.weights[i]
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoidVec(z)
            acts.append(a)

        # Above, but for last layer
        b = self.biases[-1]
        w = self.weights[-1]
        z = np.dot(w, a) + b
        zs.append(z)
        a = z       # Apply linear activation func. instead of sigmoidal
        acts.append(a)

        # Backward Pass last layer
        delta = self.costDerivative(acts[-1], targetVec) * 1        # BP1 - You multiply by one in place of sigmoidPrime because linear act. func. in output layer
        nablaB[-1] = delta                                          # BP3
        nablaW[-1] = np.dot(delta, acts[-2].T)                      # BP4

        # Backward pass rest of layers
        for l in xrange(2, self.numLayers):
            z = zs[-l]
            spv = sigmoidPrimeVec(z)
            delta = np.dot(self.weights[-l+1].T, delta) * spv       # BP2
            nablaB[-l] = delta                                      # BP3
            nablaW[-l] = np.dot(delta, acts[-l-1].T)

        return nablaB, nablaW


    def costDerivative(self, outputActivations, target):
        """
        Return vector of partial derivatives of cost function
        for output activations
        """
        return (outputActivations - target)


    def evaluate(self, evInputs, evTargets):
        """
        Return vector of estimates for any given inputs.
        Will be same size as the targets also supplied
        """
        outputs = np.zeros(evTargets.shape)
        evSamples = evInputs.shape[1]
        mse = 0.0
        for t in xrange(evSamples):
            outputs[:,t] = self.regFeedForward(evInputs[:,t].reshape(-1,1), sigmoidVec).T
            mse += self.squaredError(outputs[:,t], evTargets[:,t])
        mse /= evSamples
        return np.sum(mse), outputs








## Main -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Read Data
    prefix = "building"     # or "house"
    inputs = np.genfromtxt(prefix + "Inputs.csv",delimiter=",");
    targets = np.genfromtxt(prefix + "Targets.csv",delimiter=",");
    

    # Initialize Neural Network - Uncomment one
    # NN = Network([3,4,2])         # Test
    # NN = Network([13,20,1])       # House
    NN = Network([14, 11, 20, 3])       # Building
    

    # Split data
    inputsTrain, inputsTest, inputsVal, targetsTrain, targetsTest, targetsVal, = dataSplit(inputs, targets)


    # Plot Original Data to verify split worked - Optional
    # plt.figure()
    # plt.plot(inputsTrain.T)
    # plt.figure()
    # plt.plot(inputsTest.T)
    # plt.figure()
    # plt.plot(inputsVal.T)
    # plt.figure()
    # plt.plot(inputs.T)
    # plt.figure()
    # plt.plot(targetsTrain.T)
    # plt.figure()
    # plt.plot(targetsTest.T)
    # plt.figure()
    # plt.plot(targetsVal.T)
    # plt.figure()
    # plt.plot(targets.T)


    # Train Network
    NN.train(inputsTrain, targetsTrain, 491, valInputs=inputsVal, valTargets=targetsVal)

    # Test Network
    MSEtest, outputsTest = NN.evaluate(inputsTest, targetsTest)
    print "Test MSE =", MSEtest

    # Plot Test Output
    # plt.figure()
    # plt.plot(targetsTest.T)
    # plt.plot(outputsTest.T)
    # plt.show()

    np.savetxt("out.csv", outputsTest.T, delimiter=",")
