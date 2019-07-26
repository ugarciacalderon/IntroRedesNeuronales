from Perceptron.Perceptron import *

if __name__ == '__main__':

    inputs = [
        [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ]

    outputs = [0,0,0,1]

    perceptron = Perceptron(inputs, outputs)
    perceptron.Fit()