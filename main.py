# A neural network consists of inputs, hidden nodes, and outputs.
# I must code inputs, which can be values of my choice.
# For this i will choose coordinates, X, Y, Z.
# And i must have an output, containing info about the prediction.
# For the start this will be wether coordinates are inside my chosen room
# Room:  3 < x < 10, 2 < y < 8, 6 < z < 12.
'''

x
  \
y -> nodes -> prediction
  /
z

'''
import numpy as np
import random


# Lets start with building a neural network manually


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neural_network():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None

        self.layers = []

        # how many of the layers its been through
        self.progress = 0

        # keeps track of current values (output from previous neurons, or start input)
        self.values = [self.x, self.y, self.z]

    def compute(self):
        # computes next values using the neurons on the current values

        curr_values = self.values
        next_values = []

        for neuron in self.layers[self.progress]:
            output = neuron.compute_value(curr_values)
            next_values.append(output)


class Neuron(Neural_network):
    def __init__(self):
        super.__init__()
        self.w = None
        self.b = None
        self.value = None

    def compute_value(self, values:list):
        # computes a value based on input values
        curr_value = 0

        for n in range(len(values)):
            curr_value += values[n] * self.w[n] * self.b

        curr_value = sigmoid(curr_value)

        return curr_value


def start_network():
    neural_network = Neural_network()

    x = random.randint(-20, 20)
    y = random.randint(-20, 20)
    z = random.randint(-20, 20)

    layers = 1
    neurons = 4

    for i in range(layers):
        # For simplicity each neuron is fed each input in this test
        # so we need one weight for each neuron
        layer = []
        for j in range(neurons):
            neuron = Neuron()

            weights = []

            for n in range(1,3):
                weights.append(random.random())

            neuron.w = weights
            neuron.b = random.random()

            layer.append(neuron)

        neural_network.layers.append(layer)
