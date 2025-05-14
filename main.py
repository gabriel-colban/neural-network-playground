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

        # Randomly selected coordinates
        self.x = random.randint(-20, 20)
        self.y = random.randint(-20, 20)
        self.z = random.randint(-20, 20)

        self.layers = []

        # how many of the layers its been through
        self.progress = 0

        # keeps track of current values (output from previous neurons, or start input)
        self.values = [self.x, self.y, self.z]

    def compute(self):
        # computes next values using the neurons on the current values

        curr_values = self.values
        next_values = []

        if self.progress < len(self.layers):
            for neuron in self.layers[self.progress]:
                output = neuron.compute_value(curr_values)
                next_values.append(output)
        elif self.progress == len(self.layers):
            final_value = 0
            for n in curr_values:
                final_value += n
            final_value = sigmoid(final_value)
            next_values.append(final_value)
            print("Layers completed")
        else:
            next_values = self.values
            print("Layers completed")

        self.values = next_values
        self.progress += 1


class Neuron():
    def __init__(self, num_inputs):
        self.w = np.random.rand(num_inputs)
        self.b = random.random()

    def compute_value(self, values:list):
        # computes a value based on input values
        curr_value = 0
        print(values)
        for n in range(len(values)):
            curr_value += values[n] * self.w[n] * self.b

        curr_value = sigmoid(curr_value)

        return curr_value


def is_inside_room(x,y,z):


    # Room:  3 < x < 10, 2 < y < 8, 6 < z < 12.

    return x > 3 and x < 10 and y > 2 and y < 8 and z > 6 and z < 12

def start_network():
    neural_network = Neural_network()

    layers = 1
    neurons = 4

    for i in range(layers):
        # For simplicity each neuron is fed each input in this test
        # so we need one weight for each neuron
        layer = []
        for j in range(neurons):
            if i == 0:
                # If its the first input neuron takes as many inputs as the network has
                num_inputs = len(neural_network.values)
            else:
                # If its deeper, neuron takes as many inputs as there are neurons before it.
                num_inputs = len(neural_network.layers[i-1])
            neuron = Neuron(num_inputs)

            layer.append(neuron)

        neural_network.layers.append(layer)

    # run through layers

    for x in range(len(neural_network.layers)+1):
        neural_network.compute()
    print(f"Coordinates: {neural_network.x}, {neural_network.y}, {neural_network.z}")
    print("Final value: ", neural_network.values[0])
    prediction = neural_network.values[0] > 0.5
    if prediction:
        print("Network predicted room is inside")
    else:
        print("Network predicted room is outside")
    is_inside = is_inside_room(neural_network.x,neural_network.y,neural_network.z)
    if is_inside:
        print("Random coordinates is inside room")
    else:
        print("Random coordinates is outside room")

start_network()