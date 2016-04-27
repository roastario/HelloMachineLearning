import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        ##training data is a list of tuples which contains (x,y) -> x being input data, y being expected output
        n = len(training_data)  ##number of training inputs
        for j in xrange(epochs):  ## loop over training set
            random.shuffle(training_data)  ## shuffle the training data so that we get new inputs every loop
            mini_batches = []
            for k in xrange(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k + mini_batch_size])  ## take a slice of the training_data

            for mini_batch in mini_batches:
                ##here we have to find the shifts for our weights / bias vectors
                ##we will use the back propagation algorithm to do quick gradient calculations
                bias_prime = [np.zeros(b.shape) for b in self.biases]  ## init to a "matrix" of zeroes
                weights_prime = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:  ## x input data, y expected output
                    bias_deltas = [np.zeros(b.shape) for b in self.biases]
                    weight_deltas = [np.zeros(w.shape) for w in self.weights]

                    activation = x
                    activations = [x]  # list to store all the activations, layer by layer
                    zs = []  # list to store all the z vectors, layer by layer - z being the value of <activation>.<weights>
                    ## forward pass to calculate the output of the hidden layer.
                    for b, w in zip(self.biases, self.weights):
                        z = np.dot(w, activation)
                        zs.append(z)
                        activation = sigmoid(z)
                        activations.append(activation)

                    ##compute delta_w for all weights from hidden layer to output layer
                    error_vector = y - activations[-1]  ##the output of the network is final activations
                    delta = error_vector * sigmoid_prime(zs[-1])
                    bias_deltas[-1] = delta
                    weight_deltas[-1] = np.dot(delta, x.transpose())

                    ##in -> hid -> out
                    ##compute delta_w for all weights from input layer to hidden layer (only 3 layers, so will only loop once)
                    for l in xrange(2, self.num_layers):
                        z = zs[-l]
                        sp = sigmoid_prime(z)
                        delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                        bias_deltas[-l] = delta
                        weight_deltas[-l] = np.dot(delta, activations[-l - 1].transpose())

                    bias_prime = [nb + dnb for nb, dnb in zip(bias_prime, bias_deltas)]
                    weights_prime = [nw + dnw for nw, dnw in zip(weights_prime, weight_deltas)]

                self.weights = [w - (learning_rate / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, weights_prime)]
                self.biases = [b - (learning_rate / len(mini_batch)) * nb
                               for b, nb in zip(self.biases, bias_prime)]





    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
