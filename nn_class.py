# Handcrafted Neural Network class and activation functions
# Matias J. Curzi

import numpy as np
import matplotlib.pyplot as plt


# Activation Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    # x is already sigmoid(x)
    return x * (1.0 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    # x is already tanh(x)
    return 1.0 - x**2
    
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # x is already relu(x)
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.01):  # Negative is not zero as ReLU, it "leaks" a small value (alpha * x)
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    # x is already leaky_relu(x)
    return np.where(x > 0, 1.0, alpha)


def linear(x):  # Linear (identity)
    return x

def linear_derivative(x):
    return np.ones_like(x)

def get_activation_pair(name):
    if name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif name == 'tanh':
        return tanh, tanh_derivative
    elif name == 'relu':
        return relu, relu_derivative
    elif name == 'leaky_relu':
        return leaky_relu, leaky_relu_derivative
    elif name == 'linear':
        return linear, linear_derivative
    else:
        raise ValueError(f"Unknown activation function: {name}")


class NeuralNetwork:
    def __init__(self, layers, activations):
        if len(activations) != len(layers) - 1:
            raise ValueError(
                f"Number of activations ({len(activations)}) must match "
                f"number of weight layers ({len(layers) - 1})."
            )

        self.layers = layers
        self.activation_names = activations
        self.activations = []
        self.activation_primes = []
        self.weights = []
        self.losses = []

        for name in activations:
            act, act_prime = get_activation_pair(name)
            self.activations.append(act)
            self.activation_primes.append(act_prime)

        # Weights initialization
        for i in range(len(layers) - 1):
            n_in = layers[i] + 1   # +1 due to bias
            n_out = layers[i + 1]
            activation = activations[i]

            if activation in ['sigmoid', 'tanh', 'linear']:
                limit = np.sqrt(6 / (n_in + n_out))   # Xavier uniform
                w = np.random.uniform(-limit, limit, (n_in, n_out))

            elif activation in ['relu', 'leaky_relu']:
                std = np.sqrt(2 / n_in)               # He normal
                w = np.random.randn(n_in, n_out) * std

            else:
                raise ValueError(f"Unknown activation function: {activation}")

            self.weights.append(w)


    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # Trains the network using Stochastic Gradient Descent over full epochs

        for epoch in range(epochs):
            # Shuffle dataset indices at the start of each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            epoch_loss = 0.0

            for idx in indices:
                # Add bias to current input sample
                x = np.concatenate(([1.0], X[idx]))
                a = [x]

                # Forward propagation
                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activations[l](dot_value)

                    # Add bias to hidden layers, but not to output layer
                    if l < len(self.weights) - 1:
                        activation = np.concatenate(([1.0], activation))

                    a.append(activation)

                # Error and loss
                error = y[idx] - a[-1]
                loss = np.mean(error**2)
                epoch_loss += loss

                # Output delta
                deltas = [error * self.activation_primes[-1](a[-1])]

                # Backpropagation for hidden layers
                # Take the last computed delta, which belongs to the next layer, and multiply
                # it by the transpose of the current weight matrix. This sends the error
                # backward and distributes it across the neurons of this layer according to
                # their connections. The result shows how much each neuron contributes to the
                # final error, but it still does not include the local effect of the activation
                # function. Then remove the bias component, because it is not a real neuron, and
                # multiply elementwise by the derivative of this layer's activation. This
                # applies the chain rule and turns the propagated error into the proper delta
                # for the current hidden layer.

                for l in range(len(a) - 2, 0, -1):
                    delta = deltas[-1].dot(self.weights[l].T)
                    delta = delta[1:] * self.activation_primes[l-1](a[l][1:])
                    deltas.append(delta)

                deltas.reverse()

                # Weight update
                for l in range(len(self.weights)):
                    layer = np.atleast_2d(a[l])
                    delta = np.atleast_2d(deltas[l])
                    self.weights[l] += learning_rate * layer.T.dot(delta)

            # Save average loss for the epoch
            self.losses.append(epoch_loss / X.shape[0])

            if epoch % 500 == 0:
                print(f"Epoch: {epoch}, Loss: {self.losses[-1]:.6f}")
        print(f"End of training. Epochs: {epoch}, Loss: {self.losses[-1]:.6f}")

    def predict(self, x):
        # Predicts the output for a given input array.
        # Add bias to input
        a = np.concatenate(([1.0], np.array(x)))
        
        for l in range(0, len(self.weights)):
            a = self.activations[l](np.dot(a, self.weights[l]))
            
            # Add bias to hidden layers only
            if l < len(self.weights) - 1:
                a = np.concatenate(([1.0], a))

        return a

    def get_losses(self):
        return self.losses
        
    def save(self, filename):
        np.savez(
            filename,
            layers=np.array(self.layers),
            activations=np.array(self.activation_names),
            weights=np.array(self.weights, dtype=object) # dtype=object creates an array where each element is a Python object
        )                                               # This is because each layer can have matrices of different size

# Function to load nn from a saved model
def load_model(filename):  
    data = np.load(filename, allow_pickle=True) # allow_pickle allows to load arrays with serialized Python objects (

    layers = data["layers"].tolist()
    activations = data["activations"].tolist()

    nn = NeuralNetwork(layers, activations)
    nn.weights = data["weights"].tolist()

    return nn


