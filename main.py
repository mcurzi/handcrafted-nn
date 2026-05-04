"""
Handcrafted Neural Network
Implemented from scratch using Python and NumPy, without high-level frameworks.

Description:
A simple fully connected neural network that processes 10 input features
through a hidden layer of 8 neurons to produce 3 outputs.
Backpropagation and weight updates are implemented manually.

Author: Matias Curzi
"""

import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    # x is already tanh(x)
    return 1.0 - x**2

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivative

        # Weights initialization
        self.weights = []
        self.deltas = []

        # Initialize random weights between -1 and 1
        for i in range(1, len(layers) - 1):
            # Assign random weights to Input layer and Hidden layer  (+1 for Bias unit)
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)

        # Output layer weights
        r = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # Trains the network using Stochastic Gradient Descent.

        # Add a column of ones to X for the Bias unit
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            # Stochastic Gradient Descent (single sample update)
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # Forward propagation
            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)

            # Calculate error and delta at the output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # Backward propagation to calculate deltas for hidden layers
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            self.deltas.append(deltas)
            deltas.reverse()

            # Weight update (Gradient Descent)
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0:
                print(f'Epoch: {k}')

    def predict(self, x):
        # Predicts the output for a given input array.
        # Add bias to input
        a = np.concatenate(([1.0], np.array(x)))
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def get_deltas(self):
        return self.deltas

### Execution Block 
if __name__ == "__main__":
    # Initialize network: 10 inputs -> 8 hidden neurons -> 3 outputs
    nn = NeuralNetwork([10, 8, 3], activation='tanh')

    # Load training dataset
    # Ensure "nn_training.csv" exists in the same directory
    try:
        training_set = np.loadtxt(open("nn_training.csv", "rb"), delimiter=",", skiprows=1)
        inputs = training_set[:, 0:10]
        targets = training_set[:, 10:13]

        print("Starting training...")
        nn.fit(inputs, targets, learning_rate=0.03, epochs=10000)

        # Batch predictions for verification
        print("\n--- Training Set Predictions ---")
        for i, e in enumerate(inputs):
            print(f"X: {e} | Target: {targets[i]} | Predicted: {nn.predict(e)}")

        # Individual predictions for unknown data
        print("\n--- Individual Testing ---")
        test_samples = [
            np.array([1., 0., 1., 1., 0., 1., 0., 0., 1., 0.]), # Expected: [1. 1. 0.5]
            np.array([0., 0., 1., 0., 1., 1., 1., 0., 1., 0.]), # Expected: [1. 0. 1.]
            np.array([1., 0., 0., 0., 1., 1., 0., 0., 0., 1.])  # Expected: [0. 1. 0.5]
        ]

        for sample in test_samples:
            print(f"Input: {sample} | Prediction: {nn.predict(sample)}")

        # --- Cost Function Visualization ---
        deltas = nn.get_deltas()
        error_values = []

        for d in deltas:
            # Summing deltas of the output layer as a proxy for error magnitude
            error_values.append(abs(d[1][0]) + abs(d[1][1]))

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(error_values)), error_values, color='b')
        plt.ylim([0, 1])
        plt.ylabel('Error Magnitude')
        plt.xlabel('Epochs')
        plt.title('Training Convergence')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: 'nn_training.csv' not found. Please include the dataset.")
