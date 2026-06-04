"""
Handcrafted Neural Network
Implemented from scratch using Python and NumPy, without high-level frameworks.

Matias J. Curzi
"""

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
        self.losses = []

        # Initialize random weights between -1 and 1
        for i in range(len(layers) - 1):
            r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
            self.weights.append(r)

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
                    activation = self.activation(dot_value)

                    # Add bias to hidden layers, but NOT to output layer
                    if l < len(self.weights) - 1:
                        activation = np.concatenate(([1.0], activation))

                    a.append(activation)

                # Error and loss
                error = y[idx] - a[-1]
                loss = np.mean(error**2)
                epoch_loss += loss

                # Output delta
                deltas = [error * self.activation_prime(a[-1])]

                # Backpropagation for hidden layers
                for l in range(len(a) - 2, 0, -1):
                    delta = deltas[-1].dot(self.weights[l].T)
                    delta = delta[1:] * self.activation_prime(a[l][1:])
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

    def predict(self, x):
        # Predicts the output for a given input array.
        # Add bias to input
        a = np.concatenate(([1.0], np.array(x)))
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
            
            # Add bias to hidden layers only
            if l < len(self.weights) - 1:
                a = np.concatenate(([1.0], a))

        return a

    def get_losses(self):
        return self.losses


### Execution Block 
if __name__ == "__main__":
    # Initialize network: 10 inputs -> 32 (hidden) - 16 (hidden) - 8 (hidden) -> 3 outputs
    nn = NeuralNetwork([10, 32, 16, 8, 3], activation='tanh')  # Several inner layers are required for the parity problem, the others are easier
    # nn = NeuralNetwork([10, 12, 1], activation='tanh') # To solve only the parity problem, for testing purposes

    # Load training dataset
    # Ensure "nn_training.csv" exists in the same directory
    try:
        training_set = np.loadtxt(open("nn_training.csv", "rb"), delimiter=",", skiprows=1)
        inputs = training_set[:, 0:10]
        targets = training_set[:, 10:13]
        #targets = training_set[:, 10:11]   # Focused only on the parrity problem result, just for testing

        print("Starting training...")
        nn.fit(inputs, targets, learning_rate=0.005, epochs=2000)  # Low learning rate (<=0.01) might help woth the parity problem

        # Batch predictions for verification
        #print("\n--- Training Set Predictions ---")
        #for i, e in enumerate(inputs):
        #    print(f"X: {e} | Target: {targets[i]} | Predicted: {nn.predict(e)}")

        # Individual predictions for unknown data
        print("\n--- Individual Testing ---")
        test_samples = [
            np.array([1., 1., 0., 0., 1., 0., 1., 0., 1., 1.]),  # Expected: [0., 1., 0.5]
            np.array([0., 1., 1., 1., 0., 1., 1., 0., 0., 0.]),  # Expected: [1., 0., 1.0]
            np.array([1., 1., 1., 0., 0., 0., 0., 1., 0., 0.]),  # Expected: [0., 1., 0.0]
            np.array([0., 0., 0., 1., 1., 0., 1., 1., 1., 0.]),  # Expected: [1., 0., 0.5]
            np.array([1., 0., 1., 0., 1., 1., 1., 1., 0., 0.]),  # Expected: [0., 1., 1.0]
            np.array([0., 1., 0., 1., 0., 0., 0., 0., 1., 1.]),  # Expected: [0., 0., 0.0]
            np.array([1., 0., 0., 1., 1., 1., 0., 1., 1., 0.]),  # Expected: [0., 1., 0.5]
            np.array([0., 0., 1., 1., 0., 0., 1., 0., 0., 1.]),  # Expected: [0., 0., 0.5]
            np.array([1., 1., 0., 1., 0., 1., 0., 1., 0., 1.]),  # Expected: [0., 1., 0.5]
            np.array([0., 1., 1., 0., 1., 0., 0., 1., 1., 1.])   # Expected: [0., 0., 0.0]
        ]

        expected = [
            np.array([0., 1., 0.5]),
            np.array([1., 0., 1.0]),
            np.array([0., 1., 0.0]),
            np.array([1., 0., 0.5]),
            np.array([0., 1., 1.0]),
            np.array([0., 0., 0.0]),
            np.array([0., 1., 0.5]),
            np.array([0., 0., 0.5]),
            np.array([0., 1., 0.5]),
            np.array([0., 0., 0.0])
        ]

        for sample, exp in zip(test_samples, expected):
            print(f"Test: {sample} | Res: {exp} | Pred: {nn.predict(sample)}")

        # --- Cost Function Visualization ---
        losses = nn.get_losses()

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(losses)), losses, color='b')
        plt.ylabel('MSE Loss')
        plt.xlabel('Epochs')
        plt.title('Training Convergence')
        plt.tight_layout()
        plt.show()


    except FileNotFoundError:
        print("Error: 'nn_training.csv' not found. Please include the dataset.")
