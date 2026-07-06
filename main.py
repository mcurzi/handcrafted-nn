"""
Handcrafted Neural Network
Implemented from scratch using Python and NumPy, without high-level frameworks.

Matias J. Curzi
"""

from nn_class import *

# Initialize network: 10 inputs -> 32 (hidden) - 16 (hidden) - 8 (hidden) -> 3 outputs
# Mor than one inner layer is required for the parity problem, the other targets are easier
# Hidden layers use tanh. The output layer uses sigmoid to keep predictions bewteen 0 and 1.
nn = NeuralNetwork([10, 32, 16, 8, 3], activations=['tanh','tanh', 'tanh', 'sigmoid'])


# Load training dataset
# Ensure "nn_training.csv" exists in the same directory
try:
    training_set = np.loadtxt(open("nn_training.csv", "rb"), delimiter=",", skiprows=1)
    inputs = training_set[:, 0:10]
    targets = training_set[:, 10:13]

    print("Starting training...")
    nn.fit(inputs, targets, learning_rate=0.005, epochs=2000)  # Low learning rate (<=0.01) might help with the parity problem

    # Saves the model:
    nn.save("nn_test_model.npz")

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
        pred = nn.predict(sample)
        pred_str = "[" + ", ".join(f"{v:.4f}" for v in pred) + "]" # Converts pred to sting, 4 decimal places
        print(f"Test: {sample} | Res: {exp} | Pred: {pred_str}")

    # Cost Function Visualization
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

