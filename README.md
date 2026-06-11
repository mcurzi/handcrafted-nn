
# Handcrafted Neural Network from Scratch

A personal implementation of a fully connected neural network built from the ground up using only Python and NumPy, without relying on high-level frameworks like TensorFlow or Keras. The core logic was implemented manually as a learning exercise, with the goal of keeping the code clear and educational.

## Features

*   **Customizable Architecture:** Support for any number of hidden layers and neurons.
*   **Activation Functions:** Manual implementation of `sigmoid`, `tanh`, `ReLU` and `Leaky ReLU`.
*   **Bias Integration:** Bias units handled automatically via input augmentation.
*   **Weight Updates via Backpropagation:** Using stochastic gradient descent.
*   **Loss Function:** Mean Squared Error (MSE) with loss curve visualization using Matplotlib.

## How it Works

The network follows the standard forward and backward pass of a neural network:
1. **Linear Combination:** Each neuron computes a weighted sum of its inputs plus a bias term  
2. **Activation:** The result is passed through a non-linear function to introduce complexity into the model  
3. **Backpropagation:** The network computes gradients using the chain rule and updates weights accordingly

## Current Architecture

*    **5 layers**: 10 (input) - 32 (hidden) - 16 (hidden) - 8 (hidden) - 3 (output)
*    Inputs: 10 binary numbers (0/1)
*    Outputs: 3 continuous values (float), interpreted according to the target task
*    Activation functions (can vary per layer): **tanh**, **tanh**, **tanh**, **sigmoid**
*    Bias neurons included in each layer
*    This simple implementation supports only fully connected architectures

## Output layer after training with the example dataset:

*    Neuron 1: learns to approximate parity (whether the sum of the 10 inputs is odd)  
*    Neuron 2: learns to replicate the first input (identity)  
*    Neuron 3: learns to approximate the mean of bits 6 and 7  

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib (for cost visualization)

## Usage

To train the model with the provided dataset, **nn_training.csv**, run `python main.py`

There is also a complete data set included with all 1024 bit combinations, **nn_training_complete.csv**.

You can create your own data sets for your own learning objectives, to see how it works.

## Empirical Analysis

Initial experiments with shallow architectures (single hidden layer) showed strong performance on linear tasks (identity and averaging) but clear limitations when dealing with the parity problem (predicting whether the sum of inputs is even or odd). After increasing both network depth and width, the model was able to successfully learn all three targets simultaneously.

This is the actual output from the last run before updating this document:

```bash
$ python main.py
Starting training...
Epoch: 0, Loss: 0.227653
Epoch: 500, Loss: 0.080997
Epoch: 1000, Loss: 0.013778
Epoch: 1500, Loss: 0.003282
End of training. Epochs: 1999, Loss: 0.002737

--- Individual Testing ---
Test: [1. 1. 0. 0. 1. 0. 1. 0. 1. 1.] | Res: [0.  1.  0.5] | Pred: [0.0151, 0.9999, 0.5203]
Test: [0. 1. 1. 1. 0. 1. 1. 0. 0. 0.] | Res: [1. 0. 1.] | Pred: [0.9669, 0.0014, 0.9827]
Test: [1. 1. 1. 0. 0. 0. 0. 1. 0. 0.] | Res: [0. 1. 0.] | Pred: [0.0326, 0.9991, 0.0108]
Test: [0. 0. 0. 1. 1. 0. 1. 1. 1. 0.] | Res: [1.  0.  0.5] | Pred: [0.9738, 0.0024, 0.5145]
Test: [1. 0. 1. 0. 1. 1. 1. 1. 0. 0.] | Res: [0. 1. 1.] | Pred: [0.0220, 0.9957, 0.9855]
Test: [0. 1. 0. 1. 0. 0. 0. 0. 1. 1.] | Res: [0. 0. 0.] | Pred: [0.0054, 0.0005, 0.0049]
Test: [1. 0. 0. 1. 1. 1. 0. 1. 1. 0.] | Res: [0.  1.  0.5] | Pred: [0.0232, 0.9972, 0.5172]
Test: [0. 0. 1. 1. 0. 0. 1. 0. 0. 1.] | Res: [0.  0.  0.5] | Pred: [0.0081, 0.0003, 0.4991]
Test: [1. 1. 0. 1. 0. 1. 0. 1. 0. 1.] | Res: [0.  1.  0.5] | Pred: [0.0306, 0.9981, 0.5064]
Test: [0. 1. 1. 0. 1. 0. 0. 1. 1. 1.] | Res: [0. 0. 0.] | Pred: [0.0089, 0.0005, 0.0051]
```

### Learning curve 

![Larning Curve demo](nn_training_demo.png)
