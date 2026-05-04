# Handcrafted Neural Network from Scratch

A personal implementation of a fully connected neural network built from the ground up using only Python and NumPy, without relying on high-level frameworks like TensorFlow or Keras. The core logic was implemented manually as a learning exercise, with the goal of keeping the code clear and educational.

## 🚀 Features

*   **Customizable Architecture:** Support for any number of hidden layers and neurons.
*   **Activation Functions:** Manual implementation of `Sigmoid` and `Hyperbolic Tangent (tanh)` with their respective derivatives.
*   **Bias Integration:** Automatic handling of bias units through matrix concatenation.
*   **Weight updates via backpropagation:** Using stochastic gradient descent.
*   **Loss Visualization:** Visualization of the loss curve using Matplotlib

## 🛠️ How it Works

The network follows the standard forward and backward pass of a neural network:
1. **Linear Combination:** $z = w \cdot x + b$
2. **Activation:** $a = f(z)$
3. **Backpropagation:** Computing deltas for each layer using the chain rule to update weights.

## 📋 Requirements

*   Python 3.x
*   NumPy
*   Matplotlib (for cost visualization)

## 💻 Usage

To train the model with the provided dataset:

```bash
python main.py

```
---
## 📊 Empirical Analysis & Limitations

The network successfully learns linear relationships (like identity and averaging) but struggles with the Parity Problem (first output neuron), which requires capturing more complex non-linear patterns.

**Comparison of Expected vs. Predicted values:**

| Metric | Target (Expected) | Prediction (Actual) | Analysis |
| :--- | :--- | :--- | :--- |
| **Parity (Sum is Odd)** | `1.00` | `0.45` | Struggling |
| **Identity (Col 1)** | `1.00` | `0.97` | **Success** |
| **Mean (Col 1 & 7)** | `0.50` | `0.49` | **Success** |


**Individual Testing Samples:**

*   **Sample 1:**
    *   Input: `[1, 0, 1, 1, 0, 1, 0, 0, 1, 0]`
    *   Target: `[1.0, 1.0, 0.5]` → Predicted: `[0.45, 0.97, 0.49]`
*   **Sample 2:**
    *   Input: `[0, 0, 1, 0, 1, 1, 1, 0, 1, 0]`
    *   Target: `[1.0, 0.0, 1.0]` → Predicted: `[0.48, -0.01, 0.87]`
*   **Sample 3:**
    *   Input: `[1, 0, 0, 0, 1, 1, 0, 0, 0, 1]`
    *   Target: `[0.0, 1.0, 0.5]` → Predicted: `[0.45, 0.97, 0.57]`



