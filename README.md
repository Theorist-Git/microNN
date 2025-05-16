# Simple Autograd and ANN Engine

This project is a simple implementation of an automatic differentiation engine and a neural network (ANN) built from scratch in Python. It is inspired by and builds upon Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), extending it with a multi-layer perceptron (MLP) architecture and more activation functions.

## Features

- Custom `Value` class implementing forward and backward passes for automatic differentiation.
- Fully connected `Neuron`, `Layer`, and `MLP` classes supporting flexible architectures.
- Supports activation functions: `linear`, `sigmoid`, `tanh`, and `relu`.
- Training loop with gradient descent optimizer.
- Example regression dataset for testing and demonstration.
- Simple API to build and train neural networks from scratch.

## Getting Started

### Requirements

- Python 3.7+
- `numpy`
- `graphviz` (optional, for computation graph visualization)
