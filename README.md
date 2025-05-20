# MicroNN
## Minimal Autograd and ANN Engine

- Implements a fully vectorized automatic differentiation engine and neural network (ANN) framework from scratch in Python, inspired by Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd).
- Extends the scalar-based approach to support tensors with NumPy-based broadcasting, vectorized operations, and batched computation—enabling efficient training and inference similar to libraries like PyTorch.

## Installation

Clone the repository and ensure you have NumPy installed.

```bash
git clone https://github.com/Theorist-Git/microNN
pip install numpy
```

## Usage

### MLP Definition

```python
from MLP.nn import MLP

# MLP with 2 input features, two hidden layers, and 1 output neuron
net = MLP(
    n_inputs=2,
    layers=[(4, "relu"), (1, "sigmoid")],
    epochs=100,
    learning_rate=0.01
)
```

### Training

```python
import numpy as np

# Example data
X = np.random.randn(100, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(float)

# Fit model
net.fit(x=X, y=y, loss_fn="binary_cross_entropy", batch_size=16)
```

---

## Autograd Graph Visualization

The `.draw_graph()` method in the `Value` class uses `graphviz` to produce computation graphs.

```python
from MLP.grad_engine import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a * b
c.backward()
dot = c.draw_graph()
dot.render("graph", format="svg")
```

### Requirements

- Python 3.7+
- `numpy`
- `graphviz` (optional, for computation graph visualization)
