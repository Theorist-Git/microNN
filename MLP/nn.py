import numpy as np
from typing import Callable, List, Tuple
from MLP.grad_engine import Value


class Layer:

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "linear"):
        if activation == "tanh":
            std = np.sqrt(1.0 / n_inputs)
            W = np.random.randn(n_inputs, n_neurons) * std
            self.w = Value(W)
            self.b = Value(np.zeros((1, n_neurons)))
        elif activation == "relu":
            std = np.sqrt(2.0 / n_inputs)
            W = np.random.randn(n_inputs, n_neurons) * std
            self.w = Value(W)
            self.b = Value(np.zeros((1, n_neurons)))
        else:
            W = np.random.randn(n_inputs, n_neurons) * 0.01
            self.w = Value(W)
            self.b = Value(np.zeros((1, n_neurons)))

        activation_map: dict[str, Callable[[Value], Value]] = {
            "tanh": lambda x: x.tanh(),
            "relu": lambda x: x.relu(),
            "sigmoid": lambda x: x.sigmoid(),
            "linear": lambda x: x,
        }

        self.activation = activation_map[activation]

    def __call__(self, x: Value):
        return self.activation(x @ self.w + self.b)

    def parameters(self):
        params = [self.w, self.b]

        return params


class MLP:

    def __init__(self, n_inputs: int, layers: List[Tuple[int, str]], epochs: int, learning_rate = 0.01):
        arch = [n_inputs]
        activations = []

        for n_neurons, activation in layers:
            arch.append(n_neurons)
            activations.append(activation)

        self.layers = [Layer(arch[i], arch[i + 1], activation=activations[i]) for i in range(len(layers))]
        self.epochs = epochs
        self.lr     = learning_rate

    def __call__(self, x: np.ndarray) -> Value:

        out = Value(x)

        for layer in self.layers:
            out = layer(out)

        return out

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def fit(self, x: np.array, y: np.array, loss_fn: str, patience: int = None):
        eps = 1e-8

        loss_map = {
            "mse": lambda y_true, y_hat: (y_true - y_hat) ** 2,
            "binary_cross_entropy": lambda y_true, y_hat: -(y_true * (y_hat + eps).ln()) - \
                                                          ((1 - y_true) * (1 - y_hat + eps).ln())
        }

        y_true = Value(y.reshape(-1,1))

        for k in range(self.epochs):

            # forward propagation
            y_pred = self(x)
            # noinspection PyTypeChecker
            cost: Value = loss_map[loss_fn](y_true, y_pred).collapse_to_scalar() / len(y)

            # backpropagation
            self.zero_grad()    
            cost.backward()

            # update
            for p in self.parameters():
                p.data -= self.lr * p.grad

            print(f"EPOCH {k}: {loss_fn} = {cost.data.item()}")
