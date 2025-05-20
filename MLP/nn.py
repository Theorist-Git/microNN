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
            "softmax": lambda x: x.softmax(),
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

    @staticmethod
    def __mse(y_true: Value, y_pred: Value, batch_size: int) -> Value:
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")

        _vector_loss = (y_true - y_pred) ** 2

        return _vector_loss.collapse_to_scalar() / batch_size

    @staticmethod
    def __bce(y_true: Value, y_pred: Value, batch_size: int) -> Value:
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")

        eps = 1e-8

        _vector_loss: Value =  -(y_true * (y_pred + eps).ln()) - ((1 - y_true) * (1 - y_pred + eps).ln())

        return _vector_loss.collapse_to_scalar() / batch_size

    @staticmethod
    def __cce(y_true: Value, y_pred: Value, batch_size: int) -> Value:
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")

        eps = 1e-8

        # y_pred are the softmax probabilities
        # [ [p1, p2, p3, ....] ]
        # y_true [ [1, 0, 2, ...] ]
        y_true_1d: np.ndarray = y_true.data.ravel()
        assert y_true_1d.shape[0] == batch_size

        probability_pairs: Value = y_pred[np.arange(len(y_true_1d)), y_true_1d.astype(int)]
                           # Value([0.6, 0.6, 0.5])

        _vector_loss     : Value = -(probability_pairs + eps).ln()

        return _vector_loss.collapse_to_scalar() / batch_size

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def fit(self, x: np.array, y: np.array, loss_fn: str, batch_size: int, verbose: bool = True, patience: int = None):
        eps = 1e-8

        loss_map = {
            "mse": self.__mse,
            "binary_cross_entropy": self.__bce,
            "categorical_cross_entropy": self.__cce,
        }

        n_samples = x.shape[0]

        for k in range(self.epochs):
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                xi = x[batch_idx]

                # forward propagation
                y_pred: Value = self(xi)
                y_true        = Value(y[batch_idx].reshape(-1,1))

                # cost calculation
                # noinspection PyTypeChecker
                cost: Value = loss_map[loss_fn](y_true, y_pred, len(batch_idx))

                # backpropagation
                self.zero_grad()
                cost.backward()

                # update
                for p in self.parameters():
                    p.data -= self.lr * p.grad

                if verbose:
                    print(f"EPOCH {k}: Batch {i}(size={len(batch_idx)}): {loss_fn} = {cost.data.item()}")
