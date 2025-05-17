import numpy as np
from typing import Callable, List, Tuple
from MLP.grad_engine import Value


class Neuron:

    def __init__(self, n_inputs: int, activation: str = "linear"):
        self.w = [Value(np.random.rand() * 2 - 1) for _ in range(n_inputs)]
        self.b = Value(np.random.rand() * 2 - 1)

        activation_map: dict[str, Callable[[Value], Value]] = {
            "tanh"   : lambda x: x.tanh(),
            "relu"   : lambda x: x.relu(),
            "sigmoid": lambda x: x.sigmoid(),
            "linear" : lambda x: x,
        }

        self.activation = activation_map[activation]

    def __call__(self, x):
        assert len(x) == len(self.w), f"Expected {len(self.w)} inputs but got {len(x)}"

        weighted_sum = self.b
        for xi, wi in zip(x, self.w):
            weighted_sum += wi * xi  # Uses Value.__mul__ and __add__

        return self.activation(weighted_sum)

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "linear"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
        self.activation = activation

    def __call__(self, x):
        outs = [_neuron(x) for _neuron in self.neurons]

        if len(outs) == 1:
            return outs[0]

        return outs

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

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

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

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

        no_improvement_cnt = 0
        prev_cost          = 0

        for k in range(self.epochs):

            # forward propagation
            y_pred = [self(x) for x in x]
            # noinspection PyTypeChecker
            cost: Value = sum([loss_map[loss_fn](y_true, y_hat) for y_true, y_hat in zip(y, y_pred)]) / len(y)

            if patience is not None:
                if abs(cost.data - prev_cost) < 1e-4:
                    no_improvement_cnt += 1
                else:
                    no_improvement_cnt = 0

                if no_improvement_cnt >= patience:
                    print(f"Early Stopping: EPOCH {k}: {loss_fn} = {cost.data}")
                    break

                prev_cost = cost.data

            # backpropagation
            self.zero_grad()
            cost.backward()

            # update
            for p in self.parameters():
                p.data -= self.lr * p.grad

            print(f"EPOCH {k}: {loss_fn} = {cost.data}")
