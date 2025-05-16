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
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return self.activation(x @ self.w + self.b)

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

    def __init__(self, n_inputs: int, layers: List[Tuple[int, str]]):
        arch = [n_inputs]
        activations = []

        for n_neurons, activation in layers:
            arch.append(n_neurons)
            activations.append(activation)

        self.layers = [Layer(arch[i], arch[i + 1], activation=activations[i]) for i in range(len(layers))]

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