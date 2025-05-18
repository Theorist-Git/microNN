from graphviz import Digraph
from math import exp, log
from sys import exit
import numpy as np

class Value:

    def __init__(self, data: np.ndarray, _children=(), _op="", label=""):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        self.data = data
        self.grad = np.zeros_like(data, dtype=float)

        self._backward = lambda: None
        self._prev = _children  # check for set
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        """
        -> Value  + 2(int/float)
        -> Value1 + Value2
        Checks if **other**, the right hand operator (a + b) is a
        Value type object or not. If not, it is converted
        """
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():

            if self.data.shape != out.grad.shape:
                # summing up leading axes
                extra_axes = tuple(range(out.grad.ndim - self.data.ndim))
                self_grad = out.grad.sum(axis=extra_axes, keepdims=True)

                # axes that were broadcast because their original size was 1
                # collapse any axis where self.data was 1 but grad is >1
                broadcast_axes = []

                for axis_idx, (orig_dim, grad_dim) in enumerate(zip(self.data.shape, self_grad.shape)):
                    if orig_dim == 1 and grad_dim > 1:
                        broadcast_axes.append(axis_idx)

                if broadcast_axes:
                    self_grad = self_grad.sum(axis=tuple(broadcast_axes), keepdims=True)

                self.grad += self_grad
            else:
                self.grad += out.grad

            if other.data.shape != out.grad.shape:
                # summing up leading axes
                extra_axes = tuple(range(out.grad.ndim - other.data.ndim))
                other_grad = out.grad.sum(axis=extra_axes, keepdims=True)

                # axes that were broadcast because their original size was 1
                # collapse any axis where self.data was 1 but grad is >1
                broadcast_axes = []

                for axis_idx, (orig_dim, grad_dim) in enumerate(zip(other.data.shape, other_grad.shape)):
                    if orig_dim == 1 and grad_dim > 1:
                        broadcast_axes.append(axis_idx)

                if broadcast_axes:
                    other_grad = other_grad.sum(axis=tuple(broadcast_axes), keepdims=True)

                other.grad += other_grad

            else:
                other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Value"):
        """
        -> Value  * 2(int/float)
        -> Value1 * Value2
        Checks if **other**, the right hand operator (a + b) is a
        Value type object or not. If not, it is converted
        """
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "Only int/float powers are supported"

        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):  #
        """
        -> 2(int/float) + Value
        First, 2.__add__(Value) will fail and python will try
        Value.__radd__(2), we overload __radd__ and make it
        equivalent to Value.__add__(2)
        """
        return self + other

    def __rmul__(self, other):  # other * self
        """
        -> 2(int/float) * Value
        First, 2.__mul__(Value) will fail and python will try
        Value.__rmul__(2), we overload __rmul__ and make it
        equivalent to Value.__mul__(2)
        """
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def tanh(self):
        x = self.data
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1)

        out = Value(t, (self,), _op="tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        z = 1 / (1 + exp(-x))

        out = Value(z, (self, ), _op="sigmoid")

        def _backward():
            self.grad += z * (1 - z) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        r = max(0, x)

        out = Value(r, (self, ), _op="ReLU")

        def _backward():
            if x <= 0:
                self.grad += 0
            else:
                self.grad += 1 * out.grad

        out._backward = _backward

        return out

    def exp_(self):
        x = self.data

        out = Value(exp(x), (self,), _op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def ln(self):
        x = self.data

        try:
            out = Value(log(x), (self, ), _op="ln")
        except ValueError as e:
            print(f"Exception: `{e}` occurred during the calculation ln({x})")
            exit(1)

        def _backward():
            self.grad += (1 / x) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, is_expanded = stack.pop()

            if node in visited:
                continue
            if is_expanded:
                visited.add(node)
                topo.append(node)
            else:
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))

        self.grad = np.ones_like(self.data)  # start backprop from output node
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _trace(root: "Value"):
        """
        starts from the final output node and grows(traces) backwards to display
        the journey of the output.
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)

        d = a * b + c

        a \
            * \
        b /    \
                + ----- D(root)
               /
            c /
        """

        nodes, edges = set(), set()

        def build(vertex: Value):
            if vertex not in nodes:
                nodes.add(vertex)

                for child in vertex._prev:
                    edges.add((child, vertex))  # child ---> vertex
                    build(child)

        build(root)

        return nodes, edges

    def draw_graph(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

        nodes, edges = Value._trace(self)

        for n in nodes:
            uid = str(id(n))

            dot.node(name=uid, label=f"{{ {n.label} | data{n.data} | grad {n.grad}}}", shape="record")

            if n._op != "":
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)

        for from_, to_ in edges:
            dot.edge(str(id(from_)), str(id(to_)) + to_._op)

        return dot