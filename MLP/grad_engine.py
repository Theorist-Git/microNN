from graphviz import Digraph
from sys import exit
import numpy as np

class Value:

    def __init__(self, data, _children=(), _op="", label=""):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        if data.ndim == 0:
            data = data.reshape(1, 1)

        self.data = data
        self.grad = np.zeros_like(data, dtype=float)

        self._backward = lambda: None
        self._prev = _children  # check for set
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __reverse_numpy_broadcast(self, gradient: np.ndarray):
        # summing up leading axes
        extra_axes = tuple(range(gradient.ndim - self.data.ndim))
        gradient = gradient.sum(axis=extra_axes)

        # axes that were broadcast because their original size was 1
        # collapse any axis where self.data was 1 but grad is >1
        broadcast_axes = []

        for axis_idx, (orig_dim, grad_dim) in enumerate(zip(self.data.shape, gradient.shape)):
            if orig_dim == 1 and grad_dim > 1:
                broadcast_axes.append(axis_idx)

        if broadcast_axes:
            gradient = gradient.sum(axis=tuple(broadcast_axes), keepdims=True)

        return gradient

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
            gradient = out.grad

            if self.data.shape != out.grad.shape:
                self.grad += self.__reverse_numpy_broadcast(gradient=gradient)
            else:
                self.grad += gradient

            if other.data.shape != out.grad.shape:
                other.grad += other.__reverse_numpy_broadcast(gradient=gradient)
            else:
                other.grad += gradient

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

    def collapse_to_scalar(self):

        out = Value(self.data.sum(), (self, ), _op="collapse_to_scalar")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
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
            gradient = other.data * out.grad

            if self.data.shape != out.grad.shape:
                self.grad += self.__reverse_numpy_broadcast(gradient=gradient)
            else:
                self.grad += gradient

            gradient = self.data * out.grad

            if other.data.shape != out.grad.shape:
                other.grad += other.__reverse_numpy_broadcast(gradient=gradient)
            else:
                other.grad += gradient

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        """
        -> 2(int/float) * Value
        First, 2.__mul__(Value) will fail and python will try
        Value.__rmul__(2), we overload __rmul__ and make it
        equivalent to Value.__mul__(2)
        """
        return self * other

    def __matmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        mat_a, mat_b = self, other

        out = Value(mat_a.data @ mat_b.data, (mat_a, mat_b), "matmul")

        def _backward():
            grad_a = out.grad @ mat_b.data.T
            grad_b = mat_a.data.T @ out.grad

            if grad_a.shape != mat_a.data.shape:
                grad_a = mat_a.__reverse_numpy_broadcast(grad_a)
            if grad_b.shape != mat_b.data.shape:
                grad_b = mat_b.__reverse_numpy_broadcast(grad_b)

            mat_a.grad += grad_a
            mat_b.grad += grad_b

        out._backward = _backward

        return out

    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "Only int/float powers are supported"

        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

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
        t = np.tanh(x)

        out = Value(t, (self,), _op="tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        z = 1 / (1 + np.exp(-x))

        out = Value(z, (self, ), _op="sigmoid")

        def _backward():
            self.grad += z * (1 - z) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        r = np.maximum(x, 0)

        out = Value(r, (self, ), _op="ReLU")

        def _backward():
            grad_mask = (x > 0).astype(float)
            self.grad += grad_mask * out.grad

        out._backward = _backward

        return out

    def exp_(self):
        x = self.data

        out = Value(np.exp(x), (self,), _op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def ln(self):
        x = self.data

        try:
            out = Value(np.log(x), (self, ), _op="ln")
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

        for node in topo:
            # if node.data is an array, keep dtype/shape
            node.grad = np.zeros_like(node.data, dtype=float)

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