from numbers import Real


class Value(object):
    def __init__(self, data, parents=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._operation = operation
        self._backward = lambda: None
        self._parents = set(parents)

    def __add__(self, other):
        assert isinstance(other, (Real, Value)), "Unsupported addition type."
        if isinstance(other, Real):
            other = Value(other)

        new_value = Value(self.data + other.data, parents=(self, other), operation='+')

        # Addition broadcasts gradient since d (other + self) / d self = 1
        # Also, the gradient must be accumulated:
        # Take self + self, the gradient is 2 in this case, not 1
        def backward():
            self.grad += new_value.grad * 1
            other.grad += new_value.grad * 1

        new_value._backward = backward
        return new_value

    def __radd__(self, other):
        return other + self

    def __mul__(self, other):
        assert isinstance(other, (Real, Value)), "Unsupported multiplication type."
        if not isinstance(other, Value):
            other = Value(other)

        new_value = Value(self.data * other.data, parents=(self, other), operation='*')

        # Multiplication gradient for self:
        # d (other * self) / d self = other
        def backward():
            self.grad += new_value.grad * other.data
            other.grad += new_value.grad * self.data

        new_value._backward = backward
        return new_value

    def __pow__(self, other):
        assert isinstance(other, Real), "Unsupported power type."

        new_value = Value(self.data ** other, parents=(self,), operation=f'**{other}')

        # Power gradient for self:
        # d (self ** other) / d self = other * self ** (other - 1)
        def backward():
            self.grad += new_value.grad * (other * self.data ** (other - 1))

        new_value._backward = backward
        return new_value

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __rmul__(self, other):
        return other * self

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def backward(self):
        sorted_children = []
        visited = set()

        def topologically_sort_children(value):
            if value not in visited:
                visited.add(value)
                for child in value._parents:
                    topologically_sort_children(child)
                sorted_children.append(value)

        topologically_sort_children(self)

        # Need to set to 1 since by default it is 0.
        # Using the chain rule with 0 will mean multiplying by 0 again and again.
        self.grad = 1.0
        for value in reversed(sorted_children):
            value._backward()

    def __repr__(self):
        return f"Value(data={self.data!r}, grad={self.grad!r})"  # _operation={self._operation!r}, _parents={self._parents!r}, _backward={self._backward}