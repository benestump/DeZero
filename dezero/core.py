class Variable:
    def __init__(self, data:ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # get function
        if f is not None:
            x = f.input # get function's input
            x.grad = f.backward(self.grad) # call functions's backward
            x.backward()

class Function:
    def __call__(self, input:Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x:ndarray):
        raise NotImplementedError()

    def backward(self, gy:ndarray):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x:ndarray) -> ndarray:
        return x ** 2

    def backward(self, gy:ndarray) -> ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x:ndarray) -> Variable:
        return np.exp(x)

    def backward(self, gy:ndarray) -> ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def neumerical_diff(f:Function, x:Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

