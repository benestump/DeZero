class Variable:
    def __init__(self, data:ndarray):
        self.data = data

class Function:
    def __call__(self, input:Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x) -> Variable:
        return x ** 2

class Exp(Function):
    def forward(self, x) -> Variable:
        return np.exp(x)

def neumerical_diff(f:Function, x:Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

