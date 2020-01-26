import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import trange
from abc import ABCMeta, abstractmethod


class Parameter(object):
    def __init__(self, data, requires_grad, skip_decay=False):
        self.data = data
        self.grad = None
        self.skip_decay = skip_decay
        self.requires_grad = requires_grad

    @property
    def T(self):
        return self.data.T


class SGD(object):
    def __init__(self, parameters, lr, decay=0):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.decay_rate = 1.0 - decay

    def update(self):
        for p in self.parameters:
            if self.decay_rate < 1 and not p.skip_decay: p.data *= self.decay_rate
            # print("data",p.data,"grad",p.grad)
            p.data -= self.lr * p.grad


class MSE:
    def __init__(self):
        pass

    def gradient(self):
        return self.a - self.y

    def __call__(self, output, target, requires_acc=True):
        self.a = output
        self.y = np.reshape(target, (-1, 1))
        loss = 0.5 * np.multiply(self.a - self.y, self.a - self.y).mean()
        if requires_acc:
            acc = np.sum(np.sign(output) == self.y) / output.shape[0]
            return loss, acc
        return loss


class Layer():
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass


class Tanh(Layer):
    def forward(self, x):
        ex = np.exp(x)
        esx = np.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', 1 - self.y, 1 + self.y, eta, optimize=True)


class Linear(Layer):
    def __init__(self, shape, requires_grad=True, bias=True, **kwargs):
        '''
        shape = (in_size, out_size)
        '''
        W = np.random.randn(*shape) * (2 / shape[0] ** 0.5)
        #  print(W.shape)
        self.W = Parameter(W, requires_grad)
        self.b = Parameter(np.zeros(shape[-1]), requires_grad) if bias else None
        self.require_grad = requires_grad

    def forward(self, x):
        if self.require_grad: self.x = x
        out = np.dot(x, self.W.data)
        if self.b is not None: out = out + self.b.data
        return out

    def backward(self, eta):
        if self.require_grad:
            batch_size = eta.shape[0]
            self.W.grad = np.dot(self.x.T, eta) / batch_size
            if self.b is not None: self.b.grad = np.sum(eta, axis=0) / batch_size
        return np.dot(eta, self.W.T)


class Net(Layer):
    def __init__(self, layer_configures):
        self.layers = []
        self.parameters = []
        for config in layer_configures:
            self.layers.append(self.createLayer(config))

    def createLayer(self, config):
        return self.getDefaultLayer(config)

    def getDefaultLayer(self, config):
        t = config['type']
        if t == 'linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.b is not None: self.parameters.append(layer.b)
        elif t == 'tanh':
            layer = Tanh()
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta

    def train(self, X, Y, optimizer, batch_size=16, epochs=500, loss=MSE()):
        n = len(Y)
        for epoch in trange(epochs):
            i = 0
            while i <= n - batch_size:
                x, y = X[i:i + batch_size, ], Y[i:i + batch_size, ]
                i += batch_size
                output = self.forward(x)
                batch_loss, batch_acc = loss(output, y)
                eta = loss.gradient()
                self.backward(eta)
                optimizer.update()
                # if epoch % 100==0:
                # print("epoch: %d, batch: %5d, batch_acc:    %.2f,batch loss: %.2f" % \
                # (epoch, i/batch_size,batch_acc*100,batch_loss))
        print("epoch: %d, batch: %5d, batch_acc:    %.2f,batch loss: %.2f" % \
              (epoch, i / batch_size, batch_acc * 100, batch_loss))

    def predict(self, X):
        return self.forward(X)


'''
layers = [
    {'type': 'linear', 'shape': (2, 1)},
    {'type': 'tanh'},
    {'type': 'linear', 'shape': (1, 1)},
    {'type': 'tanh'},
    {'type': 'linear', 'shape': (1, 1)},
    {'type': 'tanh'}
]
net = Net(layers)
opt = SGD(net.parameters, lr=1e-3)
net.train(X, Y, opt)

'''
