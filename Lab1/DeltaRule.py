import numpy as np

np.random.seed(1000)
DEBUG = 1

if DEBUG:
    my_print = print
else:
    def my_print():
        pass

# generate inputs
n_samples = np.random.randint(10, 100)
n_x = np.random.randint(10, 100)
X = np.random.normal(0, 1, (n_samples, n_x))

l1 = np.ones(n_samples // 2)
l2 = np.ones(n_samples - n_samples // 2) * (-1)
T = np.append(l1, l2)
np.random.shuffle(T)
T = np.transpose(T)
my_print(T)

# settings
# learning rate eta
eta = 1e-3
EPOCH = 20


class Delta:
    def __init__(self, inputs=[], targets=[], EPOCH=20, eta=1e-3):
        self.inputs = inputs  # n_samples x n_x
        self.targets = targets  # n_samples x 1
        self.EPOCH = EPOCH
        self.eta = eta
        self.n_samples = inputs.shape[0]
        self.n_x = inputs.shape[1]
        my_print(n_samples, n_x)
        self.W = np.random.normal(0, 0.01, self.n_x)  # n_x x 1
        self.W = np.transpose(self.W)
        my_print(self.W)

    def update(self):
        for i in range(self.EPOCH):
            d_W = -self.eta * np.dot(np.transpose(self.inputs), (np.dot(self.inputs, self.W) - self.targets))
            self.W = self.W + d_W
            my_print("Epoch " + str(i))


if __name__ == '__main__':
    delta = Delta(X, T, EPOCH, eta)
    delta.update()
