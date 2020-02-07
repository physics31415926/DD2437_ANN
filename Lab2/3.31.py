import math as math
import numpy as np
import random
import matplotlib.pyplot as plt

step = 0.1
x_train = np.arange(0, 2*math.pi, step)
# np.random.shuffle(x_train)
x_test = np.arange(0.05, 2*math.pi, step)
y_train = np.zeros(len(x_train))
y_test = np.zeros(len(x_test))
n = len(x_train)
for i in range(n): y_train[i] = math.sin(2*x_train[i])
for i in range(len(x_test)): y_test[i] = math.sin(2*x_test[i])
# noise = np.random.normal(0, 0.1, len(y_train))
# y_train = y_train + noise
# y_test = y_test + noise

N = 13
centers = np.random.normal(3, 1, N)
temp = np.zeros(N)
sigma = 0.25
rate1 = 0.2
epoch = 5000
for i in range(epoch):
    x = random.choice(x_train)
    a = float('inf')
    for j in range(N):
        temp[j] = (x - centers[j])*(x - centers[j])
        if temp[j] < a:
            index = j
            a = temp[j]
    centers[index] += rate1*(x - centers[index])

print(centers)

weights = np.random.rand(N)
delta_w = np.zeros(N)
rate = 0.3
hidden_out = np.zeros((n, N))
for i in range(n):
    for j in range(N):
        hidden_out[i][j] = math.exp(-0.5/sigma/sigma*np.square(x_train[i]-centers[j]))
    train_result = np.dot(hidden_out[i], weights)
    train_error = y_train[i] - train_result
    delta_w = rate*train_error*hidden_out[i]
    weights = weights + delta_w

# print(weights)
hidden_out2 = np.zeros((len(x_test), N))
results = np.zeros(len(x_test))
for i in range(len(x_test)):
    for j in range(N):
        hidden_out2[i][j] = math.exp(-0.5/sigma/sigma*np.square(x_test[i]-centers[j]))
    results[i] = np.dot(hidden_out2[i], weights)

error = np.mean(np.abs(y_test - results))
print(error)
plt.plot(x_test, results, color='red', label='Predicted data')
plt.plot(x_test, y_test, color='blue', label='Test data')
plt.legend()
plt.title('Result of comparative learning on test set')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
