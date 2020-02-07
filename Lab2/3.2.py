import math as math
import numpy as np

step = 0.1
x_train = np.arange(0, 2*math.pi, step)
np.random.shuffle(x_train)
print(x_train)
x_test = np.arange(0.05, 2*math.pi, step)
y_train = np.zeros(len(x_train))
y_test = np.zeros(len(x_test))
n = len(x_train)
for i in range(n):
    y_train[i] = math.sin(2*x_train[i])
for i in range(len(x_test)):
    y_test[i] = math.sin(2*x_test[i])
noise = np.random.normal(0, 0.1, len(y_train))
y_train = y_train + noise
y_test = y_test + noise

step2 = 0.5
sigma = 0.25
error = np.zeros(1000)
for k in range(1000):
    rate = 0.001*k
    centers = np.arange(0, 2*math.pi, step2)
    N = len(centers)
    weights = np.random.normal(1, 1, N)
    delta_w = np.zeros(N)

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
    #     print(results[i])
    # print(results)
    # print(y_test)
    error[k] = np.mean(np.abs(y_test - results))
    print(rate, error[k])
    # print('Error:', error)
