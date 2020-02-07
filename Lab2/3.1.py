import math as math
import numpy as np

step = 0.1
x_train = np.arange(0, 2*math.pi, step)
x_test = np.arange(0.05, 2*math.pi, step)
y_train = np.zeros(len(x_train))
y_test = np.zeros(len(x_test))
n = len(x_train)
for i in range(n):
    y_train[i] = math.sin(2*x_train[i])
for i in range(len(x_test)):
    y_test[i] = math.sin(2*x_test[i])

step2 = 0.5
sigma = 1.1
centers = np.arange(0, 2*math.pi, step2)
N = len(centers)
weights = np.random.normal(1, 1, N)
hidden_out = np.zeros((n, N))
for i in range(n):
    for j in range(N):
        hidden_out[i][j] = math.exp(-0.5/sigma/sigma*np.square(x_train[i]-centers[j]))

# for i in range(n):
#     for j in range(N):
#         hidden_out[i][j] = math.exp(-0.5/sigma/sigma*np.square(x_train[i]-centers[j]))
#     results1 = np.dot(hidden_out[i, :], weights)

temp1 = np.dot(hidden_out.T, hidden_out)
temp2 = np.dot(hidden_out.T, y_train)
weights = np.dot(np.linalg.inv(temp1), temp2)
print(weights)
hidden_out2 = np.zeros((len(x_test), N))
results = np.zeros(len(x_test))
for i in range(len(x_test)):
    for j in range(N):
        hidden_out2[i][j] = math.exp(-0.5/sigma/sigma*np.square(x_test[i]-centers[j]))
    results[i] = np.dot(hidden_out2[i, :], weights)

error = np.mean(np.abs(y_test - results))
print('Error:', error)
