import numpy as np
import matplotlib.pyplot as plt
import random
import math

train_set = np.loadtxt('D:\\学习资料\\KTH\\ANN\\Lab\\Lab2\\ballist.dat')
test_set = np.loadtxt('D:\\学习资料\\KTH\\ANN\\Lab\\Lab2\\balltest.dat')

x_train = train_set[:, 0:2]
y_train = train_set[:, 2:4]
x_test = test_set[:, 0:2]
y_test = test_set[:, 2:4]

N = 7
centers = np.zeros((N, 2))
centers[:, 0] = np.random.rand(N)
centers[:, 1] = np.random.rand(N)
temp = np.zeros(N)
sigma = 0.25
rate1 = 0.2
epoch = 5000
for i in range(epoch):
    x = random.choice(x_train)
    a = float('inf')
    for j in range(N):
        temp[j] = (x[0] - centers[j][0])*(x[0] - centers[j][0]) + (x[1] - centers[j][1])*(x[1] - centers[j][1])
        if temp[j] < a:
            index = j
            a = temp[j]
    centers[index] += rate1*(x - centers[index])

plt.scatter(centers[:, 0], centers[:, 1])
plt.title('Position of cluster centers using CL approach (7 units)')
plt.xlabel('angle')
plt.ylabel('velocity')
plt.show()

n = len(x_train)
weights = np.zeros((N, 2))
weights[:, 0] = np.random.rand(N)
weights[:, 1] = np.random.rand(N)
delta_w = np.zeros((N, 2))
delta_w = np.zeros((N, 2))
rate = 0.3
hidden_out = np.zeros((n, N))
for i in range(n):
    for j in range(N):
        hidden_out[i][j] = math.exp(-0.5/sigma/sigma*(np.square(x_train[i][0] - centers[j][0]) + np.square(x_train[i][1] - centers[j][1])))
    train_result = np.dot(hidden_out[i], weights)
    train_error = y_train[i] - train_result
    delta_w[:, 0] = rate*train_error[0]*hidden_out[i]
    delta_w[:, 1] = rate * train_error[1] * hidden_out[i]
    weights = weights + delta_w

# print(weights)
hidden_out2 = np.zeros((len(x_test), N))
results = np.zeros((len(x_test), 2))
for i in range(len(x_test)):
    for j in range(N):
        hidden_out2[i][j] = math.exp(-0.5/sigma/sigma*(np.square(x_test[i][0] - centers[j][0]) + np.square(x_test[i][1] - centers[j][1])))
    results[i] = np.dot(hidden_out2[i], weights)

error_dis = np.mean(np.abs(y_test[:, 0] - results[:, 0]))
error_hei = np.mean(np.abs(y_test[:, 1] - results[:, 1]))
print('Error of distance on test set: ', error_dis)
print('Error of height on test set: ', error_hei)
# plt.scatter(results[:, 0], color='red')
# plt.scatter(y_test[:, 0], color='blue')
# plt.show()
