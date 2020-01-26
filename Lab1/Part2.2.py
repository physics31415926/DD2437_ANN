import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd
#  Data generation
data_number = 1525
x = np.zeros(data_number)
x[0] = 1.5
for i in range(0, 25):
    x[i+1] = 0.9*x[i]
for i in range(25, data_number - 1):
    x[i+1] = 0.9*x[i]+0.2*x[i-25]/(1+pow(x[i-25], 10))
sigma = 0.03
x = x + np.random.normal(0, sigma, data_number)

# Plot raw data
# plt.plot(x, color='blue')
# plt.xlabel('t', fontsize=22)
# plt.ylabel('x(t)', fontsize=22)
# plt.tick_params(labelsize=18)
# plt.legend('x', fontsize=18)
# plt.title('Data generation', fontsize=26)
# plt.show()

output = np.zeros((data_number - 25, 1))
input = np.zeros((data_number - 25, 5))
for t in range(0, data_number - 25):
    output[t] = x[t+5]
    for i in range(0, 5):
        input[t, i] = x[t + i * 5]

# Split data set
training_length = 800
validation_length = 200
test_length = 200
training_data = input[300:300+training_length]
training_output = output[300:300+training_length]
validation_data = input[300+training_length: 300+training_length+validation_length]
validation_output = output[300+training_length: 300+training_length+validation_length]
test_data = input[1500-test_length:1500]
test_output = output[1500-test_length:1500]

# Define data structure
xs = tf.compat.v1.placeholder(tf.float64, [None, 5])
ys = tf.compat.v1.placeholder(tf.float64, [None, 1])


# Define a ANN layer
def add_layer(in_put, in_size, out_size, activation_function=None):

    weight = tf.Variable(np.random.randn(in_size, out_size))
    biases = tf.Variable(np.zeros([1, out_size]) + 0.01)
    w_mul_x_plus_b = tf.matmul(in_put, weight)
    if activation_function == None:
        out_put = w_mul_x_plus_b
    else:
        out_put = activation_function(w_mul_x_plus_b)
    return out_put, weight

# Hidden layer
hidden_layer1_len = 5
hidden_layer2_len = 2
[hidden_layer1, weight1] = add_layer(training_data, 5, hidden_layer1_len, activation_function=None)
[hidden_layer2, weight2] = add_layer(hidden_layer1, hidden_layer1_len, hidden_layer2_len, activation_function=None)

# Output layer
[prediction, weight3] = add_layer(hidden_layer2, hidden_layer2_len, 1, activation_function=None)

# Training
step = 0.001
reg_str = 0.1
reg_item = tf.reduce_sum(tf.reduce_sum(tf.multiply(weight1, weight1)))+tf.reduce_sum(tf.reduce_sum(tf.multiply(weight2, weight2)))+tf.reduce_sum(tf.reduce_sum(tf.multiply(weight3, weight3)))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1])) + reg_str*reg_item
train_step = tf.compat.v1.train.GradientDescentOptimizer(step).minimize(loss)
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
steps = 10000
validation_error = float('inf')
# saver = tf.train.Saver()
# tf.add_to_collection('parameters', weight1, weight2)
for i in range(steps):
    sess.run(train_step, feed_dict={xs: training_data, ys: training_output})
    w1 = (sess.run(weight1))
    w2 = (sess.run(weight2))
    w3 = (sess.run(weight3))
    temp = np.dot(validation_data, w1)
    temp = np.dot(temp, w2)
    temp = np.dot(temp, w3)
    validation_result = np.mean(np.square(temp-validation_output))
    if i % 200 == 0:
        training_value = sess.run(prediction, feed_dict={xs: training_data, ys: training_output})
        plt.plot(training_value, color='red')
        plt.plot(training_output, color='blue')
        plt.show()
    if validation_result < validation_error:
        validation_error = validation_result
    else:
        break
w1 = (sess.run(weight1))
w2 = (sess.run(weight2))
w3 = (sess.run(weight3))
temp = np.dot(test_data, w1)
test_result = np.dot(temp, w2)
test_result = np.dot(test_result, w3)
test_accuacy = np.mean(np.square(test_result-test_data))
print(test_accuacy)
sess.close()
