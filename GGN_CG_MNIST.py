##FUNKTIONIERT LEIDER NOCH NICHT WEGEN LOSS ODER DATENFORMAT! (SGD)
##DIE GN-METHODE FUNKTIONIERT WAHRSCHEINLICH AUCH NOCH NICHT (noch nicht getestet...)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)

tf.random.set_seed(1)

batch_size = 100
epochs = 2
model_neurons_mnist = [784, 392, 98, 10]


def mnist_data_generator():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = tf.reshape(train_x, [60000, 784])/255
    test_x = tf.reshape(test_x, [10000, 784])/255
    train_y = tf.one_hot(train_y, depth = 10)
    test_y = tf.one_hot(test_y, depth = 10)
    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_data_generator()
#print(train_x[1, :])

def model_loss_mnist(y_true, y_pred):
    return -tf.reduce_mean(tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(tf.nn.softmax(y_pred))), axis=0))
#https://fluxml.ai/Flux.jl/v0.12/models/losses/#Flux.Losses.logitcrossentropy


#print(test_y[1:3, :])
#print(tf.nn.softmax(test_y[1:3, :]))
#print(tf.math.log(tf.nn.softmax(test_y[1:3, :])))
#print(tf.math.multiply(test_y[1:3, :], tf.math.log(tf.nn.softmax(test_y[1:3, :]))))
#print(tf.math.reduce_sum(tf.math.multiply(test_y[1:3, :], tf.math.log(tf.nn.softmax(test_y[1:3, :]))), axis=0))
#print(-tf.reduce_mean(tf.math.reduce_sum(tf.math.multiply(test_y[1:3, :], tf.math.log(tf.nn.softmax(test_y[1:3, :]))), axis=0)))
#print(model_loss_mnist(test_y[1:3, :], test_y[1:3, :]))

input_layer_mnist = tf.keras.Input(shape=(model_neurons_mnist[0]))
layer_1_mnist = tf.keras.layers.Dense(model_neurons_mnist[1], activation='relu')(input_layer_mnist)
layer_2_mnist = tf.keras.layers.Dense(model_neurons_mnist[2], activation='relu')(layer_1_mnist)
layer_3_mnist = tf.keras.layers.Dense(model_neurons_mnist[3])(layer_2_mnist)

model_mnist = tf.keras.Model(input_layer_mnist, layer_3_mnist, name='Model')

model_mnist.compile(loss=model_loss_mnist)
model_mnist.summary()

#print(tf.shape(test_y[1:3,:]))
#print(tf.shape(model_mnist.predict(test_x[1:3,:])))
#print(model_mnist.predict(test_x[1:3,:]))
#print(tf.nn.softmax(model_mnist.predict(test_x[1:3,:])))
#print(tf.math.log(tf.nn.softmax(model_mnist.predict(test_x[1:3,:]))))
print(model_loss_mnist(test_y[0:9999, :], model_mnist.predict(test_x[0:9999,:])))

layer_shape = [(model_neurons_mnist[i], model_neurons_mnist[i+1]) for i in range(np.shape(model_neurons_mnist)[0]-1)]
bias_shape = [(model_neurons_mnist[i+1]) for i in range(np.shape(model_neurons_mnist)[0]-1)]
param_shape = [x for y in zip(layer_shape, bias_shape) for x in y]
n_params = [np.prod(s) for s in param_shape]
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1


def fastmatvec(v, jac, lam):
    return tf.reduce_mean(tf.linalg.matvec(jac, tf.linalg.matvec(jac, v), transpose_a=True), axis=0) + lam * v


def cg_method(jac, x, b, min_steps, precision):  # Martens Werte: min_steps = 10, precision = 0.0005
    r = b - fastmatvec(x, jac, lam)
    d = r
    i, k = 0, min_steps
    # Wie geht das schneller????
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec(d, jac, lam)
        alpha = tf.tensordot(r, r, 1) / tf.tensordot(d, z, 1)
        x = x + alpha * d
        r_new = r - alpha * z
        beta = tf.tensordot(r_new, r_new, 1) / tf.tensordot(r, r, 1)
        d = r_new + beta * d
        r = r_new
        phi_history = np.append(phi_history, np.array(
            - 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1))))
        if i >= k:
            s = (phi_history[-1] - phi_history[-k]) / phi_history[-1]
        else:
            s = k
        i += 1
    return x


# Martens Werte: min_steps = 10, precision = 0.0005
def preconditioned_cg_method(A, x, b, min_steps, precision):
    r = b - fastmatvec(x, A, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    # Wie geht das schneller????
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec(d, A, lam)
        alpha = tf.tensordot(r, y, 1) / tf.tensordot(d, z, 1)
        x = x + alpha * d
        r_new = r - alpha * z
        y_new = r_new / (b ** 2 + lam)
        beta = tf.tensordot(r_new, y_new, 1) / tf.tensordot(r, y, 1)
        d = y_new + beta * d
        r = r_new
        y = y_new
        phi_history = np.append(phi_history, np.array(
            - 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1))))
        if i >= k:
            s = (phi_history[-1] - phi_history[-k]) / phi_history[-1]
        else:
            s = k
        i += 1

    return x


def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model_mnist(x)
        loss = model_loss_mnist(y, y_pred)

    res = y_pred - y
    if model_neurons_mnist[0] == 1:
        res = tf.reshape(res, (batch_size, 1, 1))

    theta = model_mnist.trainable_variables
    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons_mnist[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    grad_obj = tf.squeeze(tf.reduce_mean(tf.matmul(jac, res, transpose_a=True), axis=0))

    update = preconditioned_cg_method(jac, update_old, grad_obj, 5, 0.0005)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model_mnist.set_weights(theta_new)

    impr = loss - model_loss_mnist(y,  model_mnist(x))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec(update, jac, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update



def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model_mnist(x)
        loss = model_loss_mnist(y, y_pred)

    theta = model_mnist.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model_mnist.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(60000 / batch_size)


t = time.time()
for epoch in range(epochs):
    test_loss = model_loss_mnist(test_y, model_mnist.predict(test_x))
    print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, test_loss))

    for i in range(num_updates):

        start = i * batch_size
        end = start + batch_size

#        lam, update_old = train_step_generalized_gauss_newton(
#            train_x[start: end], train_y[start: end], lam, update_old)
        train_step_gradient_descent(train_x[start: end], train_y[start: end], 0.3)

elapsed = time.time() - t
