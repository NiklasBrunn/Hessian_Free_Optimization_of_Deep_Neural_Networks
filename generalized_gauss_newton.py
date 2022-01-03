import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)


train_size = 1000
test_size = 500
batch_size = 100
epochs = 20
model_neurons = [1, 25, 25, 1]
eta = 0.01


def toy_data_generator(size, noise):
    x = tf.random.normal([size, model_neurons[0]])
    y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    return x, y


x_train, y_train = toy_data_generator(train_size, 0.1)
x_test, y_test = toy_data_generator(test_size, 0)


def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


input_layer = tf.keras.Input(shape=(model_neurons[0],))
layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
layer_2 = tf.keras.layers.Dense(model_neurons[2], activation='relu')(layer_1)
layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)

model = tf.keras.Model(input_layer, layer_3, name='Model')

model.compile(loss=model_loss)
model.summary()

layer_shape = [(model_neurons[i], model_neurons[i+1]) for i in range(np.shape(model_neurons)[0]-1)]
bias_shape = [(model_neurons[i+1]) for i in range(np.shape(model_neurons)[0]-1)]
param_shape = [x for y in zip(layer_shape, bias_shape) for x in y]
n_params = [np.prod(s) for s in param_shape]
eye_eps = [tf.eye(n_params[i]) * 1e-5 for i in range(len(n_params))]


def train_step_generalized_gauss_newton(x, y):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)]
    jac = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
           for i, h in enumerate(jac)]
    jac_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_T, jac)]

    update = [tf.constant(eta * np.linalg.solve(G[i] + eye_eps[i], grad[i]))
              for i in range(len(G))]

    update = [tf.reshape(u, s) for u, s in zip(update, param_shape)]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


def train_step_gradient_descent(x, y):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(train_size / batch_size)


for epoch in range(epochs):
    test_loss = model_loss(y_test, model.predict(x_test))
    print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, test_loss))

    for i in range(num_updates):

        start = i * batch_size
        end = start + batch_size

        train_step_generalized_gauss_newton(x_train[start:end], y_train[start:end])


f, ax = plt.subplots(1, 1, figsize=(6, 4))

a = np.linspace(-4, 4, 200)
x = model.predict(a)

ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

ax.plot(a, a**2, label='Ground Truth', c='green')
ax.plot(a, x, label='Prediction', c='blue')

ax.legend(loc='upper left')
plt.show()
