import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)

tf.random.set_seed(1)

data_size = 60000
batch_size = 200
epochs = 10
model_neurons = [784, 800, 10]

def mnist_data_generator():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = tf.reshape(train_x, [60000, 784])/255
    test_x = tf.reshape(test_x, [10000, 784])/255
    train_y = tf.one_hot(train_y, depth = 10)
    test_y = tf.one_hot(test_y, depth = 10)
    return (train_x[0:data_size, :], train_y[0:data_size, :]), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_data_generator()


def model_loss(y_true, y_pred):
    #return tf.reduce_mean(-tf.math.reduce_sum(y_true * tf.math.log(tf.nn.softmax(y_pred)), axis=0)) #Loss hiermit viel besser, obwohl so nicht gedacht?!?
    #return tf.reduce_mean(-tf.math.reduce_sum(y_true * tf.math.log(tf.nn.softmax(y_pred)), axis=1))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))


def fastmatvec_naiv(v, jac_net, jac_softmax, lam):
    prod1 = tf.linalg.matvec(jac_net, v)
    prod2 = tf.linalg.matvec(jac_softmax, prod1)
    prod3 = tf.linalg.matvec(jac_net, prod2, transpose_a=True)
    return tf.reduce_mean(prod3, axis=0) + lam * v


def fastmatvec_V1(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]
    with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
        y_pred = model(x_batch)
    Jacnet_times_vec = acc.jvp(y_pred)
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        with tf.autodiff.ForwardAccumulator(y_pred, Jacnet_times_vec) as acc1:
            akt_out = tf.nn.softmax(y_pred)
        Jac_softmax_times_vec = acc1.jvp(akt_out)
    GGN_times_v = tape.gradient(y_pred, model.trainable_variables,
                             output_gradients=tf.stop_gradient(Jac_softmax_times_vec))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(GGN_times_v)], axis=0))

    return v_new / batch_size + lam * v


def fastmatvec_V2(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]

    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
            y_pred = model(x_batch)
            akt_out = tf.nn.softmax(y_pred)
        Jsoft_v = acc.jvp(akt_out) # Jsoft_v = Jacobi_softmax(bzgl. Netzwerkparams) * v = ...
        # ... = Jacobi_softmax(bzgl. Netzwerkoutp.) * Jacobi_netz(y_pred = model(x_batch)) * v
    GGN_times_v = tape.gradient(y_pred, model.trainable_variables,
                             output_gradients=tf.stop_gradient(Jsoft_v))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(GGN_times_v)], axis=0))

    return v_new / batch_size + lam * v


def fastmatvec_V3(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]

    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
            y_pred = model(x_batch)
        Jnet_v = acc.jvp(y_pred)
        akt_out = tf.nn.softmax(y_pred)
    GGN_times_v = tape.gradient(akt_out, model.trainable_variables,
                             output_gradients=tf.stop_gradient(Jnet_v))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(GGN_times_v)], axis=0))

    return v_new / batch_size + lam * v




input_layer = tf.keras.Input(shape=(model_neurons[0]))
layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
layer_2 = tf.keras.layers.Dense(model_neurons[2])(layer_1)

model = tf.keras.Model(input_layer, layer_2, name='Model')

model.compile(loss=model_loss)
model.summary()


layer_shape = [(model_neurons[i], model_neurons[i+1]) for i in range(np.shape(model_neurons)[0]-1)]
bias_shape = [(model_neurons[i+1]) for i in range(np.shape(model_neurons)[0]-1)]
param_shape = [x for y in zip(layer_shape, bias_shape) for x in y]
n_params = [np.prod(s) for s in param_shape]
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1


# Martens Werte: min_steps = 10, precision = 0.0005
def cg_method(jac, jac_softmax, x, b, min_steps, precision):
    r = b - fastmatvec_naiv(x, jac, jac_softmax, lam)
    d = r
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec_naiv(d, jac, jac_softmax, lam)
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
def preconditioned_cg_method(A, B, x, b, min_steps, precision):
    r = b - fastmatvec_naiv(x, A, B, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec_naiv(d, A, B, lam)
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


def preconditioned_cg_method_R_Op(v, x_batch, y_batch, b, min_steps, precision):
    #r = b - fastmatvec_V1(x_batch, y_batch, v, lam)
    #r = b - fastmatvec_V2(x_batch, y_batch, v, lam)
    r = b - fastmatvec_V3(x_batch, y_batch, v, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        #z = fastmatvec_V1(x_batch, y_batch, d, lam)
        #z = fastmatvec_V2(x_batch, y_batch, d, lam)
        z = fastmatvec_V3(x_batch, y_batch, d, lam)
        alpha = tf.tensordot(r, y, 1) / tf.tensordot(d, z, 1)
        v = v + alpha * d
        r_new = r - alpha * z
        y_new = r_new / (b ** 2 + lam)
        beta = tf.tensordot(r_new, y_new, 1) / tf.tensordot(r, y, 1)
        d = y_new + beta * d
        r = r_new
        y = y_new
        phi_history = np.append(phi_history, np.array(
            - 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))))
        if i >= k:
            s = (phi_history[-1] - phi_history[-k]) / phi_history[-1]
        else:
            s = k
        i += 1
    return v


def train_step_generalized_gauss_newton_R_Op(x, y, lam, update_old):
    theta = model.trainable_variables
    with tf.GradientTape() as tape:
        loss = model_loss(y, model(x))
    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method_R_Op(update_old, x, y, grad_obj, 10, 0.0005)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

#    rho = impr / (tf.tensordot(grad_obj, update, 1) +
#                  tf.tensordot(update, fastmatvec_V1(x, y, update, 0), 1))

#    rho = impr / (tf.tensordot(grad_obj, update, 1) +
#                  tf.tensordot(update, fastmatvec_V2(x, y, update, 0), 1))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec_V3(x, y, update, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update



def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        akt_out = tf.nn.softmax(y_pred)
        loss = model_loss(y, y_pred)

    #res = y_pred - y
    #if model_neurons[0] == 1:
    #    res = tf.reshape(res, (batch_size, 1, 1))
    #res = tf.reshape(res, (batch_size, 10, 1))

    theta = model.trainable_variables

    jac_softmax = tape.batch_jacobian(akt_out, y_pred)


    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    #jac_net = tf.reduce_mean(jac, axis=0)

    #grad_obj = tf.squeeze(tf.reduce_mean(tf.matmul(jac, res, transpose_a=True), axis=0))
    #grad_obj = tf.squeeze(tf.reduce_mean([tf.matmul(tf.transpose(jac, perm=[0,2,1])[i,:,:],res[i,:,:]) for i in range(batch_size)], axis=0)) #braucht tuuuuurbo lange zum berechnen :O

    update = preconditioned_cg_method(jac, jac_softmax, update_old, grad_obj, 10, 0.0005)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec_naiv(update, jac, jac_softmax, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update



def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(data_size / batch_size)


#t = time.time()
for epoch in range(epochs):
    for i in range(num_updates):
        test_loss = np.array(model_loss(test_y, model.predict(test_x)))
        print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))), epochs, test_loss))


        start = i * batch_size
        end = start + batch_size

#        t = time.time()
#        lam, update_old = train_step_generalized_gauss_newton(
#            train_x[start: end], train_y[start: end], lam, update_old)
#        elapsed = time.time() - t
#        print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
#                                                           1).zfill(len(str(epochs))), epochs, elapsed))

        t = time.time()
        lam, update_old = train_step_generalized_gauss_newton_R_Op(
            train_x[start: end], train_y[start: end], lam, update_old)
        elapsed = time.time() - t
        print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                           1).zfill(len(str(epochs))), epochs, elapsed))


#        t = time.time()
#        train_step_gradient_descent(train_x[start: end], train_y[start: end], 0.01)
#        elapsed = time.time() - t
#        print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
#                                                           1).zfill(len(str(epochs))), epochs, elapsed))


#elapsed = time.time() - t

wrong_classified = np.sum(np.where(np.argmax(test_y, axis=1) - np.argmax(tf.nn.softmax(model.predict(test_x)), axis=1) !=0, 1, 0))
print('falsch klassifizierte Test-MNIST-Zahlen:', int(wrong_classified))
print('test accuracy:', (10000 - wrong_classified) / 10000)
