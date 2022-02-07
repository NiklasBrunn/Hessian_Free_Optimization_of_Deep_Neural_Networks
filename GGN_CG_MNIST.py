import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)

################
#hyperparameter:
################
Data_Seed = 13
Model_Seed = 13
data_size = 60000
batch_size = 250
epochs = 1
CG_steps = 3
model_neurons = [784, 800, 10]
fmv_version = 2 # options are 1, 2, 3 (gibt an welche fastmatvec Funktion benutzt wird)
train_method = 'CG_R_Op' # options are: 'SGD', 'CG_naiv', 'CG_R_Op'
Net = 'Dense' # options are 'Dense', 'CNN'


####################
#loading MNIST data:
####################
def mnist_data_generator_Dense():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.reshape(x_train, [60000, 784])/255
    x_test = tf.reshape(x_test, [10000, 784])/255
    y_train = tf.one_hot(y_train, depth = 10)
    y_test = tf.one_hot(y_test, depth = 10)
    return (x_train[0:data_size, :], y_train[0:data_size, :]), (x_test, y_test)

def mnist_data_generator_CNN():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((len(x_train), 28, 28, 1)) / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1)) / 255.
    y_train = tf.one_hot(y_train[:data_size], depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    return x_train[:data_size], y_train, x_test, y_test

if Net == 'Dense':
    #DENSE:
    tf.random.set_seed(Data_Seed)
    (x_train, y_train), (x_test, y_test) = mnist_data_generator_Dense()

elif Net == 'CNN':
    #CNN:
    tf.random.set_seed(Data_Seed)
    x_train, y_train, x_test, y_test = mnist_data_generator_CNN()


######################################
#def. model loss and generating model:
######################################
def model_loss(y_true, y_pred):
    #return tf.reduce_mean(-tf.math.reduce_sum(y_true * tf.math.log(tf.nn.softmax(y_pred)), axis=0)) #Loss hiermit viel besser, obwohl so nicht gedacht?!?
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))

if Net == 'Dense':
    tf.random.set_seed(Model_Seed)

    input_layer = tf.keras.Input(shape=(model_neurons[0]))
    layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2])(layer_1)

    model = tf.keras.Model(input_layer, layer_2, name='Model')

elif Net == 'CNN':
    tf.random.set_seed(Model_Seed)

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3),
                               strides=(2, 2), padding="same", activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs, outputs)


model.compile(loss=model_loss)
model.summary()

param_shape = [np.shape(t) for t in model.trainable_variables]
n_params = [np.prod(s) for s in param_shape]
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1


#######################################
#Fast matrix-vector-products functions:
#######################################
# Naive Version; Berechnung von rechts nach links der Matrix-Vektor-Produkte,
# mit vorab berechneten Matrizen (sehr langsam!)
def fastmatvec_naiv(v, jac_net, jac_softmax, lam):
    prod1 = tf.linalg.matvec(jac_net, v)
    prod2 = tf.linalg.matvec(jac_softmax, prod1)
    prod3 = tf.linalg.matvec(jac_net, prod2, transpose_a=True)
    return tf.reduce_mean(prod3, axis=0) + lam * v


# Version 1; Berechnung von rechts nach links: (J_Net)' * J_Softmax * J_Net * v
# u := J_Net * v mit Forwarddiff
# w := J_Softmax * u mit Forwarddiff
# (J_Net)' * w mit Backwarddiff
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


# Version 2; Berechnung von rechts nach links: (J_Net)' * J_Softmax(Net) * v
# u := J_Softmax(Net) * v mit Forwarddiff
# (J_Net)' * u mit Backwarddiff
def fastmatvec_V2(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]

    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
            y_pred = model(x_batch)
            akt_out = tf.nn.softmax(y_pred)
        Jsoft_v = acc.jvp(akt_out)
    GGN_times_v = tape.gradient(y_pred, model.trainable_variables,
                             output_gradients=tf.stop_gradient(Jsoft_v))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(GGN_times_v)], axis=0))
    return v_new / batch_size + lam * v

# Version 3; Berechnung von rechts nach links: (J_Softmax(Net))' * J_Net * v
# u := J_Net * v mit Forwarddiff
# (J_Softmax(Net))' * u mit Backwarddiff
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


########################
#CG-Algorithm functions:
########################
# preconditioned_cg_method for fastmatvec_naiv
def preconditioned_cg_method(A, B, x, b, min_steps, precision):
    r = b - fastmatvec_naiv(x, A, B, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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

# preconditioned_cg_method for every other fastmatvec function using the R-Op.
def preconditioned_cg_method_R_Op(v, x_batch, y_batch, b, min_steps, precision):
    if fmv_version == 1:
        r = b - fastmatvec_V1(x_batch, y_batch, v, lam)
    elif fmv_version == 2:
        r = b - fastmatvec_V2(x_batch, y_batch, v, lam)
    else:
        r = b - fastmatvec_V3(x_batch, y_batch, v, lam)

    y = r / (b ** 2 + lam)
    d = y
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
        k = np.maximum(min_steps, int(i/min_steps))
        if fmv_version == 1:
            z = fastmatvec_V1(x_batch, y_batch, d, lam)
        elif fmv_version == 2:
            z = fastmatvec_V2(x_batch, y_batch, d, lam)
        else:
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


##############
#optimization:
##############
# optimization using the precond._cg_method with R_Op
def train_step_generalized_gauss_newton_R_Op(x, y, lam, update_old):
    theta = model.trainable_variables
    with tf.GradientTape() as tape:
        loss = model_loss(y, model(x))
    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method_R_Op(update_old, x, y, grad_obj, CG_steps, 0.0005)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    if fmv_version == 1:
        rho = impr / (tf.tensordot(grad_obj, update, 1) +
                      tf.tensordot(update, fastmatvec_V1(x, y, update, 0), 1))
    elif fmv_version == 2:
        rho = impr / (tf.tensordot(grad_obj, update, 1) +
                      tf.tensordot(update, fastmatvec_V2(x, y, update, 0), 1))
    else:
        rho = impr / (tf.tensordot(grad_obj, update, 1) +
                      tf.tensordot(update, fastmatvec_V3(x, y, update, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update


# optimization using the precond._cg_method with naiv implementation of fastmatvec
# slow ...
def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        akt_out = tf.nn.softmax(y_pred)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables

    jac_softmax = tape.batch_jacobian(akt_out, y_pred)

    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method(jac, jac_softmax, update_old, grad_obj, CG_steps, 0.0005)

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


# standard SGD optimization
def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(data_size / batch_size)


##########
#training:
##########
#t = time.time()
error_old = 100000
for epoch in range(epochs):
    train_loss = np.array(model_loss(y_train, model.predict(x_train)))
    print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, train_loss))
    for i in range(num_updates):
        error_new = np.array(model_loss(y_test, model.predict(x_test)))
        if error_new < error_old:
            print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                       1).zfill(len(str(epochs))), epochs, error_new))
            error_old = error_new

        start = i * batch_size
        end = start + batch_size

        if train_method == 'CG_naiv':
            #fastmatvec naiv (slow ...)
            t = time.time()
            lam, update_old = train_step_generalized_gauss_newton(
                x_train[start: end], y_train[start: end], lam, update_old)
            elapsed = time.time() - t
            print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                              1).zfill(len(str(epochs))), epochs, elapsed))
            wrong_classified = np.sum(np.where(np.argmax(y_test, axis=1) - np.argmax(tf.nn.softmax(model.predict(x_test)), axis=1) !=0, 1, 0))
            print('falsch klassifizierte Test-MNIST-Zahlen:', int(wrong_classified))

        elif train_method == 'SGD':
            #standard SGD.
            t = time.time()
            train_step_gradient_descent(x_train[start: end], y_train[start: end], 0.01)
            elapsed = time.time() - t
            print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, elapsed))
            wrong_classified = np.sum(np.where(np.argmax(y_test, axis=1) - np.argmax(tf.nn.softmax(model.predict(x_test)), axis=1) !=0, 1, 0))
            print('falsch klassifizierte Test-MNIST-Zahlen:', int(wrong_classified))

        else:
            #fastmatvec with R_Op.
            t = time.time()
            lam, update_old = train_step_generalized_gauss_newton_R_Op(
                x_train[start: end], y_train[start: end], lam, update_old)
            elapsed = time.time() - t
            print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, elapsed))
            wrong_classified = np.sum(np.where(np.argmax(y_test, axis=1) - np.argmax(tf.nn.softmax(model.predict(x_test)), axis=1) !=0, 1, 0))
            print('falsch klassifizierte Test-MNIST-Zahlen:', int(wrong_classified))


#elapsed = time.time() - t
print('test accuracy:', (10000 - wrong_classified) / 10000)

########################
#plots and informations:
########################
