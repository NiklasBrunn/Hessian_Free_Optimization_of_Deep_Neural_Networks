##############################################################
#Hessian-Free Optimization algorithm for MNIST classification:
##############################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)

#################
#hyperparameters:
#################
# here one can set some parameters for the training, e.g. which
# optimization method is used or what kind of Network architecture is used and
# number of epochs, CG-steps, parameter seeds, ...
# After setting these hyperparameters one can run the code and the training of
# the chosen NN will start. Training will be displayed in the console with
# some characteristic values (computation time for each iteration, test loss
# per iteration [if better than before], train loss per epoch, number of wrong
# classified MNIST numbers)

Model_Seed = 17 # seed for the random initialisation of the model parameters.
data_size = 60000
batch_size = 1000 # for the 2nd order optimization method we recommend
#                  a relatively large batchsize, e.g. >= 250 up to 1000.
#                  In our Experiments we used 250 (and 300).
epochs = 250
learningrate_SGD = 0.1 # one can choose a learningrate for SGD optimization.
CG_steps = 3 # we recommend 3 for MNIST, more steps (e.g. 10) result in longer
#              computation time but also the loss will decrease marginally
acc_CG = 0.0005 # accuracy in the CG algorithm (termination criterion).
lam_up = 1.5 # set the amount for lambda updates(1.5 is a good standard choice).

fmv_version = 2 # options are 1, 2, 3 (version 2 and 3 work best!)
#                (for the different versions see below in the code)
train_method = 'fast_CG' # options are: 'SGD', 'fast_CG'
# -> with fast_CG we mean our implementation of the Hessian-Free algorithm
Net = 'Dense' # options are 'Dense', 'CNN' (we used Dense for our experiments).
model_neurons = [784, 500, 10] # number of neurons when choosing Dense


####################
#loading MNIST data:
####################
#loading MNIST data for the Dense-Layer NN ...
def mnist_data_generator_Dense():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.reshape(x_train, [60000, 784])/255
    x_test = tf.reshape(x_test, [10000, 784])/255
    y_train = tf.one_hot(y_train, depth = 10)
    y_test = tf.one_hot(y_test, depth = 10)
    return (x_train[0:data_size, :], y_train[0:data_size, :]), (x_test, y_test)

#loading MNIST data for the CNN ...
def mnist_data_generator_CNN():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((len(x_train), 28, 28, 1)) / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1)) / 255.
    y_train = tf.one_hot(y_train[:data_size], depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    return x_train[:data_size], y_train, x_test, y_test

if Net == 'Dense':
    #DENSE:
    (x_train, y_train), (x_test, y_test) = mnist_data_generator_Dense()

elif Net == 'CNN':
    #CNN:
    x_train, y_train, x_test, y_test = mnist_data_generator_CNN()


######################################
#def. model loss and generating model:
######################################
# multi-classification loss function: logitbinary-crossentropy,
# softmax is contained in the loss function for numerical stability.
def model_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))

#def model_loss(y_true, y_pred):
    #return tf.reduce_mean(-tf.math.reduce_sum(y_true * tf.math.log(tf.nn.softmax(y_pred)),
    #                                          axis=0))
# the above loss function is just a different loss function that works surprisingly well but
# has nothing to do with the actual task


if Net == 'Dense':
    # creating Dense-NN:
    tf.random.set_seed(Model_Seed)

    input_layer = tf.keras.Input(shape=(model_neurons[0]))
    layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2])(layer_1)

    model = tf.keras.Model(input_layer, layer_2, name='Model')

elif Net == 'CNN':
    # creating CNN:
    tf.random.set_seed(Model_Seed)

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3),strides=(2, 2),
                               padding="same", activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                               activation='relu')(x)
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
# Version 1; computation from right to left: (J_Net)' * J_Softmax * J_Net * v
# u := J_Net * v with FAD
# w := J_Softmax * u with FAD
# (J_Net)' * w with BAD
# (slower than version 2 and version 3!)
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


# Version 2; computation from right to left: (J_Net)' * J_Softmax(Net) * v
# u := J_Softmax(Net) * v with FAD
# (J_Net)' * u with BAD
# (fast!)
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

# Version 3; computation from right to left: (J_Softmax(Net))' * J_Net * v
# u := J_Net * v with FAD
# (J_Softmax(Net))' * u with BAD
# (fast!)
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
# preconditioned_cg_method for every other fastmatvec function using FAD and BAD.
def fast_preconditioned_cg_method(v, x_batch, y_batch, b, min_steps, eps):
    if fmv_version == 1:
        A = fastmatvec_V1(x_batch, y_batch, v, lam)
    elif fmv_version == 2:
        A = fastmatvec_V2(x_batch, y_batch, v, lam)
    else:
        A = fastmatvec_V3(x_batch, y_batch, v, lam)

    r = b - A
    M = (b**2 + lam) ** 0.75
    y = r/M
    d = y
    i, s, k = 0, min_steps, min_steps
    phi_history = np.array(0.5 * tf.math.reduce_sum(v*(A - 2.0 * b)))
    r_dot_y = tf.math.reduce_sum(r * y)
    while i <= k or s >= eps*k or phi_history[-1] >= 0:
        k = np.maximum(min_steps, int(i/min_steps))
        if fmv_version == 1:
            z = fastmatvec_V1(x_batch, y_batch, d, lam)
        elif fmv_version == 2:
            z = fastmatvec_V2(x_batch, y_batch, d, lam)
        else:
            z = fastmatvec_V3(x_batch, y_batch, d, lam)
        alpha = r_dot_y / tf.math.reduce_sum(d*z)
        z *= alpha
        v += alpha * d
        A += z
        r -= z
        y -= z / M
        r_dot_y_new = tf.math.reduce_sum(r*y)
        d = y + d * r_dot_y_new / r_dot_y
        r_dot_y = r_dot_y_new
        phi_history = np.append(phi_history,
                                0.5 * tf.math.reduce_sum(v * (A - 2.0 * b)))
        if i >= k:
            s = 1 - phi_history[-k] / phi_history[-1]

        i += 1
    print('CG-iterations for this batch:', i)
    return v, phi_history[-1] - 0.5 * lam * tf.math.reduce_sum(v * v) + 2.0 * tf.math.reduce_sum(v * b)


##############
#optimization:
##############
# optimization using the fast precond._cg_method (with forward- and backward-AD)
def train_step_fast_generalized_gauss_newton(x, y, lam, update_old):
    theta = model.trainable_variables
    with tf.GradientTape() as tape:
        loss = model_loss(y, model(x))
    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                          for i, g in enumerate(grad_obj)], axis=0))

    update, denom = fast_preconditioned_cg_method(update_old, x, y, grad_obj,
                                           CG_steps, acc_CG)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / denom

    #print('rho:', rho)
    if rho > 0.75:
        lam /= lam_up
    elif rho < 0.25:
        lam *= lam_up

    #print('Lambda:', lam)
    return lam, update


# standard SGD optimization:
def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


##########
#training:
##########
num_updates = int(data_size / batch_size)

#t = time.time()
error_old = 100000
traintime = 0  #while-loop over time
#epoch = 0  #while-loop over time
for epoch in range(epochs):  #for-loop over epochs
#while traintime <= 120:   #while-loop over time
#    epoch += 1  #while-loop over time

    train_time = np.zeros(epochs*num_updates)  #while-loop over time
#    error_history_test = np.zeros(epochs*num_updates)  #while-loop over time
    #error_history_train = np.zeros(epochs*num_updates)  #while-loop over time
#    epochs_vec = np.zeros(epochs*num_updates)  #while-loop over time

    perm = np.random.permutation(data_size)
    x_train = np.take(x_train, perm, axis = 0)
    y_train = np.take(y_train, perm, axis = 0)
    train_loss = np.array(model_loss(y_train, model.predict(x_train)))
    print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, train_loss))
    for i in range(num_updates):
        #error_new_train = model_loss(y_train, model.predict(x_train))  #while-loop over time
    #    error_new_test = model_loss(y_test, model.predict(x_test))  #while-loop over time

        error_new = np.array(model_loss(y_test, model.predict(x_test)))
        if error_new < error_old:
            print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                       1).zfill(len(str(epochs))), epochs, error_new))
            error_old = error_new

        start = i * batch_size
        end = start + batch_size

        if train_method == 'SGD':
            #standard SGD.
            t = time.time()
            train_step_gradient_descent(x_train[start: end], y_train[start: end],
                                        learningrate_SGD)
            elapsed = time.time() - t

            print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, elapsed))
            wrong_classified = np.sum(np.where(np.argmax(y_test, axis=1) -
                                               np.argmax(tf.nn.softmax(model.predict(x_test)),
                                                         axis=1) !=0, 1, 0))
            print('wrong classified test-MNIST numbers:', int(wrong_classified))

        else:
            t = time.time()
            lam, update_old = train_step_fast_generalized_gauss_newton(
                x_train[start: end], y_train[start: end], lam, update_old)
            elapsed = time.time() - t

            print('estimated time for one batch update in epoch {}/{}: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, elapsed))
            wrong_classified = np.sum(np.where(np.argmax(y_test, axis=1) -
                                               np.argmax(tf.nn.softmax(model.predict(x_test)),
                                               axis=1) !=0, 1, 0))
            print('wrong classified test-MNIST numbers:', int(wrong_classified))



        if i == 0:  #while-loop over time
            train_time[(epoch-1)*num_updates + i] = elapsed  #while-loop over time
        else:  #while-loop over time
            train_time[(epoch-1)*num_updates + i] = train_time[i - 1] + elapsed  #while-loop over time

        #error_history_train[(epoch-1)*num_updates+i] = error_new_train  #while-loop over time
#        error_history_test[(epoch-1)*num_updates+i] = error_new_test  #while-loop over time
#        epochs_vec[(epoch-1)*num_updates+i] = epoch  #while-loop over time
        traintime += elapsed  #while-loop over time
        print('estimated time for the train steps:', traintime)
#np.savetxt('<insert path>//train_time_MNIST_SGD_100_01.npy',
#            train_time)  #while-loop over time
#np.savetxt('<insert path>//error_history_train_MNIST_SGD_100_01.npy',
#                 error_history_train)  #while-loop over time
#np.savetxt('<insert path>//error_history_test_MNIST_SGD_100_01.npy',
#                 error_history_test)  #while-loop over time
#np.savetxt('<insert path>//epochs_MNIST_SGD_100_01.npy',
#                 epochs_vec)  #while-loop over time

#elapsed = time.time() - t
print('test accuracy:', (10000 - wrong_classified) / 10000)
