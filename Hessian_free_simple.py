###########################################################################
#Hessian-Free Optimization algorithm for simple simulated sin- or x^2-data:
###########################################################################

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
# here one can set some parameters for the training, e.g.
# number of epochs, CG-steps, parameter seeds, ...
# After setting these hyperparameters one can run the code and the training
# will start. Training will be displayed in the console with some
# characteristic values (computation time for each epoch, test loss and
# train loss per epoch.
# Note: some plots are not yet optimized.

Data_Seed = 1 # Seed for generating the train- and test-data.
Model_Seed = 1 # Seed for the initialisation of the NN parameters

train_size = 10000 # number of observations for training.
test_size = 1000 # number of observations for testing.

batch_size_SGD = 1000
batch_size_GN = 1000
epochs_SGD = 1000 #1000
epochs_GN = 150 #150

CG_steps = 3 # minimum number of steps in CG (max. is the dim. of the params.).
acc_CG = 0.0005 # accuracy in the CG algorithm (termination criterion).
lam_up = 1.25 # set the amount for lambda updates(1.5 is a good standard choice).
learningrate_SGD = 0.1
model_neurons = [1, 20, 20, 1] # NN architecture (Layer dimensions).

SGD_allowed = True # NN training with SGD only if SGD_allowed = True.
GN_allowed = True # NN training with the Hessian-Free method only if
#                   GN_allowed = True.

sim_data = 'sin' # choose the simulated dataset: 'sin', 'square'.
plotting = True # showing plots only if plotting is set to True!


#########################################
#Data generation (optional mit Outliern):
#########################################
# function for generating x^2 data with noise:
def toy_data_generator(size, noise):
    if sim_data == 'sin':
        #x = 2 * tf.random.normal([size, model_neurons[0]])
        x = np.random.uniform(-1, 1, [size, model_neurons[0]]) # easy sin
        #y = np.sin(x) + noise * tf.random.normal([size, model_neurons[0]])
        y = np.sin(10*x) + noise * tf.random.normal([size, model_neurons[0]]) # easy sin
    elif sim_data == 'square':
        x = tf.random.normal([size, model_neurons[0]])
        y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    return x, y

#generating train- and test-data:
tf.random.set_seed(Data_Seed)
x_train, y_train = toy_data_generator(train_size, 0.1)
x_test, y_test = toy_data_generator(test_size, 0)


######################################
#def. model loss and generating model:
######################################
# regrassion-model-loss function (MSE)
# (previous activation function is the identity function)
def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


# generating the NN with Dense-Layers:
tf.random.set_seed(Model_Seed)

if sim_data == 'sin':
    input_layer = tf.keras.Input(shape=(model_neurons[0],))
    layer_1 = tf.keras.layers.Dense(model_neurons[1],
                                    activation='sigmoid')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2],
                                    activation='sigmoid')(layer_1)
    layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)
elif sim_data == 'square':
    input_layer = tf.keras.Input(shape=(model_neurons[0],))
    layer_1 = tf.keras.layers.Dense(model_neurons[1],
                                    activation='relu')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2],
                                    activation='relu')(layer_1)
    layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)

model = tf.keras.Model(input_layer, layer_3, name='Model')

model.compile(loss=model_loss)
model.summary()


layer_shape = [(model_neurons[i], model_neurons[i+1])
               for i in range(np.shape(model_neurons)[0]-1)]
bias_shape = [(model_neurons[i+1])
              for i in range(np.shape(model_neurons)[0]-1)]
param_shape = [x for y in zip(layer_shape, bias_shape) for x in y]
n_params = [np.prod(s) for s in param_shape]
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1


######################################
#Fast matrix-vector-products function:
######################################
# computation from right to left: (J_Net)'* J_Net * v
# u := J_Net * v with FAD
# (J_Net)'* u with BAD
def fastmatvec(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]

    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
            y_pred = model(x_batch)
        forward = acc.jvp(y_pred)
    backward = tape.gradient(y_pred, model.trainable_variables,
                             output_gradients=tf.stop_gradient(forward))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(backward)], axis=0))

    return v_new/batch_size_GN + lam * v


#######################
#CG-Algorithm function:
#######################
def fast_preconditioned_cg_method(x_batch, y_batch, v, b, min_steps, eps):
    A = fastmatvec(x_batch, y_batch, v, lam)
    r = b - A
    M = (b**2 + lam) ** 0.75
    y = r/M
    d = y
    i, s, k = 0, min_steps, min_steps
    phi_history = np.array(0.5 * tf.math.reduce_sum(v*(A - 2.0 * b)))
    r_dot_y = tf.math.reduce_sum(r * y)
    while i <= k or s >= eps*k or phi_history[-1] >= 0:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec(x_batch, y_batch, d, lam)
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
    #print('CG-iterations for this batch:', i)
    return v, phi_history[-1] - 0.5 * lam * tf.math.reduce_sum(v * v) + 2.0 * tf.math.reduce_sum(v * b)


##############
#optimization:
##############
# optimization using the fast precond._cg_method (with forward- and backward-AD)
def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables

    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                         for i, g in enumerate(grad_obj)],
                                        axis=0))

    update, denom = fast_preconditioned_cg_method(x, y, update_old, grad_obj,
                                               CG_steps, acc_CG)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / denom
    #print('Rho:', rho)

    if rho > 0.75:
        lam /= lam_up
    elif rho < 0.25:
        lam *= lam_up
    #print('Lam:', lam)
    return lam, update


# standard SGD optimization:
def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


##########
#training:
##########
# generating plots:
f, ((ax0, ax1, ax3, ax5), (ax7, ax2, ax4, ax6)) = plt.subplots(2, 4,
                                                               figsize=(18, 8))

if sim_data == 'sin':
    #a = np.linspace(-4, 4, 250) # easy sin
    #ax0.plot(a, np.sin(a), label='Ground Truth', c='green', linewidth=2) # easy sin
    #ax0.set_ylim(-2.2, 2.2) # easy sin
    #ax0.set_xlim(-5, 5) # easy sin
    a = np.linspace(-1, 1, 250)
    ax0.plot(a, np.sin(10*a), label='Ground Truth', c='green', linewidth=1)
    ax0.set_ylim(-2.2, 2.2)
    ax0.set_xlim(-1.01, 1.01)
elif sim_data == 'square':
    a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
    ax0.plot(a, a**2, label='Ground Truth', c='green', linewidth=2)
    ax0.set_ylim(-0.6, 10)
    ax0.set_xlim(-np.sqrt(10), np.sqrt(10))

ax0.scatter(x_train, y_train, label='Train Data', c='red', s=0.05)

#SGD-training:
epochs = epochs_SGD
num_updates = int(train_size / batch_size_SGD)

test_loss_vec_SGD = np.zeros(epochs)
train_loss_vec_SGD = np.zeros(epochs)
epoch_vec_SGD = [i for i in range(epochs)]
time_vec_SGD = np.zeros(epochs)

#train_time_SGD = np.zeros(epochs*num_updates)
#error_history_test_SGD = np.zeros(epochs*num_updates)
#error_history_train_SGD = np.zeros(epochs*num_updates)
#epochs_SGD = np.zeros(epochs*num_updates)

if SGD_allowed == True:
    for epoch in range(epochs):
        perm = np.random.permutation(train_size)
        x_train = np.take(x_train, perm, axis = 0)
        y_train = np.take(y_train, perm, axis = 0)
        train_loss = model_loss(y_train, model.predict(x_train))
        print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))),
                                                                epochs, train_loss))
        test_loss = model_loss(y_test, model.predict(x_test))
        print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))),
                                                               epochs, test_loss))

        test_loss_vec_SGD[epoch] = test_loss
        train_loss_vec_SGD[epoch] = train_loss

        t = time.time()
        for i in range(num_updates):

            #error_new_train_SGD = model_loss(y_train, model.predict(x_train))
            #error_new_test_SGD = model_loss(y_test, model.predict(x_test))

            start = i * batch_size_SGD
            end = start + batch_size_SGD

            #s = time.time()
            train_step_gradient_descent(x_train[start: end],
                                        y_train[start: end], learningrate_SGD)
            #elapseds = time.time() - s
            #print('estimated time for the batch-update step:', elapseds, 'sec')

            #train_time_SGD[epoch*num_updates+i] = elapseds
            #error_history_train_SGD[epoch*num_updates+i] = error_new_train_SGD
            #error_history_test_SGD[epoch*num_updates+i] = error_new_test_SGD
            #epochs_SGD[epoch*num_updates+i] = epoch

        elapsed = time.time() - t
        print('estimated time for the whole epoch:', elapsed, 'sec')
        if epoch == 0:
            time_vec_SGD[epoch] = elapsed
        else:
            time_vec_SGD[epoch] = time_vec_SGD[epoch - 1] + elapsed


    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='{}{}{}'.format('Prediction SGD(lr=', learningrate_SGD, ')'),
             c='blue', linewidth=1.2)

#np.savetxt('<insert path>//train_time_SGD.npy',
#          train_time_SGD)
#np.savetxt('<insert path>//error_history_train_SGD.npy',
#          error_history_train_SGD)
#np.savetxt('<insert path>//error_history_test_SGD.npy',
#           error_history_test_SGD)
#np.savetxt('<insert path>//epochs_SGD.npy',
#           epochs_SGD)


# GN-TRAINING:
epochs = epochs_GN
# redefining the untrained NN:
# for a fair comparison we use the same seed for the initialisation of the
# NN parameters than above for the SGD-training.
tf.random.set_seed(Model_Seed)

if sim_data == 'sin':
    input_layer = tf.keras.Input(shape=(model_neurons[0],))
    layer_1 = tf.keras.layers.Dense(model_neurons[1],
                                    activation='sigmoid')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2],
                                    activation='sigmoid')(layer_1)
    layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)
elif sim_data == 'square':
    input_layer = tf.keras.Input(shape=(model_neurons[0],))
    layer_1 = tf.keras.layers.Dense(model_neurons[1],
                                    activation='relu')(input_layer)
    layer_2 = tf.keras.layers.Dense(model_neurons[2],
                                    activation='relu')(layer_1)
    layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)

model = tf.keras.Model(input_layer, layer_3, name='Model')

model.compile(loss=model_loss)
model.summary()


test_loss_vec_GN = np.zeros(epochs)
train_loss_vec_GN = np.zeros(epochs)
epoch_vec_GN = [i for i in range(epochs)]
time_vec_GN = np.zeros(epochs)
lam_vec_GN = np.zeros(epochs)

#train_time_GN = np.zeros(epochs*num_updates)
#error_history_test_GN = np.zeros(epochs*num_updates)
#error_history_train_GN = np.zeros(epochs*num_updates)
#epochs_GN = np.zeros(epochs*num_updates)

if GN_allowed == True:
    num_updates = int(train_size / batch_size_GN)
    for epoch in range(epochs):
        perm = np.random.permutation(train_size)
        x_train = np.take(x_train, perm, axis = 0)
        y_train = np.take(y_train, perm, axis = 0)
        train_loss = model_loss(y_train, model.predict(x_train))
        print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))),
                                                                epochs, train_loss))
        test_loss = model_loss(y_test, model.predict(x_test))
        print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))),
                                                               epochs, test_loss))

        test_loss_vec_GN[epoch] = test_loss
        train_loss_vec_GN[epoch] = train_loss
        lam_vec_GN[epoch] = lam

        t = time.time()
        for i in range(num_updates):

            #error_new_train_GN = model_loss(y_train, model.predict(x_train))
            #error_new_test_GN = model_loss(y_test, model.predict(x_test))

            start = i * batch_size_GN
            end = start + batch_size_GN

            #s = time.time()
            lam, update_old = train_step_generalized_gauss_newton(
                x_train[start: end], y_train[start: end], lam, update_old)
            #elapseds = time.time() - s
            #print('estimated time for the batch-update step:', elapseds, 'sec')

            #train_time_GN[epoch*num_updates+i] = elapseds
            #error_history_train_GN[epoch*num_updates+i] = error_new_train_SGD
            #error_history_test_GN[epoch*num_updates+i] = error_new_test_SGD
            #epochs_GN[epoch*num_updates+i] = epoch

        elapsed = time.time() - t
        print('estimated time for the whole epoch:', elapsed, 'sec')
        if epoch == 0:
            time_vec_GN[epoch] = elapsed
        else:
            time_vec_GN[epoch] = time_vec_GN[epoch - 1] + elapsed

    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='{}{}{}'.format('Prediction Hessian-Free(CG=', CG_steps, ')'),
             c='orange', linewidth=1.2)

#np.savetxt('<insert path>//train_time_SGD.npy',
#            train_time_SGD)
#np.savetxt('<insert path>//error_history_train_SGD.npy',
#           error_history_train_SGD)
#np.savetxt('<insert path>//error_history_test_SGD.npy',
#           error_history_test_SGD)
#np.savetxt('<insert path>//epochs_SGD.npy',
#           epochs_SGD)


ax0.set_title('Data and Predictions')
ax0.legend(loc='upper right', prop={'size': 6})


#######
#plots:
#######
#Train_loss_epochs_plot:
if SGD_allowed == True:
    ax1.plot(epoch_vec_SGD, train_loss_vec_SGD, 'r',
             label='{}{}{}'.format('SGD(lr=', learningrate_SGD, ')'),
             linewidth=0.8)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train-Loss')
ax1.set_title('Train-Loss per Epochs:')
#ax1.set_ylim(-0.005, 0.1)
ax1.set_yscale('log')
ax1.set_xlim(0.5, min(epochs_SGD, epochs_GN))

if GN_allowed == True:
    ax1.plot(epoch_vec_GN, train_loss_vec_GN, 'b',
             label='{}{}{}'.format('Hessian-Free(CG=', CG_steps, ')'),
             linewidth=0.8)

ax1.legend(loc='upper right', prop={'size': 6})

#Test_loss_epochs_plot:
if SGD_allowed == True:
    ax2.plot(epoch_vec_SGD, test_loss_vec_SGD, 'r',
             label='{}{}{}'.format('SGD(lr=', learningrate_SGD, ')'),
             linewidth=0.8)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Test-Loss')
ax2.set_title('Test-Loss per Epochs:')
#ax2.set_ylim(-0.005, 0.1)
ax2.set_yscale('log')
ax2.set_xlim(0.5, min(epochs_SGD, epochs_GN))

if GN_allowed == True:
    ax2.plot(epoch_vec_GN, test_loss_vec_GN, 'b',
             label='{}{}{}'.format('Hessian-Free(CG=', CG_steps, ')'),
             linewidth=0.8)

ax2.legend(loc='upper right', prop={'size': 6})

#Train_loss_time_plot:
if SGD_allowed == True:
    ax3.plot(time_vec_SGD, train_loss_vec_SGD, 'r',
             label='{}{}{}'.format('SGD(lr=', learningrate_SGD, ')'),
             linewidth=0.8)
ax3.set_xlabel('Time (in seconds)')
ax3.set_ylabel('Train-Loss')
ax3.set_title('Train-Loss per Time:')
#ax3.set_ylim(-0.005, 0.05)
ax3.set_yscale('log')
ax3.set_xlim(-0.005, min(time_vec_SGD[-1], time_vec_GN[-1]))

if GN_allowed == True:
    ax3.plot(time_vec_GN, train_loss_vec_GN, 'b',
             label='{}{}{}'.format('Hessian-Free(CG=', CG_steps, ')'),
             linewidth=0.8)

ax3.legend(loc='upper right', prop={'size': 6})

#Test_loss_time_plot:
if SGD_allowed == True:
    ax4.plot(time_vec_SGD, test_loss_vec_SGD, 'r',
             label='{}{}{}'.format('SGD(lr=', learningrate_SGD, ')'),
             linewidth=0.8)
ax4.set_xlabel('Time (in seconds)')
ax4.set_ylabel('Test-Loss')
ax4.set_title('Test-Loss per Time:')
#ax4.set_ylim(-0.005, 0.05)
ax4.set_yscale('log')
ax4.set_xlim(-0.005, min(time_vec_SGD[-1], time_vec_GN[-1]))

if GN_allowed == True:
    ax4.plot(time_vec_GN, test_loss_vec_GN, 'b',
             label='{}{}{}'.format('Hessian-Free(CG=', CG_steps, ')'),
             linewidth=0.8)

ax4.legend(loc='upper right', prop={'size': 6})

#Loss per Epochs GN:
if GN_allowed == True:
    ax5.plot(epoch_vec_GN, train_loss_vec_GN, 'r',
             label='{}{}{}'.format('Hessian-Free train(CG=', CG_steps, ')'),
             linewidth=1.2)
ax5.set_xlabel('Epochs')
ax5.set_ylabel('Loss')
ax5.set_title('Loss per Epochs (Hessian-Free)')

if GN_allowed == True:
    ax5.plot(epoch_vec_GN, test_loss_vec_GN, 'b',
             label='{}{}{}'.format('Hessian-Free test(CG=', CG_steps, ')'),
             linewidth=1.2)

ax5.legend(loc='upper right', prop={'size': 6})

#Loss per Epochs SGD:
if SGD_allowed == True:
    ax6.plot(epoch_vec_SGD, train_loss_vec_SGD, 'r',
             label='{}{}{}'.format('SGD train(lr=', learningrate_SGD, ')'),
             linewidth=1.2)
ax6.set_xlabel('Epochs')
ax6.set_ylabel('Loss')
ax6.set_title('Loss per Epochs (SGD)')

if SGD_allowed == True:
    ax6.plot(epoch_vec_SGD, test_loss_vec_SGD, 'b',
             label='{}{}{}'.format('SGD test(lr=', learningrate_SGD, ')'),
             linewidth=1.2)

ax6.legend(loc='upper right', prop={'size': 6})

#lambda-plot:
if GN_allowed == True:
    ax7.plot(epoch_vec_GN, lam_vec_GN, 'r',
             label='Lambda value '+ '{}{}{}'.format('(lambda update value: ', lam_up,')'),
             linewidth=0.8)

ax7.set_xlabel('Epochs')
ax7.set_ylabel('Lambda')
ax7.set_title('Lambda updates Hessian-Free')
ax7.set_ylim(-0.05, 1.5)
ax7.set_xlim(-0.05, epochs_GN)
ax7.legend(loc='upper right', prop={'size': 6})


if plotting == True:
    plt.tight_layout()
    plt.show()
else:
    print("no plots were generated ...")
