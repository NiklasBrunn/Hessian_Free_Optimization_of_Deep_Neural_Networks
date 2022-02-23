import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist
from train_steps import train_step_hessian_free, train_step_gradient_descent, model_loss


data_type = 'mnist'  # Data set. options: 'mnist' , 'sin', 'square'
model_architecture = [784, 800, 10]  # For 'sin', 'square': [1, 15, 15, 1]
train_size = 60000
test_size = 10000
train_time = 200  # Desired train time in seconds for each method

GGN = True  # Training with the Hessian Free method. (False: No training.)
preconditioning = True  # True: PCG-Method, False: Vanilla CG-Method
min_CG_steps = 3
eps = 0.0005  # accuracy
r = 1.05  # Large r can lead to NaN errors. We dont know why.
batch_size_GGN = 1000  # Batch_size

SGD = True  # Training with the SGD. (False: No training.)
eta = 0.1  # Learning rate
batch_size_SGD = 100  # Batch size


tf.random.set_seed(1234)

# For generating the data sets
def data_generator(train_size, test_size):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = tf.reshape(x_train[:train_size], [train_size, 784])/255
        x_test = tf.reshape(x_test[:test_size], [test_size, 784])/255
        y_train = tf.one_hot(y_train[:train_size], depth=10)
        y_test = tf.one_hot(y_test[:test_size], depth=10)
    elif data_type == 'sin':
        x_train = 2 * tf.random.normal([train_size, 1])
        x_test = 2 * tf.random.normal([test_size, 1])
        y_train = np.sin(x_train) + 0.1 * tf.random.normal([train_size, 1])
        y_test = np.sin(x_test)
    elif data_type == 'square':
        x_train = 2 * tf.random.normal([train_size, 1])
        x_test = 2 * tf.random.normal([test_size, 1])
        y_train = x_train ** 2 + 0.1 * tf.random.normal([train_size, 1])
        y_test = x_test ** 2
    return x_train, y_train, x_test, y_test


# Defining the data sets
x_train, y_train, x_test, y_test = data_generator(train_size, test_size)

# Defining the SGD and GNN dense models according to the given model_architecture
models = []
for i in range(GGN+SGD):
    input_layer = tf.keras.Input(shape=model_architecture[0])
    for i, num in enumerate(model_architecture[1:]):
        if i == 0:
            x = tf.keras.layers.Dense(num, activation='relu')(input_layer)
        elif i == np.shape(model_architecture[1:])[0]-1:
            output_layer = tf.keras.layers.Dense(num)(x)
        else:
            x = tf.keras.layers.Dense(num, activation='relu')(x)

    models.append(tf.keras.Model(input_layer, output_layer))

# Metrics for evaluation during training
def metric(model, data, label):
    if data_type == 'mnist':
        error = tf.reduce_sum(tf.where(tf.argmax(label, axis=1) -
                                       tf.argmax(tf.nn.softmax(model(data)), axis=1) != 0, 1, 0))
    else:
        error = model_loss(label, model(data))
    return np.array(error)

# Shuffeling the dataset and computing loss on train and test dataset at the begin of every epoch
def shuffle_and_evaluate(opt, x_train, y_train):
    perm = np.random.permutation(train_size)
    x_train = np.take(x_train, perm, axis=0)
    y_train = np.take(y_train, perm, axis=0)

    if opt == 'SGD':
        th = time_history_SGD
        model = model_SGD
    else:
        th = time_history_GGN
        model = model_GGN

    error_test = metric(model, x_test,  y_test)
    if data_type == 'mnist':
        error_train = metric(model, x_train, y_train)
        print('Epoch {}, Time {:.2f}s. Loss on test data: {:.5f}. Accuracy: Test {:.5f} ({}), Train {:.5f} ({})'.format(epoch, np.cumsum(
            th)[-1], np.array(model_loss(y_test, model(x_test))), 1-error_test/test_size, error_test, 1-error_train/train_size, error_train))
    else:
        print('Epoch {}, Time {:.2f}s. Loss on test data: {:.5f}'.format(
            epoch, np.cumsum(th)[-1], error_test))
    return x_train, y_train, error_test


# Training with the Hessian Free method.
if GGN == True:
    model_GGN = models[0]
    model_GGN.compile(loss=model_loss)
    model_GGN.summary()

    num_updates_GGN = int(train_size / batch_size_GGN)
    error_history_GGN = [metric(model_GGN, x_test,  y_test)]
    time_history_GGN = [0.]
    lam = 1.
    v = [tf.zeros(tf.shape(t)) for t in model_GGN.trainable_variables]
    lam_history = [lam]
    epoch = 1

    print('Training with GGN')
    while np.cumsum(time_history_GGN)[-1] < train_time:
        x_train, y_train, error_test = shuffle_and_evaluate('GGN', x_train, y_train)
        for i in range(num_updates_GGN):
            start = i * batch_size_GGN
            end = start + batch_size_GGN

            toc = time.time()
            lam, v = train_step_hessian_free(
                model_GGN, x_train[start: end], y_train[start: end], tf.constant(lam), v, preconditioning, min_CG_steps, eps, tf.constant(r))

            time_history_GGN = np.append(time_history_GGN, time.time() - toc)
            lam_history = np.append(lam_history, lam)

            if i == 0:
                error_new = error_test

            elif i > 0 and i % min(max(int(epoch/3), 1), max(int(num_updates_GGN/6), 1)) == 0:
                error_new = metric(model_GGN, x_test, y_test)

            error_history_GGN = np.append(error_history_GGN, error_new)
        epoch += 1

# Training with the SGD.
if SGD == True:
    model_SGD = models[-1]
    model_SGD.compile(loss=model_loss)
    model_SGD.summary()

    num_updates_SGD = int(train_size / batch_size_SGD)
    error_history_SGD = [metric(model_SGD, x_test,  y_test)]
    time_history_SGD = [0.]
    epoch = 1
    print('Training with SGD')

    while np.cumsum(time_history_SGD)[-1] < train_time:
        x_train, y_train, error_test = shuffle_and_evaluate('SGD', x_train, y_train)
        for i in range(num_updates_SGD):
            start = i * batch_size_SGD
            end = start + batch_size_SGD

            toc = time.time()
            train_step_gradient_descent(model_SGD, x_train[start: end], y_train[start: end], eta)

            time_history_SGD = np.append(time_history_SGD, time.time() - toc)

            if i == 0:
                error_new = error_test

            elif i > 0 and i % min(epoch, max(int(num_updates_GGN/4), 1)) == 0:
                error_new = metric(model_SGD, x_test, y_test)

            error_history_SGD = np.append(error_history_SGD, error_new)
        epoch += 1


# Plotting the results
if GGN == True:
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6*1.5, 4*1.5))
    ax1.plot(np.cumsum(time_history_GGN), error_history_GGN, c='navy',
             label='H({}), Batch size: {}'.format(min_CG_steps, batch_size_GGN), lw=1.5, alpha=0.9)
    if data_type == 'mnist':
        ax1.set_ylim(min(145, np.min(error_history_GGN)-5), 1500)

if GGN == False:
    f, ax1 = plt.subplots(1, 1, figsize=(6*1.5, 3*1.5))
    if data_type == 'mnist':
        ax1.set_ylim(145, 1500)


if SGD == True:
    ax1.plot(np.cumsum(time_history_SGD), error_history_SGD, c='orange',
             label='SGD({}), Batch size: {}'.format(eta, batch_size_SGD), lw=1.5, alpha=0.9)

ax1.legend(loc='upper right', fancybox=False, edgecolor='black', framealpha=1)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([0.1, 1, 10, 100, 1000, 3000])
ax1.set_xticklabels(['0.1', '1', '10', '100', '1000', '3000'])

if data_type == 'mnist':
    ax1.axhline(160, c='black', label='Benchmark', lw=2, ls='--')
    ax1.set_yticks([160, 200, 500, 1000])
    ax1.set_yticklabels(['160', '200', '500', '1000'])
    ax1.set_xlim(1, train_time)
else:
    ax1.set_yticks([0.001, 0.001, 0.01, 0.1])
    ax1.set_yticklabels(['0.0001', '0.001', '0.01', '0.1'])
    if SGD == True and GGN == True:
        ax1.set_ylim(min(np.min(error_history_SGD)*0.9, np.min(error_history_GGN)*0.9), 0.2)
    elif SGD == True and GGN == False:
        ax1.set_ylim(np.min(error_history_SGD)*0.9, 0.2)
    elif SGD == False and GGN == True:
        ax1.set_ylim(np.min(error_history_GGN)*0.9, 0.2)
    ax1.set_xlim(0.5, train_time)

ax1.set_ylabel('Total Error on Test Dataset')
ax1.set_xlabel('Train Time in seconds')
ax1.grid()

if GGN == True:
    ax2.plot(np.cumsum(time_history_GGN), lam_history, c='red', lw=1.5, alpha=0.9)
    ax2.set_xlim(1, train_time)
    ax2.set_ylabel('$\lambda$')
    ax2.set_xlabel('Train Time in seconds')
    ax2.grid()
plt.tight_layout()
plt.show()

f, ax = plt.subplots(1, 1, figsize=(6*1.5, 3*1.5))
a = np.linspace(-6, 6, 250)
x1 = model_SGD.predict(a)
x2 = model_GGN.predict(a)
ax.scatter(x_train, y_train, c='green', s=0.08)
ax.plot(a, x1, c='red', lw=1.5)
ax.plot(a, x2, c='blue', lw=1.5)
ax.grid()
plt.show()
