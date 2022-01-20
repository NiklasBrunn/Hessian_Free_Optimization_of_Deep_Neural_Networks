import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)



################
#Hyperparameter:
################

Data_Seed = 1
Model_Seed = 1
train_size = 1500
test_size = 500
batch_size = 100
epochs = 5
CG_steps = 10
#outliers = True
#max_num_outliers = 100
model_neurons = [1, 30, 30, 1]
GN_allowed = True # wenn TRUE, dann wird auch GN-Update nach dem SGD-Update performt!
SGD_allowed = True # wenn TRUE, dann wird auch SGD-Update nach dem SGD-Update performt!
plotting = False # wenn True, dann werden die generierten Plots angezeigt!

#################
#Data generation:
#################

def toy_data_generator(size, noise, outliers, max_num_outliers):
    x = tf.random.normal([size, model_neurons[0]])
    y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    #if outliers == True:
    #    outliers_index_vec = np.zeros([size, model_neurons[0]])
    #    print(outliers_index_vec)
    #    vec = np.random.randint(0, size, max_num_outliers)
    #    for j in range(vec):
    #        print(outliers_index_vec[j][1])
            #outliers_index_vec[j] = [vec[j]]
        #print(outliers_index_vec)
        #for i in range(max_num_outliers):
        #    print(y[outliers_index_vec[i]]) #+= tf.random.normal([1], 0, 1)
        #    print(tf.random.normal([1], 0, 1))
    #print(tf.shape(y))
    #print(y)
    return x, y

tf.random.set_seed(Data_Seed)

x_train, y_train = toy_data_generator(train_size, 0.1, outliers, max_num_outliers)
x_test, y_test = toy_data_generator(test_size, 0, outliers, max_num_outliers)



##########################
#Def model loss and model:
##########################

def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


tf.random.set_seed(Model_Seed)

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
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1



###########
#Functions:
###########

def fastmatvec(v, jac, lam):
    return tf.reduce_mean(tf.linalg.matvec(jac, tf.linalg.matvec(jac, v), transpose_a=True), axis=0) + lam * v

# Martens Werte: min_steps = 10, precision = 0.0005
def cg_method(jac, x, b, min_steps, precision):
    r = b - fastmatvec(x, jac, lam)
    d = r
    i, k = 0, min_steps
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
    r = b - fastmatvec(x, A, lam) # (A+lam*I) * x
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
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

# Martens Werte: min_steps = 10, precision = 0.0005
def preconditioned_cg_method_complete_GN(GN, x, b, min_steps, precision):
    r = b - (tf.linalg.matvec(GN, x) + lam * x)
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = tf.linalg.matvec(GN, d) + lam * d
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



############
#OPTIMIZERS:
############

def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    res = y_pred - y
    if model_neurons[0] == 1:
        res = tf.reshape(res, (batch_size, 1, 1))

    theta = model.trainable_variables
    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    # Gradient berechnet durch Jacobi_Vector_Produkt:
    grad_obj = tf.squeeze(tf.reduce_mean(tf.matmul(jac, res, transpose_a=True), axis=0))

    # (optional) Gradient mit Tape berechnen (!! ist gleich schnell):
    #grad_obj = tape.gradient(loss, theta)
    #grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))


    update = preconditioned_cg_method(jac, update_old, grad_obj, CG_steps, 0.0005)

    # (optional) mit vorher berechneter GN-Matrix (!!dauert signifikant länger oben):
    # GN-Matrix optional berechnet
    #GN = tf.reduce_mean(tf.matmul(jac, jac, transpose_a = True), axis=0)
    #update = preconditioned_cg_method_complete_GN(GN, update_old, grad_obj, 10, 0.0005)


    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec(update, jac, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update


def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(train_size / batch_size)



##########
#TRAINING:
##########

#SGD-TRAINING:
#t = time.time()
test_loss_vec_SGD = np.zeros(epochs)
train_loss_vec_SGD = np.zeros(epochs)
epoch_vec_SGD = [i for i in range(epochs)]
time_vec_SGD = np.zeros(epochs)

if SGD_allowed == True:
    for epoch in range(epochs):
        train_loss = model_loss(y_train, model.predict(x_train))
        print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))), epochs, train_loss))
        test_loss = model_loss(y_test, model.predict(x_test))
        print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))), epochs, test_loss))

        test_loss_vec_SGD[epoch] = test_loss
        train_loss_vec_SGD[epoch] = train_loss

        t = time.time()
        for i in range(num_updates):
            #test_loss = model_loss(y_test, model.predict(x_test))
            #print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
            #                                                           1).zfill(len(str(epochs))), epochs, test_loss))
            start = i * batch_size
            end = start + batch_size

            train_step_gradient_descent(x_train[start: end], y_train[start: end], 0.3)
        elapsed = time.time() - t
        print('time for the update step:', elapsed)
        if epoch == 0:
            time_vec_SGD[epoch] = elapsed
        else:
            time_vec_SGD[epoch] = time_vec_SGD[epoch - 1] + elapsed

    #print(time_vec_SGD)
    #elapsed = time.time() - t
    #print(elapsed)

    #Approximated_function_plot:
    f, ax4 = plt.subplots(1, 1, figsize=(6, 4))

    a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
    x = model.predict(a)

    ax4.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

    ax4.plot(a, a**2, label='Ground Truth', c='green')
    ax4.plot(a, x, label='Prediction', c='blue')

    ax4.set_ylim(-0.6, 10)
    ax4.set_xlim(-np.sqrt(10), np.sqrt(10))
    ax4.set_title('Prediction SGD_Model')

    ax4.legend(loc='upper right')


#GN-TRAINING:

# Modellparameter müssen nochmal gelost werden (sind die selben wie oben),
# damit die Methoden vergleichbar sind!
tf.random.set_seed(Model_Seed)

input_layer = tf.keras.Input(shape=(model_neurons[0],))
layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
layer_2 = tf.keras.layers.Dense(model_neurons[2], activation='relu')(layer_1)
layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)

model = tf.keras.Model(input_layer, layer_3, name='Model')

model.compile(loss=model_loss)
model.summary()

#t = time.time()
test_loss_vec_GN = np.zeros(epochs)
train_loss_vec_GN = np.zeros(epochs)
epoch_vec_GN = [i for i in range(epochs)]
time_vec_GN = np.zeros(epochs)

if GN_allowed == True:
    for epoch in range(epochs):
        train_loss = model_loss(y_train, model.predict(x_train))
        print('Epoch {}/{}. Loss on train data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))), epochs, train_loss))
        test_loss = model_loss(y_test, model.predict(x_test))
        print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                                   1).zfill(len(str(epochs))), epochs, test_loss))

        test_loss_vec_GN[epoch] = test_loss
        train_loss_vec_GN[epoch] = train_loss

        t = time.time()
        for i in range(num_updates):
            #test_loss = model_loss(y_test, model.predict(x_test))
            #print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
            #                                                           1).zfill(len(str(epochs))), epochs, test_loss))
            start = i * batch_size
            end = start + batch_size

            lam, update_old = train_step_generalized_gauss_newton(
                x_train[start: end], y_train[start: end], lam, update_old)
        elapsed = time.time() - t
        print('time for the update step:', elapsed)
        if epoch == 0:
            time_vec_GN[epoch] = elapsed
        else:
            time_vec_GN[epoch] = time_vec_GN[epoch - 1] + elapsed

    #Approximated_function_plot:
    f, ax5 = plt.subplots(1, 1, figsize=(6, 4))

    a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
    x = model.predict(a)

    ax5.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

    ax5.plot(a, a**2, label='Ground Truth', c='green')
    ax5.plot(a, x, label='Prediction', c='blue')

    ax5.set_ylim(-0.6, 10)
    ax5.set_xlim(-np.sqrt(10), np.sqrt(10))
    ax5.set_title('Prediction GN_Model')

    ax5.legend(loc='upper right')

    #print(time_vec_GN)
    #elapsed = time.time() - t
    #print(elapsed)



#######
#PLOTS:
#######

####Train_loss_epochs_plot:
h, ax0 = plt.subplots(1, 1, figsize=(6, 4))

ax0.plot(epoch_vec_SGD, train_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax0.set_xlabel('Epochs')
ax0.set_ylabel('Train-Loss')
ax0.set_title('Train-Loss per Epochs:')

if GN_allowed == True:
    ax0.plot(epoch_vec_GN, train_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax0.legend(loc='upper right')

####Test_loss_epochs_plot:
g, ax1 = plt.subplots(1, 1, figsize=(6, 4))

ax1.plot(epoch_vec_SGD, test_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Test-Loss')
ax1.set_title('Test-Loss per Epochs:')

if GN_allowed == True:
    ax1.plot(epoch_vec_GN, test_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax1.legend(loc='upper right')

####Train_loss_time_plot:
g1, ax2 = plt.subplots(1, 1, figsize=(6, 4))

ax2.plot(time_vec_SGD, train_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax2.set_xlabel('Time (in seconds)')
ax2.set_ylabel('Train-Loss')
ax2.set_title('Train-Loss per Time:')

if GN_allowed == True:
    ax2.plot(time_vec_GN, train_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax2.legend(loc='upper right')

####Test_loss_time_plot:
g2, ax3 = plt.subplots(1, 1, figsize=(6, 4))

ax3.plot(time_vec_SGD, test_loss_vec_SGD, 'ro', label='SGD', linewidth=1.2)
ax3.set_xlabel('Time (in seconds)')
ax3.set_ylabel('Test-Loss')
ax3.set_title('Test-Loss per Time:')

if GN_allowed == True:
    ax3.plot(time_vec_GN, test_loss_vec_GN, 'bo', label='GN', linewidth=1.2)

ax3.legend(loc='upper right')

if plotting == True:
    plt.show()
else:
    print("no plots were generated ...")


#################
#TO-DOs und Link:
#################

'''
https://sudonull.com/post/61595-Hessian-Free-optimization-with-TensorFlow

1) Epochen (x-Achse) / Time (y-Achse) - Plot für die Modelle hinzufügen
2) Plots sollen gemittelte Werte mit Auswertungen von ca. 5 verschidenen Random-
   Seeds zeigen
3) CasADi
4) R-Methode bzw. fastmatvec-Methode implementieren
5) [optional] optimale Hyperparameter für SGD herausfinden und gegentesten mit
              optimalen Hyperparametern für GN (z.B. Anzahl Schritte in CG ...)
'''
