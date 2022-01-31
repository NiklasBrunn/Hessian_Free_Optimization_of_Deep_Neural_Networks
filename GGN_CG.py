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
epochs = 10
CG_steps = 10 # minimale Anzahl der Schritte in der CG-Methode.
model_neurons = [1, 30, 30, 1]
num_updates = int(train_size / batch_size)

outliers = False # wenn True, dann werden in den Daten für die y-Werte Outlieres generiert.
max_num_outliers = 100 # Maximale Anzahl der Outliers im generierten Datensatz.

SGD_allowed = True # wenn True, dann wird SGD-Update performt!
GN_allowed = True # wenn True, dann wird auch GN-Update nach dem SGD-Update performt!
gradient_cal = 'alternativ' # kann als 'standard' oder 'alternativ' gesetzt werden!
                          # wenn standard, dann wird der Gradient mit Rückwärts AD durch
                          # die ges. Objektfunktion berechnet, mit alternativ wird
                          # nur die Jacobi bezgl. der Netzwerkparameter berechnet und dann
                          # mit den Residuen multipliziert.
GN_cal = 'R_Op' # wenn True, dann wird die GN-Matrix vor dem CG-update komplett berechnet,
              # ansonsten werden im CG-update Matrix-Vektor Produkte berechnet,
              # ohne Verwendung der GN-Matrix!
              # wenn nicht False oder True dann wird mit dem R-Operator im CG-Verfahren
              # optimiert
plotting = True # wenn True, dann werden die generierten Plots angezeigt!

#########################################
#Data generation (optional mit Outliern):
#########################################

def toy_data_generator(size, noise, outliers, max_num_outliers):
    x = tf.random.normal([size, model_neurons[0]])

    if outliers == True:

        vec = np.zeros(size)
        outliers_index_vec = np.random.randint(0, size, max_num_outliers)

        for j in range(max_num_outliers):

            # y-Werte werden mit Normal-3-1-gezogenen Werten addiert
            #vec[outliers_index_vec[j]] = np.random.normal(3, 1, 1)[0]

            # y-Werte werden mit Normal-0-1-gezogenen Werten addiert
            #vec[outliers_index_vec[j]] = np.random.normal(0, 1, 1)[0]

            # y-Werte werden mit 6 addiert
            vec[outliers_index_vec[j]] =  6.0

        vec = tf.constant(vec, dtype=tf.float32, shape=[size, 1])
        y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]]) + vec

    else:

        y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])

    return x, y

tf.random.set_seed(Data_Seed) # Test und Trainingsdaten ziehen so verschidene x-Werte

x_train, y_train = toy_data_generator(train_size, 0.1, outliers, max_num_outliers)
x_test, y_test = toy_data_generator(test_size, 0, False, max_num_outliers)



##########################
#def. model loss and model:
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
#functions:
###########

#Funktion für Hesse-Vektor-Produkte ohne davor Gradienten berechnet zu haben:
def efficient_hessian_vec(v, x, y, theta, lam):
    #s = time.time() # Einfügen ermöglicht die Rechenzeit für die Funktion auszugeben
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            y_pred = model(x)
            loss = model_loss(y, y_pred)
        theta = model.trainable_variables
        grad = tape1.gradient(loss, theta)

        #grad = tape1.gradient(loss, theta[0])# neue gute Methode zum Vergleich
        #gradd = tape1.gradient(loss, theta[0])## langsame Berechnung mit der korrekten Hesse

        grad_1 = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)], axis=0))
        mat_vec = tf.math.multiply(grad_1, tf.stop_gradient(v))

        #mat_vec = tf.math.multiply(grad_1, tf.stop_gradient(v[0:n_params[0]]))# neue gute Methode zum Vergleich

    mat_vec_res = tape2.gradient(mat_vec, theta)

    #mat_vec_res = tape2.gradient(mat_vec, theta[0])# neue gute Methode zum Vergleich

    mat_vec_res = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(mat_vec_res)], axis=0))

    #hess = tape2.jacobian(gradd, theta[0])## langsame Berechnung mit der korrekten Hesse
    #hess = tf.reshape(hess, [n_params[0], n_params[0]])## langsame Berechnung mit der korrekten Hesse

    #hess_vec = tf.linalg.matvec(hess, v[0:n_params[0]])## langsame Berechnung mit der korrekten Hesse
    #
    #print(hess_vec)## langsame Berechnung mit der korrekten Hesse

    #print(mat_vec_res)# neue gute Methode zum Vergleich

    #print(hess_vec - mat_vec_res)

    #elapsed = time.time() - s # Einfügen ermöglicht die Rechenzeit für die Funktion auszugeben
    #print('estimated time for hessian-vector calculation is:', elapsed)
    return mat_vec_res + lam * v


def hessappr_vec(v, jac, lam):
    return tf.reduce_mean(tf.linalg.matvec(jac, tf.linalg.matvec(jac, v), transpose_a=True), axis=0) + lam * v

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
    return v_new/batch_size + lam * v

def preconditioned_cg_method_R_Op(x_batch, y_batch, v, b, min_steps, precision):
    r = b - fastmatvec(x_batch, y_batch, v, lam) # (A+lam*I) * x
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec(x_batch, y_batch, d, lam)
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



# Martens Werte: min_steps = 10, precision = 0.0005
def preconditioned_cg_method(A, x, b, min_steps, precision):
    r = b - hessappr_vec(x, A, lam) # (A+lam*I) * x
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = hessappr_vec(d, A, lam)
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
def preconditioned_cg_method_hess(v, b, min_steps, precision, xin, yin, theta):
    r = b - efficient_hessian_vec(v, xin, yin, theta, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = efficient_hessian_vec(d, xin, yin, theta, lam)
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



##############
#OPTIMIZATION:
##############

###NEW!! das ist der standard_GN_Trainingsstep, bei dem noch Hesse_Vektor_Produkte berechnet werden!
def train_step_Hesse_vec(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables

    grad_obj = tape.gradient(loss, theta)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method_hess(update_old, grad_obj, CG_steps, 0.0005, x, y, theta)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / (tf.tensordot(grad_obj, update, 1) + tf.tensordot(update, efficient_hessian_vec(update, x, y, theta, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update



# standard_GN_train_step:
def train_step_generalized_gauss_newton(x, y, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)
        #y_gather = [tf.gather(y_pred, i) for i in range(batch_size)]

    res = y_pred - y
    if model_neurons[0] == 1:
        res = tf.reshape(res, (batch_size, 1, 1))

    theta = model.trainable_variables

    #grad_net = [tf.concat([tf.reshape(h, [model_neurons[-1], n_params[i]]) for i, h in enumerate(tape.gradient(y_gather[j], theta))], axis=1) for j in range(batch_size)]
    #print(grad_net)
    #print('stop')

    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    if gradient_cal == 'standard':
        # Gradient berechnet durch Jacobi_Vector_Produkt:
        grad_obj = tf.squeeze(tf.reduce_mean(tf.matmul(jac, res, transpose_a=True), axis=0))

    elif gradient_cal == 'alternativ':
        # (optional) Gradient mit Tape berechnen (!! ist gleich schnell):
        grad_obj = tape.gradient(loss, theta)
        grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_obj)], axis=0))

    else:
        print('es wird kein Gradient berechnet, da <gradient_call> nicht richtig definiert ist!')


    if GN_cal == True:
        # (optional) mit vorher berechneter GN-Matrix (!!dauert signifikant länger oben):
        # GN-Matrix optional berechnet
        GN = tf.reduce_mean(tf.matmul(jac, jac, transpose_a = True), axis=0)
        update = preconditioned_cg_method_complete_GN(GN, update_old, grad_obj, CG_steps, 0.0005)

    elif GN_cal == False:
        update = preconditioned_cg_method(jac, update_old, grad_obj, CG_steps, 0.0005)
    else:
        update = preconditioned_cg_method_R_Op(x, y, update_old, grad_obj, CG_steps, 0.0005)



    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    #rho = impr / (tf.tensordot(grad_obj, update, 1) +
    #              tf.tensordot(update, hessappr_vec(update, jac, 0), 1))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec(x, y, update, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update


#standard_SGD_Training:
def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])



##########
#TRAINING:
##########

#Erstellen der Data-Plots:
f, ((ax0, ax1, ax3, ax5), (ax7, ax2, ax4, ax6)) = plt.subplots(2, 4, figsize=(18, 8))

a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
ax0.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)
ax0.plot(a, a**2, label='Ground Truth', c='green')


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
        print('estimated time for the update step:', elapsed, 'sec')
        if epoch == 0:
            time_vec_SGD[epoch] = elapsed
        else:
            time_vec_SGD[epoch] = time_vec_SGD[epoch - 1] + elapsed

    #print(time_vec_SGD)
    #elapsed = time.time() - t
    #print(elapsed)

    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='Prediction SGD', c='blue')


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
        print('estimated time for the update step:', elapsed, 'sec')
        if epoch == 0:
            time_vec_GN[epoch] = elapsed
        else:
            time_vec_GN[epoch] = time_vec_GN[epoch - 1] + elapsed

    #print(time_vec_GN)
    #elapsed = time.time() - t
    #print(elapsed)

    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='Prediction GN', c='orange')

ax0.set_ylim(-0.6, 10)
ax0.set_xlim(-np.sqrt(10), np.sqrt(10))
ax0.set_title('Data and Predictions')
ax0.legend(loc='upper right')


#######
#PLOTS:
#######

####Train_loss_epochs_plot:
ax1.plot(epoch_vec_SGD, train_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train-Loss')
ax1.set_title('Train-Loss per Epochs:')
ax1.set_ylim(-0.005, 0.1)
ax1.set_xlim(0.5, epochs)

if GN_allowed == True:
    ax1.plot(epoch_vec_GN, train_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax1.legend(loc='upper right')

####Test_loss_epochs_plot:
ax2.plot(epoch_vec_SGD, test_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Test-Loss')
ax2.set_title('Test-Loss per Epochs:')
ax2.set_ylim(-0.005, 0.1)
ax2.set_xlim(0.5, epochs)

if GN_allowed == True:
    ax2.plot(epoch_vec_GN, test_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax2.legend(loc='upper right')

####Train_loss_time_plot:
ax3.plot(time_vec_SGD, train_loss_vec_SGD, 'r--',label='SGD', linewidth=1.2)
ax3.set_xlabel('Time (in seconds)')
ax3.set_ylabel('Train-Loss')
ax3.set_title('Train-Loss per Time:')

if GN_allowed == True:
    ax3.plot(time_vec_GN, train_loss_vec_GN, 'b--', label='GN', linewidth=1.2)

ax3.legend(loc='upper right')

####Test_loss_time_plot:
ax4.plot(time_vec_SGD, test_loss_vec_SGD, 'ro', label='SGD', linewidth=1.2)
ax4.set_xlabel('Time (in seconds)')
ax4.set_ylabel('Test-Loss')
ax4.set_title('Test-Loss per Time:')

if GN_allowed == True:
    ax4.plot(time_vec_GN, test_loss_vec_GN, 'bo', label='GN', linewidth=1.2)

ax4.legend(loc='upper right')

####Loss per Epochs GN:
ax5.plot(epoch_vec_GN, train_loss_vec_GN, 'r', label='GN_Train', linewidth=1.2)
ax5.set_xlabel('Epochs')
ax5.set_ylabel('Loss')
ax5.set_title('Loss per Epochs (GN)')

if GN_allowed == True:
    ax5.plot(epoch_vec_GN, test_loss_vec_GN, 'b', label='GN_Test', linewidth=1.2)

ax5.legend(loc='upper right')

####Loss per Epochs SGD:
ax6.plot(epoch_vec_GN, train_loss_vec_SGD, 'r', label='SGD_Train', linewidth=1.2)
ax6.set_xlabel('Epochs')
ax6.set_ylabel('Loss')
ax6.set_title('Loss per Epochs (SGD)')

if GN_allowed == True:
    ax6.plot(epoch_vec_GN, test_loss_vec_SGD, 'b', label='SGD_Test', linewidth=1.2)

ax6.legend(loc='upper right')

#placeholder:
ax7.plot([0], [0], 'r', linewidth=0.1)
ax7.set_title('empty plot')


if plotting == True:
    plt.tight_layout()
    plt.show()
else:
    print("no plots were generated ...")


#################
#TO-DOs und Link:
#################

'''
https://sudonull.com/post/61595-Hessian-Free-optimization-with-TensorFlow

1) Plots sollen gemittelte Werte mit Auswertungen von ca. 5 verschidenen Random-
   Seeds zeigen
2) CasADi Auswertung als Vergleich für die schriftliche Ausarbeitung   
3) R_OP nochmal wegen den Batches überprüfen ob das so stimmt!!!
[4) ADAM optimizer gegentesten (konvergiert das schneller?)]
'''
