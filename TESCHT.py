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
Model_Seed = 2
train_size = 1500
test_size = 500
batch_size = 1500
epochs = 5
CG_steps = 10 # minimale Anzahl der Schritte in der CG-Methode.
model_neurons = [1, 3, 3, 1]
num_updates = int(train_size / batch_size)

#########################################
#Data generation (optional mit Outliern):
#########################################

def toy_data_generator(size, noise):
    x = tf.random.normal([size, model_neurons[0]])
    y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    return x, y

tf.random.set_seed(Data_Seed) # Test und Trainingsdaten ziehen so verschidene x-Werte

x_train, y_train = toy_data_generator(train_size, 0.1)
x_test, y_test = toy_data_generator(test_size, 0)



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


def model_loss(y_pred, y_true):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


theta = model.trainable_variables
#theta = tf.Variable(tf.squeeze(tf.concat([tf.reshape(t, [n_params[i], 1]) for i, t in enumerate(theta)], axis=0)))
#print(theta[0])
#with tf.autodiff.ForwardAccumulator(
#   primals=theta,
#   tangents=tf.ones(22)) as acc:
#  y_pred = model(x_train[0:5, :])
#acc.jvp(y_pred)



#theta = tf.squeeze(tf.concat([tf.reshape(param, [n_params[i], 1]) for i, param in enumerate(theta)], axis=0))
#print(theta)

#y_pred = model(x_train[0:5, :])
#print(y_pred)
#print(tf.gather(y_pred, 2))

#with tf.GradientTape(persistent=True) as tape:
#    y_pred = model(x_train[0:5, :])
#    y_gather = [tf.gather(y_pred, i) for i in range(batch_size)]
#    loss = model_loss(y_pred, y_train[0:5, :])
#grad = tape.gradient(y_gather[3], theta)
#print(grad)

def G_v_Rop(vec, x, theta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
#        print(y_pred)
        y_gather = [tf.gather(y_pred, i) for i in range(batch_size)]
#    print(y_gather)
    #grads = [tape.gradient(y_gather[i], theta) for i in range(batch_size)]
    #grads = [tf.squeeze(tf.stack(tf.concat([tf.reshape(g, [n_params[j], 1]) for j, g in enumerate(tape.gradient(y_gather[i], theta))], axis=0))) for i in range(batch_size)]
    #print(grads)
    #grads = [tf.math.multiply(vec, tf.squeeze(tf.stack(tf.concat([tf.reshape(g, [n_params[j], 1]) for j, g in enumerate(tape.gradient(y_gather[i], theta))], axis=0)))) for i in range(batch_size)]
    #print(grads)
    J_v = [tf.reduce_sum(tf.math.multiply(vec, tf.squeeze(tf.stack(tf.concat([tf.reshape(g, [n_params[j], 1]) for j, g in enumerate(tape.gradient(y_gather[i], theta))], axis=0))))) for i in range(batch_size)]
#    grads = [tf.reduce_sum(tf.math.multiply(vec, tf.squeeze(tf.stack(tf.concat([tf.reshape(g, [n_params[j], 1]) for j, g in enumerate(tape.gradient(y_gather[i], theta))], axis=0))))) for i in range(batch_size)]
    #v_grads = tf.math.multiply(vec, tf.reduce_sum(grads, axis=0))
    #J_v = tf.reduce_sum(v_grads)
    #J_v = [tf.reduce_sum([tf.reduce_sum(v * tape.gradient(y_gather[i], theta)[j]) for j, v in enumerate(vec)]) for i in range(batch_size)]
#    [tf.reduce_sum(v * tape.gradient(y_gather[i], theta)[j]) for j, v in enumerate(vec)]
    return J_v

#print(G_v_Rop(tf.Variable(tf.ones(22)), x_train[0:5, :], theta))
s = time.time()
G_v_Rop(tf.Variable(tf.ones(22)), x_train[0:1500, :], theta)
selapsed = time.time() - s
print('estimated time for the jacobian_vector product (efficient):', selapsed)

#print(G_v_Rop(tf.Variable(tf.ones(22)), x_train[0:3, :], theta))

t = time.time()
with tf.GradientTape(persistent=True) as tape1:
    y_pred = model(x_train[0:1500, :])

theta = model.trainable_variables

jac = tape1.jacobian(y_pred, theta)
jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                 for i, h in enumerate(jac)], axis=2)
jac_v = tf.linalg.matvec(jac, tf.ones(22))

#jac_v = [tf.squeeze(tf.linalg.matvec(j, tf.ones(22))) for (i, j) in enumerate(jac)]
elapsed = time.time() - t
print('estimated time for the jacobian and multiplication with v:', elapsed)

print(jac_v)
