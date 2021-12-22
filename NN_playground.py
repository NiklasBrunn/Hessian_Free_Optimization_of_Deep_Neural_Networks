# strg + shift + b (Atom run script)
import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
import matplotlib as mpl
from matplotlib import pyplot as plt

##################
#  hyperparameter:
##################
data_size = 10000
batch_size = 100
epochs = 10
act = 'relu'
layer_dims = [1, 100, 100, 1]

###################
#  generating data:
###################
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

x_train_mnist = x_train_mnist.reshape(60000, 784).astype("float32") / 255
x_test_mnist = x_test_mnist.reshape(10000, 784).astype("float32") / 255

y_train_mnist = y_train_mnist.astype("float32")
y_test_mnist = y_test_mnist.astype("float32")


def data_generator(data_size):
    x_value = np.random.normal(loc=0, scale=1, size=(data_size, 1))
    y_value = x_value ** 2 + np.random.normal(loc=0, scale=0.1, size=(data_size, 1))
    return x_value, y_value

x_train, y_train = data_generator(data_size)

######################
#  defining objective:
######################
def model_loss(x, x_predicted):
    return 0.5 * (x - x_predicted) ** 2

#################
#  generating NN:
#################
#input_layer = tf.keras.Input(shape=(layer_dims[0],))
#for j, i in enumerate(layer_dims[1:]):
#    if j == 0:
#        dense_layer = tf.keras.layers.Dense(
#            i, activation=act, name='layer_{}'.format(j+1))(input_layer)
#
#    elif j == len(layer_dims)-1:
#        dense_layer = tf.keras.layers.Dense(
#            i, name='layer_{}'.format(j+1))(dense_layer)
#
#    else:
#        dense_layer = tf.keras.layers.Dense(
#            i, activation=act, name='layer_{}'.format(j+1))(dense_layer)


model = tf.keras.Model(input_layer, dense_layer, name='Model')
model.summary()
model.compile(optimizer='adam', loss=model_loss)

###############
#  training NN:
###############
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

#####################
#  model predictions:
#####################
f, ax = plt.subplots(1, 1, figsize=(6, 4))

a = np.linspace(-5, 5, 200)

x = model.predict(a)

ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.2)
ax.plot(a, x, label='Prediction')
ax.plot(a, a**2, label='Ground Truth')


ax.legend(loc='upper left')
plt.show()
