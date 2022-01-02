import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt


data_size = 10000
batch_size = 10
epochs = 20
eta = 0.005
model_neurons = [1, 100, 100, 1]


def data_generator(data_size):
    x_value = np.random.normal(loc=0, scale=1, size=(data_size, 1))
    y_value = x_value ** 2 + np.random.normal(loc=0, scale=0.1, size=(data_size, 1))
    return x_value, y_value


x_train, y_train = data_generator(data_size)


def model_loss(y, x_predicted):
    return 0.5 * (y - x_predicted) ** 2


input_layer = tf.keras.Input(shape=(model_neurons[0],))
dense_layer = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
dense_layer = tf.keras.layers.Dense(model_neurons[2], activation='relu')(dense_layer)
dense_layer = tf.keras.layers.Dense(model_neurons[3])(dense_layer)

model = tf.keras.Model(input_layer, dense_layer, name='Model')

model.compile(loss=model_loss)
model.summary()


def step(x, y):
    with tf.GradientTape(persistent=True) as tape:

        pred = model(x)
        loss = model_loss(y, pred)

    parameters = model.trainable_variables
    gradients_loss = tape.gradient(loss, parameters)
#    gradients_model = tape.gradient(pred, parameters)

#    print(gradients_model)

    eta_g = [g * eta for g in gradients_loss]

    model.set_weights([p - e_g for (p, e_g) in zip(parameters, eta_g)])


num_updates = int(data_size / batch_size)

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))

    for i in range(num_updates):

        start = i * batch_size
        end = start + batch_size

        step(x_train[start:end], y_train[start:end])


f, ax = plt.subplots(1, 1, figsize=(6, 4))

a = np.linspace(-5, 5, 200)
x = model.predict(a)

ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.2)
ax.plot(a, x, label='Prediction')
ax.plot(a, a**2, label='Ground Truth')


ax.legend(loc='upper left')
plt.show()
