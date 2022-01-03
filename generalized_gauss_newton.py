import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data_size = 2000
batch_size = 100
epochs = 15

model_neurons = [1, 25, 25, 1]
eta = 0.01


x_train = tf.random.normal([data_size, model_neurons[0]])
y_train = x_train ** 2 + 0.1 * tf.random.normal([data_size, model_neurons[0]])


def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


input_layer = tf.keras.Input(shape=(model_neurons[0],))
layer_1 = tf.keras.layers.Dense(model_neurons[1], activation='relu')(input_layer)
layer_2 = tf.keras.layers.Dense(model_neurons[2], activation='relu')(layer_1)
layer_3 = tf.keras.layers.Dense(model_neurons[3])(layer_2)

model = tf.keras.Model(input_layer, layer_3, name='Model')

model.compile(loss=model_loss)
model.summary()


def train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)
    jac_model = tape.jacobian(y_pred, theta)

    param_shape = [tf.shape(g).numpy() for g in grad_loss]
    n_params = [tf.reduce_prod(s).numpy() for s in param_shape]

    grad_vec = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad_loss)]
    jac_mat = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
               for i, h in enumerate(jac_model)]
    jac_mat_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac_mat]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_mat_T, jac_mat)]

    eye_eps = [tf.eye(G[i].shape[0]) * 1e-5 for i in range(len(G))]

    update = [tf.constant(eta * np.linalg.solve(G[i] + eye_eps[i], grad_vec[i]))
              for i in range(len(G))]

    update = [tf.reshape(u, s) for u, s in zip(update, param_shape)]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(data_size / batch_size)


for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))

    for i in range(num_updates):

        start = i * batch_size
        end = start + batch_size

        train_step(x_train[start:end], y_train[start:end])


f, ax = plt.subplots(1, 1, figsize=(6, 4))

a = np.linspace(-4, 4, 200)
x = model.predict(a)

ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

ax.plot(a, a**2, label='Ground Truth', c='green')
ax.plot(a, x, label='Prediction', c='blue')

ax.legend(loc='upper left')
plt.show()
