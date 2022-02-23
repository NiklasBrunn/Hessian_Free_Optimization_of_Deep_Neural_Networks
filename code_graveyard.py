###########################################################################
#here we contain old or unused code that may be usefull or nice to have ...
###########################################################################


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

train_size = 1500
test_size = 500
batch_size = 100
epochs = 20
model_neurons = [1, 12, 12, 1]


def toy_data_generator(size, noise):
    x = tf.random.normal([size, model_neurons[0]])
    y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    return x, y


x_train, y_train = toy_data_generator(train_size, 0.1)
x_test, y_test = toy_data_generator(test_size, 0)


def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


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
lam = 1


def cg_method(A, x, b, kmax, eps):
    r = b - tf.linalg.matvec(A, x)
    d = r
    k = 0
    while tf.math.reduce_euclidean_norm(r) > eps and k < kmax:
        z = tf.linalg.matvec(A, d)
        alpha = tf.tensordot(r, r, 1) / tf.tensordot(d, z, 1)
        x = x + alpha * d
        r_new = r - alpha * z
        beta = tf.tensordot(r_new, r_new, 1) / tf.tensordot(r, r, 1)
        d = r_new + beta * d
        r = r_new
        k += 1
    return tf.reshape(x, [x.shape[0], 1])


def train_step_generalized_gauss_newton(x, y, lam):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)], axis=0)

    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    G = tf.reduce_mean(tf.matmul(jac, jac, transpose_a=True), axis=0)

#    update = tf.constant(np.linalg.solve(G + tf.eye(G.shape[0]) * lam, grad))
    update = cg_method(G + tf.eye(G.shape[0]) * lam,
                       tf.zeros(G.shape[0]), tf.squeeze(grad), 100, 1e-10)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / (tf.reduce_sum(grad * update) + tf.reduce_sum(update * tf.matmul(G, update)))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam


def train_step_gradient_descent(x, y, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad_loss = tape.gradient(loss, theta)

    update = [tf.constant(eta) * g for g in grad_loss]

    model.set_weights([p - u for (p, u) in zip(theta, update)])


num_updates = int(train_size / batch_size)


for epoch in range(epochs):
    test_loss = model_loss(y_test, model.predict(x_test))
    print('Epoch {}/{}. Loss on test data: {:.4f}.'.format(str(epoch +
                                                               1).zfill(len(str(epochs))), epochs, test_loss))

    for i in range(num_updates):

        start = i * batch_size
        end = start + batch_size

        lam = train_step_generalized_gauss_newton(x_train[start: end], y_train[start: end], lam)
    #    train_step_gradient_descent(x_train[start: end], y_train[start: end], 0.3)

f, ax = plt.subplots(1, 1, figsize=(6, 4))

a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
x = model.predict(a)

ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

ax.plot(a, a**2, label='Ground Truth', c='green')
ax.plot(a, x, label='Prediction', c='blue')

ax.set_ylim(-0.6, 10)
ax.set_xlim(-np.sqrt(10), np.sqrt(10))

ax.legend(loc='upper left')
plt.show()

# https://sudonull.com/post/61595-Hessian-Free-optimization-with-TensorFlow

'''
lam = np.ones(np.shape(n_params))

def train_step_generalized_gauss_newton(x, y, lam):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)]
    jac = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
           for i, h in enumerate(jac)]
    jac_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_T, jac)]

    eye_lam = [tf.eye(n_params[i]) * lam[i] for i in range(len(n_params))]

    update = [tf.constant(np.linalg.solve(G[i] + eye_lam[i], grad[i]))
              for i in range(len(G))]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, update, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))
    rho = np.array([impr/(tf.reduce_sum(g * u) + tf.reduce_sum(u * tf.linalg.matmul(G_mat, u)))
                    for (G_mat, g, u) in zip(G, grad, update)])

    lam = np.where(rho > 0.75, lam/1.5, lam)
    lam = np.where(rho < 0.25, lam*1.5, lam)

    return lam


def train_step_generalized_gauss_newton_cg(x, y, lam):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)]
    jac = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
           for i, h in enumerate(jac)]
    jac_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_T, jac)]

#    eye_lam = [tf.eye(n_params[i]) * lam[i] for i in range(len(n_params))]
    eye_lam = [tf.eye(n_params[i]) * 1e-5 for i in range(len(n_params))]

    update = []
    for i in range(len(G)):
        x = tf.zeros([n_params[i]])
        r = tf.squeeze(grad[i]) - tf.linalg.matvec(G[i] + eye_lam[i], x)
        d = r
        k = 0
        while tf.math.reduce_euclidean_norm(r) > 1e-10 and k < 100:
            z = tf.linalg.matvec(G[i] + eye_lam[i], d)
            alpha = tf.tensordot(r, r, 1) / tf.tensordot(d, z, 1)
            x = x + alpha * d
            r_new = r - alpha * z
            beta = tf.tensordot(r_new, r_new, 1) / tf.tensordot(r, r, 1)
            d = r_new + beta * d
            r = r_new
            k += 1

        update.append(tf.reshape(x*0.01, [n_params[i], 1]))
#        update.append(tf.reshape(x, [n_params[i], 1]))

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, update, param_shape)]

    model.set_weights(theta_new)

#    impr = loss - model_loss(y,  model(x))
#    rho = np.array([impr/(tf.reduce_sum(g * u) + tf.reduce_sum(u * tf.linalg.matmul(G_mat, u)))
#                    for (G_mat, g, u) in zip(G, grad, update)])

#    lam = np.where(rho > 0.75, lam/1.5, lam)
#    lam = np.where(rho < 0.25, lam*1.5, lam)

    return lam
'''














'''
lam = np.ones(np.shape(n_params))
def train_step_generalized_gauss_newton(x, y, lam):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)]
    jac = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
           for i, h in enumerate(jac)]
    jac_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_T, jac)]

    eye_lam = [tf.eye(n_params[i]) * lam[i] for i in range(len(n_params))]

    update = [tf.constant(np.linalg.solve(G[i] + eye_lam[i], grad[i]))
              for i in range(len(G))]

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, update, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))
    rho = np.array([impr/(tf.reduce_sum(g * u) + tf.reduce_sum(u * tf.linalg.matmul(G_mat, u)))
                    for (G_mat, g, u) in zip(G, grad, update)])

    lam = np.where(rho > 0.75, lam/1.5, lam)
    lam = np.where(rho < 0.25, lam*1.5, lam)

    return lam


def train_step_generalized_gauss_newton_cg(x, y, lam):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x)
        loss = model_loss(y, y_pred)

    theta = model.trainable_variables
    grad = tape.gradient(loss, theta)
    jac = tape.jacobian(y_pred, theta)

    grad = [tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)]
    jac = [tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
           for i, h in enumerate(jac)]
    jac_T = [tf.transpose(j, perm=[0, 2, 1]) for j in jac]

    G = [tf.constant(1/batch_size)*tf.reduce_sum(tf.constant(np.matmul(j_T, j)), axis=0)
         for (j_T, j) in zip(jac_T, jac)]

#    eye_lam = [tf.eye(n_params[i]) * lam[i] for i in range(len(n_params))]
    eye_lam = [tf.eye(n_params[i]) * 1e-5 for i in range(len(n_params))]

    update = []
    for i in range(len(G)):
        x = tf.zeros([n_params[i]])
        r = tf.squeeze(grad[i]) - tf.linalg.matvec(G[i] + eye_lam[i], x)
        d = r
        k = 0
        while tf.math.reduce_euclidean_norm(r) > 1e-10 and k < 100:
            z = tf.linalg.matvec(G[i] + eye_lam[i], d)
            alpha = tf.tensordot(r, r, 1) / tf.tensordot(d, z, 1)
            x = x + alpha * d
            r_new = r - alpha * z
            beta = tf.tensordot(r_new, r_new, 1) / tf.tensordot(r, r, 1)
            d = r_new + beta * d
            r = r_new
            k += 1

        update.append(tf.reshape(x*0.01, [n_params[i], 1]))
#        update.append(tf.reshape(x, [n_params[i], 1]))

    theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, update, param_shape)]

    model.set_weights(theta_new)

#    impr = loss - model_loss(y,  model(x))
#    rho = np.array([impr/(tf.reduce_sum(g * u) + tf.reduce_sum(u * tf.linalg.matmul(G_mat, u)))
#                    for (G_mat, g, u) in zip(G, grad, update)])

#    lam = np.where(rho > 0.75, lam/1.5, lam)
#    lam = np.where(rho < 0.25, lam*1.5, lam)

    return lam


    def train_step_generalized_gauss_newton_old(x, y, lam):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x)
            loss = model_loss(y, y_pred)

        theta = model.trainable_variables
        grad = tape.gradient(loss, theta)
        jac = tape.jacobian(y_pred, theta)

        grad = tf.concat([tf.reshape(g, [n_params[i], 1]) for i, g in enumerate(grad)], axis=0)

        jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                         for i, h in enumerate(jac)], axis=2)

        G = tf.reduce_mean(tf.matmul(jac, jac, transpose_a=True), axis=0)

    #    update = tf.constant(np.linalg.solve(G + tf.eye(G.shape[0]) * lam, grad))
        update = cg_method(G + tf.eye(G.shape[0]) * lam,
                           tf.zeros(G.shape[0]), tf.squeeze(grad), 100, 1e-10)

        theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

        theta_new = [p - tf.reshape(u, s) for (p, u, s) in zip(theta, theta_new, param_shape)]

        model.set_weights(theta_new)

        impr = loss - model_loss(y,  model(x))

        rho = impr / (tf.reduce_sum(grad * update) + tf.reduce_sum(update * tf.matmul(G, update)))

        if rho > 0.75:
            lam /= 1.5
        elif rho < 0.25:
            lam *= 1.5

        return lam

        def cg_method(A, x, b, kmax, eps):
            r = b - tf.linalg.matvec(A, x)
            d = r
            k = 0
            while tf.math.reduce_euclidean_norm(r) > eps and k < kmax:
                z = tf.linalg.matvec(A, d)
                alpha = tf.tensordot(r, r, 1) / tf.tensordot(d, z, 1)
                x = x + alpha * d
                r_new = r - alpha * z
                beta = tf.tensordot(r_new, r_new, 1) / tf.tensordot(r, r, 1)
                d = r_new + beta * d
                r = r_new
                k += 1
            return tf.reshape(x, [x.shape[0], 1])






            ####Approximated_function_plot:
            f, ax = plt.subplots(1, 1, figsize=(6, 4))

            a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
            x = model.predict(a)

            ax.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)

            ax.plot(a, a**2, label='Ground Truth', c='green')
            ax.plot(a, x, label='Prediction', c='blue')

            ax.set_ylim(-0.6, 10)
            ax.set_xlim(-np.sqrt(10), np.sqrt(10))

            ax.legend(loc='upper right')
            plt.show()


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
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time

'''
v = tf.Variable([1., 2.])
with tf.autodiff.ForwardAccumulator(v, tf.constant([1., 3.])) as acc:
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(v ** 3.)
        backward = tape.gradient(y, v)

print(backward)  # gradient from backprop
print(acc.jvp(backward))

'''
train_size = 1500
test_size = 500
batch_size = 100
epochs = 10
CG_steps = 10
model_neurons = [1, 10, 10, 1]
num_updates = int(train_size / batch_size)

tf.random.set_seed(1)


def toy_data_generator(size, noise):
    x = tf.random.normal([size, model_neurons[0]])
    y = x ** 2 + noise * tf.random.normal([size, model_neurons[0]])
    return x, y


x_train, y_train = toy_data_generator(train_size, 0.1)
x_test, y_test = toy_data_generator(test_size, 0)


def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


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


def fastmatvec(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]
    with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
        y_pred = model(x_batch)
    forward = acc.jvp(y_pred)
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
    backward = tape.gradient(y_pred, model.trainable_variables,
                             output_gradients=tf.stop_gradient(forward))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(backward)], axis=0))
#    print(v_new/batch_size + lam * v)
    return v_new/batch_size + lam * v


def preconditioned_cg_method(x_batch, y_batch, v, b, min_steps, precision, lam):
    r = b - fastmatvec(x_batch, y_batch, v, lam)
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


def train_step_generalized_gauss_newton(x_batch, y_batch, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x_batch)
        loss = model_loss(y_batch, y_pred)

    grad_obj = tape.gradient(loss, model.trainable_variables)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                     for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method(x_batch, y_batch, update_old, grad_obj, CG_steps, 0.0005, lam)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(model.trainable_variables, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y_batch,  model(x_batch))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec(x_batch, y_batch, update, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update


# for epoch in range(epochs):

    #    print(model_loss(y_test,  model(x_test)))
    #    print(model_loss(y_train,  model(x_train)))
for i in range(num_updates):
    start = i * batch_size
    end = start + batch_size
    lam, update_old = train_step_generalized_gauss_newton(
        x_train[start: end], y_train[start: end], lam, update_old)

'''
# Martens Werte: min_steps = 10, precision = 0.0005
def cg_method(jac, jac_softmax, x, b, min_steps, precision):
    r = b - fastmatvec_naiv(x, jac, jac_softmax, lam)
    d = r
    i, k = 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(x, b, 1) + tf.tensordot(x, r, 1)))
    while (i > k and phi_history[-1] < 0 and s < precision*k) == False:
        k = np.maximum(min_steps, int(i/min_steps))
        z = fastmatvec_naiv(d, jac, jac_softmax, lam)
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
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

train_size = 60000
test_size = 10000
batch_size = 250
epochs = 100
CG_steps = 10
num_updates = int(train_size / batch_size)

tf.random.set_seed(13)


def mnist_data_generator():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((len(x_train), 28, 28, 1)) / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1)) / 255.
    y_train = tf.one_hot(y_train[:train_size], depth=10)
    y_test = tf.one_hot(y_test[:test_size], depth=10)

    return x_train[:train_size], y_train, x_test[:test_size], y_test


x_train, y_train, x_test, y_test = mnist_data_generator()


def model_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))


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

param_shape = [tf.shape(t) for t in model.trainable_variables]
n_params = [np.prod(s) for s in param_shape]
ind = np.insert(np.cumsum(n_params), 0, 0)
update_old = tf.zeros(ind[-1])
lam = 1


def fastmatvec(x_batch, y_batch, v, lam):
    v_new = [v[i:j] for (i, j) in zip(ind[:-1], ind[1:])]
    v_new = [tf.Variable(tf.reshape(u, s)) for (u, s) in zip(v_new, param_shape)]
    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v_new) as acc:
            y_pred = model(x_batch)
            akt_out = tf.nn.softmax(y_pred)
        HJv = acc.jvp(akt_out)
    J_tHJv = tape.gradient(y_pred, model.trainable_variables,
                           output_gradients=tf.stop_gradient(HJv))

    v_new = tf.squeeze(tf.concat([tf.reshape(v, [n_params[i], 1])
                                  for i, v in enumerate(J_tHJv)], axis=0))
    return v_new / batch_size + lam * v


def preconditioned_cg_method(x_batch, y_batch, v, b, min_steps, precision, lam):
    r = b - fastmatvec(x_batch, y_batch, v, lam)
    y = r / (b ** 2 + lam)
    d = y
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1)))
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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


def train_step_generalized_gauss_newton(x_batch, y_batch, lam, update_old):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x_batch)
        loss = model_loss(y_batch, y_pred)

    grad_obj = tape.gradient(loss, model.trainable_variables)
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                     for i, g in enumerate(grad_obj)], axis=0))

    update = preconditioned_cg_method(x_batch, y_batch, update_old,
                                      grad_obj, CG_steps, 0.0005, lam)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(model.trainable_variables, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y_batch,  model(x_batch))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update, fastmatvec(x_batch, y_batch, update, 0), 1))

    if rho > 0.75:
        lam /= 1.5
    elif rho < 0.25:
        lam *= 1.5

    return lam, update


error_old = 10000
for epoch in range(epochs):

    error_test = np.sum(np.where(np.argmax(y_test, axis=1) -
                                 np.argmax(tf.nn.softmax(model.predict(x_test)), axis=1) != 0, 1, 0))
    error_train = np.sum(np.where(np.argmax(y_train, axis=1) -
                                  np.argmax(tf.nn.softmax(model.predict(x_train)), axis=1) != 0, 1, 0))
    print('Epoch {}/{}. Loss on test data: {:.5f}. Accuracy: Test {:.5f} ({}), Train {:.5f} ({})'.format(epoch +
                                                                                                         1, epochs, np.array(model_loss(y_test, model(x_test))),  1-error_test/test_size, error_test, 1-error_train/train_size, error_train))
    tic = time.time()
    for i in range(num_updates):
        #        error_new = np.sum(np.where(np.argmax(y_test, axis=1) -
        #                                    np.argmax(tf.nn.softmax(model.predict(x_test)), axis=1) != 0, 1, 0))
        #        if error_new < error_old:
        #            print('Loss on test data: {:.5f}. Wrong classified: {}'.format(
        #                np.array(model_loss(y_test, model(x_test))), error_new))
        #            error_old = error_new
        start = i * batch_size
        end = start + batch_size
        lam, update_old = train_step_generalized_gauss_newton(
            x_train[start: end], y_train[start: end], lam, update_old)
    elapsed = time.time() - tic

    print('Time elapsed: {:.5f}'.format(elapsed))

print(error_old)










#########################################################################
# Below we contain some early work from the Hessian_free_simple.py -file:
#########################################################################

# here we teste some runtime improvements where we calculated the matrix-
# vector products in different ways and used different versions of the
# CG method. Also we checked if we can get some improvements in runtime by
# calculating the gradient of the loss in different ways.
# We implemented, in contrast to our efficient method using fast-matrix-vector-
# products, an alternative way where we compute and store the damped GGN first
# and then calculate the matrix vector product (very slow ...)
# We also just for fun implemented a way to generate some outliers to the
# train data.

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

Data_Seed = 7
Model_Seed = 7
train_size = 1500
test_size = 500
batch_size = 100
epochs = 2
CG_steps = 3
model_neurons = [1, 15, 15, 1]
num_updates = int(train_size / batch_size)

outliers = False # if True, then we generate some outliers to the train data
#                  (for more details see below in the code).
max_num_outliers = 100 # max. number of generated outliers.

SGD_allowed = True # wenn True, dann wird SGD-Update performt!
GN_allowed = True # wenn True, dann wird auch GN-Update nach dem SGD-Update performt!
gradient_cal = 'standard' # can be set to 'standard' or anything else ...
#                           'standard': whole gradient of the loss is calculated
#                           else: we calculate just the Jac of the NN outputs
#                           and multiply it with the residuals to obtain the
#                           gradient of the loss.
GN_cal = False # if set to TRUE we calculate the whole damped GGN before the
#                CG-algorithm.
#                Else, we use fast-matrix-vector-products without storing and
#                computing the damped GGN at any time (more efficient).
plotting = True # if TRUE, then plots will be displayed at the end of the
#                 training.

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

x_train, y_train = toy_data_generator(train_size, 0.1,
                                      outliers, max_num_outliers)
x_test, y_test = toy_data_generator(test_size, 0, False, max_num_outliers)



##########################
#def. model loss and model:
##########################

def model_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)


tf.random.set_seed(Model_Seed)

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



###########
#functions:
###########

#Funktion für Hesse-Vektor-Produkte ohne davor Gradienten berechnet zu haben:
def efficient_hessian_vec(v, x, y, theta, lam):
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            y_pred = model(x)
            loss = model_loss(y, y_pred)
        theta = model.trainable_variables
        grad = tape1.gradient(loss, theta)

        grad_1 = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                       for i, g in enumerate(grad)],
                                      axis=0))
        mat_vec = tf.math.multiply(grad_1, tf.stop_gradient(v))

    mat_vec_res = tape2.gradient(mat_vec, theta)
    mat_vec_res = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                        for i, g in enumerate(mat_vec_res)],
                                       axis=0))
    return mat_vec_res + lam * v


def hessappr_vec(v, jac, lam):
    return tf.reduce_mean(tf.linalg.matvec(jac, tf.linalg.matvec(jac, v),
                                           transpose_a=True), axis=0) + lam * v

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
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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
    i, s, k = 0, 0, min_steps
    phi_history = np.array(- 0.5 * (tf.tensordot(v, b, 1) + tf.tensordot(v, r, 1))).reshape([1])
    while i <= k or s >= precision*k or phi_history[-1] >= 0:
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
    grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                     for i, g in enumerate(grad_obj)],
                                    axis=0))

    update = preconditioned_cg_method_hess(update_old, grad_obj,
                                           CG_steps, 0.0005, x, y, theta)

    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(theta, theta_new, param_shape)]

    model.set_weights(theta_new)

    impr = loss - model_loss(y,  model(x))

    rho = impr / (tf.tensordot(grad_obj, update, 1) +
                  tf.tensordot(update,
                               efficient_hessian_vec(update, x, y, theta, 0), 1))

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

    jac = tape.jacobian(y_pred, theta)
    jac = tf.concat([tf.reshape(h, [batch_size, model_neurons[-1], n_params[i]])
                     for i, h in enumerate(jac)], axis=2)

    if gradient_cal == 'standard':
        # (optional) Gradient mit Tape berechnen (!! ist gleich schnell):
        grad_obj = tape.gradient(loss, theta)
        grad_obj = tf.squeeze(tf.concat([tf.reshape(g, [n_params[i], 1])
                                         for i, g in enumerate(grad_obj)],
                                        axis=0))

    else:
        # Gradient berechnet durch Jacobi_Vector_Produkt:
        grad_obj = tf.squeeze(tf.reduce_mean(tf.matmul(jac, res, transpose_a=True),
                                             axis=0))


    if GN_cal == True:
        # (optional) mit vorher berechneter GN-Matrix (!!dauert signifikant länger oben):
        # GN-Matrix optional berechnet
        GN = tf.reduce_mean(tf.matmul(jac, jac, transpose_a = True), axis=0)
        update = preconditioned_cg_method_complete_GN(GN, update_old, grad_obj,
                                                      CG_steps, 0.0005)

    elif GN_cal == False:
        update = preconditioned_cg_method(jac, update_old, grad_obj,
                                          CG_steps, 0.0005)
    else:
        update = preconditioned_cg_method_R_Op(x, y, update_old, grad_obj,
                                               CG_steps, 0.0005)



    theta_new = [update[i:j] for (i, j) in zip(ind[:-1], ind[1:])]

    theta_new = [p - tf.reshape(u, s)
                 for (p, u, s) in zip(theta, theta_new, param_shape)]

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
f, ((ax0, ax1, ax3, ax5), (ax7, ax2, ax4, ax6)) = plt.subplots(2, 4,
                                                               figsize=(18, 8))

a = np.linspace(-np.sqrt(10), np.sqrt(10), 250)
ax0.scatter(x_train, y_train, label='Train Data', c='red', s=0.3)
ax0.plot(a, a**2, label='Ground Truth', c='green')


#SGD-TRAINING:
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

            start = i * batch_size
            end = start + batch_size

            s = time.time()
            train_step_gradient_descent(x_train[start: end], y_train[start: end], 0.3)
            elapseds = time.time() - s

            #train_time_SGD[epoch*num_updates+i] = elapseds
            #error_history_train_SGD[epoch*num_updates+i] = error_new_train_SGD
            #error_history_test_SGD[epoch*num_updates+i] = error_new_test_SGD
            #epochs_SGD[epoch*num_updates+i] = epoch

        elapsed = time.time() - t
        print('estimated time for the update step:', elapsed, 'sec')
        if epoch == 0:
            time_vec_SGD[epoch] = elapsed
        else:
            time_vec_SGD[epoch] = time_vec_SGD[epoch - 1] + elapsed


    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='Prediction SGD', c='blue')

#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//train_time_SGD.npy',
#          train_time_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//error_history_train_SGD.npy',
#          error_history_train_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//error_history_test_SGD.npy',
#           error_history_test_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//epochs_SGD.npy',
#           epochs_SGD)


#GN-TRAINING:

# Modellparameter müssen nochmal gelost werden (sind die selben wie oben),
# damit die Methoden vergleichbar sind!
tf.random.set_seed(Model_Seed)

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

#train_time_GN = np.zeros(epochs*num_updates)
#error_history_test_GN = np.zeros(epochs*num_updates)
#error_history_train_GN = np.zeros(epochs*num_updates)
#epochs_GN = np.zeros(epochs*num_updates)


if GN_allowed == True:
    for epoch in range(epochs):
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

        t = time.time()
        for i in range(num_updates):

            #error_new_train_GN = model_loss(y_train, model.predict(x_train))
            #error_new_test_GN = model_loss(y_test, model.predict(x_test))

            start = i * batch_size
            end = start + batch_size

            s = time.time()
            lam, update_old = train_step_generalized_gauss_newton(
                x_train[start: end], y_train[start: end], lam, update_old)
            elapseds = time.time() - s

            #train_time_GN[epoch*num_updates+i] = elapseds
            #error_history_train_GN[epoch*num_updates+i] = error_new_train_SGD
            #error_history_test_GN[epoch*num_updates+i] = error_new_test_SGD
            #epochs_GN[epoch*num_updates+i] = epoch

        elapsed = time.time() - t
        print('estimated time for the update step:', elapsed, 'sec')
        if epoch == 0:
            time_vec_GN[epoch] = elapsed
        else:
            time_vec_GN[epoch] = time_vec_GN[epoch - 1] + elapsed


    # prediction-plot of the model:
    x = model.predict(a)
    ax0.plot(a, x, label='Prediction GN', c='orange')

#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//train_time_SGD.npy',
#            train_time_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//error_history_train_SGD.npy',
#           error_history_train_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//error_history_test_SGD.npy',
#           error_history_test_SGD)
#np.savetxt('/Users/niklasbrunn/Desktop/Numopt_Werte//epochs_SGD.npy',
#           epochs_SGD)

ax0.set_ylim(-0.6, 10)
ax0.set_xlim(-np.sqrt(10), np.sqrt(10))
ax0.set_title('Data and Predictions')
ax0.legend(loc='upper right')


#######
#PLOTS:
#######
####Train_loss_epochs_plot:
ax1.plot(epoch_vec_SGD, train_loss_vec_SGD, 'r',label='SGD', linewidth=0.8)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train-Loss')
ax1.set_title('Train-Loss per Epochs:')
ax1.set_ylim(-0.005, 0.1)
ax1.set_xlim(0.5, epochs)

if GN_allowed == True:
    ax1.plot(epoch_vec_GN, train_loss_vec_GN, 'b', label='GN', linewidth=0.8)

ax1.legend(loc='upper right')

####Test_loss_epochs_plot:
ax2.plot(epoch_vec_SGD, test_loss_vec_SGD, 'r',label='SGD', linewidth=0.8)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Test-Loss')
ax2.set_title('Test-Loss per Epochs:')
ax2.set_ylim(-0.005, 0.1)
ax2.set_xlim(0.5, epochs)

if GN_allowed == True:
    ax2.plot(epoch_vec_GN, test_loss_vec_GN, 'b', label='GN', linewidth=0.8)

ax2.legend(loc='upper right')

####Train_loss_time_plot:
ax3.plot(time_vec_SGD, train_loss_vec_SGD, 'r',label='SGD', linewidth=0.8)
ax3.set_xlabel('Time (in seconds)')
ax3.set_ylabel('Train-Loss')
ax3.set_title('Train-Loss per Time:')
ax3.set_ylim(-0.005, 0.1)

if GN_allowed == True:
    ax3.plot(time_vec_GN, train_loss_vec_GN, 'b', label='GN', linewidth=0.8)

ax3.legend(loc='upper right')

####Test_loss_time_plot:
ax4.plot(time_vec_SGD, test_loss_vec_SGD, 'r', label='SGD', linewidth=0.8)
ax4.set_xlabel('Time (in seconds)')
ax4.set_ylabel('Test-Loss')
ax4.set_title('Test-Loss per Time:')
ax4.set_ylim(-0.005, 0.1)

if GN_allowed == True:
    ax4.plot(time_vec_GN, test_loss_vec_GN, 'b', label='GN', linewidth=0.8)

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
