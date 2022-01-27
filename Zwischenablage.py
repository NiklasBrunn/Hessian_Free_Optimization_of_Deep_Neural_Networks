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
