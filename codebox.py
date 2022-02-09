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
