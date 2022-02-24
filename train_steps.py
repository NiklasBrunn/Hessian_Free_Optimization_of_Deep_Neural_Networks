import numpy as np
import tensorflow as tf

# Model Loss functions
def model_loss(y_true, y_pred):
    if tf.shape(y_true)[-1] == 10:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))
    else:
        return tf.reduce_mean(0.5 * (y_true - y_pred) ** 2)

# SGD update step
def train_step_gradient_descent(model, x_batch, y_batch, eta):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x_batch)
        loss = model_loss(y_batch, y_pred)

    grad_obj = tape.gradient(loss, model.trainable_variables)

    update = [tf.constant(eta) * g for g in grad_obj]

    model.set_weights([p - u for (p, u) in zip(model.trainable_variables, update)])

# Fast matrix-vector product using BAD and FAD
def fastmatvec(model, x_batch, y_batch, v, lam):
    with tf.GradientTape() as tape:
        with tf.autodiff.ForwardAccumulator(model.trainable_variables, v) as acc:
            y_pred = model(x_batch)
            akt_out = tf.nn.softmax(y_pred)
        HJv = acc.jvp(akt_out)
    J_tHJv = tape.gradient(y_pred, model.trainable_variables,
                           output_gradients=tf.stop_gradient(HJv))
    batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
    return [arg_1 / batch_size + lam * arg_2 for arg_1, arg_2 in zip(J_tHJv, v)]

# Hessian free update step
def train_step_hessian_free(model, x_batch, y_batch, lam, v, preconditioning, min_CG_steps, eps, r_value):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x_batch)
        loss = model_loss(y_batch, y_pred)
    grad_obj = tape.gradient(loss, model.trainable_variables)

    # CG-Method
    J_tHJv = fastmatvec(model, x_batch, y_batch, v, lam)
    r = [arg_1 - arg_2 for arg_1, arg_2 in zip(grad_obj, J_tHJv)]
    if preconditioning == True:
        M = [(arg_1 ** 2 + lam) ** 0.75 for arg_1 in grad_obj]
    else:
        M = [tf.ones(tf.shape(t)) for t in model.trainable_variables]
    y = [arg_1 / arg_2 for arg_1, arg_2 in zip(r, M)]
    d = y
    i, j, s = 0, min_CG_steps, min_CG_steps
    phi_history = 0.5 * tf.reduce_sum([tf.reduce_sum((arg_1 - 2.0 * arg_2) * arg_3)
                                       for arg_1, arg_2, arg_3 in zip(J_tHJv, grad_obj, v)])
    r_dot_y = tf.reduce_sum([tf.reduce_sum(arg_1 * arg_2) for arg_1, arg_2 in zip(r, y)])

    while i <= j or s >= eps * j or phi_history[-1] >= 0:
        j = np.maximum(min_CG_steps, int(i/min_CG_steps))
        z = fastmatvec(model, x_batch, y_batch, d, lam)
        d_dot_z = tf.reduce_sum([tf.reduce_sum(arg_1 * arg_2) for arg_1, arg_2 in zip(d, z)])
        alpha = r_dot_y / d_dot_z
        z = [alpha * arg_1 for arg_1 in z]
        J_tHJv = [arg_1 + arg_2 for arg_1, arg_2 in zip(J_tHJv, z)]
        v = [arg_1 + alpha * arg_2 for arg_1, arg_2 in zip(v, d)]
        r = [arg_1 - arg_2 for arg_1, arg_2 in zip(r, z)]
        y = [arg_1 - arg_2 / arg_3 for arg_1, arg_2, arg_3 in zip(y, z, M)]
        r_dot_y_new = tf.reduce_sum([tf.reduce_sum(arg_1 * arg_2) for arg_1, arg_2 in zip(r, y)])
        d = [arg_1 + r_dot_y_new / r_dot_y * arg_2 for arg_1, arg_2 in zip(y, d)]
        r_dot_y = r_dot_y_new
        phi = 0.5 * tf.reduce_sum([tf.reduce_sum((arg_1 - 2.0 * arg_2) * arg_3)
                                   for arg_1, arg_2, arg_3 in zip(J_tHJv, grad_obj, v)])
        phi_history = np.append(phi_history, phi)
        if i >= j:
            s = 1 - phi_history[-j] / phi_history[-1]
        i += 1

    model.set_weights([p - u for (p, u) in zip(model.trainable_variables, v)])

    # Updating lambda
    impr = loss - model_loss(y_batch, model(x_batch))

    cor = tf.reduce_sum([tf.reduce_sum((2.0 * arg_1 - 0.5 * lam * arg_2) * arg_2)
                         for arg_1, arg_2 in zip(grad_obj, v)])

    rho = impr / (phi_history[-1] + cor)

    if rho > 0.75:
        lam /= r_value
    elif rho < 0.25:
        lam *= r_value
    return np.array(lam), v
