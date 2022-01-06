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
'''
