import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib as mpl
from matplotlib import pyplot as plt


input = tf.constant([[1.]])
y = input**2

hidden_dim_1 = 3
hidden_dim_2 = 3

weights_l1 = tf.Variable(tfp.distributions.Normal(loc=0, scale=1).sample((hidden_dim_1, 1)))
bias_l1 = tf.Variable(tfp.distributions.Normal(loc=0, scale=1).sample((hidden_dim_1, 1)))


weights_l2 = tf.Variable(tfp.distributions.Normal(
    loc=0, scale=1).sample((hidden_dim_2, hidden_dim_1)))
bias_l2 = tf.Variable(tfp.distributions.Normal(loc=0, scale=1).sample((hidden_dim_2, 1)))


weights_l3 = tf.Variable(tfp.distributions.Normal(loc=0, scale=1).sample((1, hidden_dim_2)))
bias_l3 = tf.Variable(tfp.distributions.Normal(loc=0, scale=1).sample((1, 1)))


with tf.GradientTape() as tape:
    act_l1 = tf.nn.relu(tf.matmul(weights_l1, input) + bias_l1)
    act_l2 = tf.nn.relu(tf.matmul(weights_l2, act_l1) + bias_l2)
    output = tf.matmul(weights_l3, act_l2) + bias_l3
    loss = 0.5 * (y - output) ** 2

dl_dw_l1, dl_dw_l2, dl_dw_l3, dl_db_l1, dl_db_l2, dl_db_l3 = tape.gradient(
    loss, [weights_l1, weights_l2, weights_l3, bias_l1, bias_l2, bias_l3])

weights_l1 = weights_l1 + dl_dw_l1 * tf.constant(0.01)

print(weights_l1)
#print('Gradient Weights Layer 1:', dl_dw_l1, dl_db_l1, dl_dw_l2, dl_db_l2, dl_dw_l3, dl_db_l3)
