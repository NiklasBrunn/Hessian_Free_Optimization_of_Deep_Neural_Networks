from random import choice, uniform
from math import sin, cos, sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

Y = [0]
#Z = [0]
#N = 200
T = 5
S = 500

for i in range(S):
    plt.clf()
    Y.append(uniform(-1,1))
    plt.plot(np.cumsum(Y))
    plt.axis([0, i, -2*sqrt(i), 2*sqrt(i)])
    plt.draw()
    plt.pause(T/S)
