from random import choice, uniform
from math import sin, cos, sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

Tau = 1
Seconds = 1
m = 1000


Y = [0]+list(np.random.normal(0,1,m))
S = np.cumsum(Y)/sqrt(m)

for i in range(m):
    plt.clf()
    xl = np.linspace(0,Tau*i/m,i+1)
    plt.plot(xl, S[:i+1])
    #plt.axis([0, i, -2*sqrt(i), 2*sqrt(i)])
    plt.draw()
    plt.pause(Seconds/m)
