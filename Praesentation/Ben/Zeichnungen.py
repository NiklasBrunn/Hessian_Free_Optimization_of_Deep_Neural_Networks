from random import choice, uniform
from math import sin, cos, sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np

#a=10
#plt.plot(list(range(a)),[0]+list(np.cumsum(np.random.normal(0,1,a-1))))

fig, axs = plt.subplots(1,1)

axs.plot([0,1],[0,0.2])
axs.set_ylim([-0.5, 1])

Y = np.random.normal(0.2,0.3,6)
for y in Y:
    axs.plot([0,1],[0,y], linestyle = '--', color='Orange')

plt.show()
