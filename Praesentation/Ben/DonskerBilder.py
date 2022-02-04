from random import choice, uniform
from math import sin, cos, sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np

Tau = 1

'''
# 1 x 4
m_seq = (2**3,2**5,2**7,2**9)
Y = np.random.normal(0,1,max(m_seq))
Y = [0]+list(Y)

fig,axs = plt.subplots(1,4)
for i in range(len(m_seq)):
    m = m_seq[i]
    xl = np.linspace(0,Tau,m)
    W = np.cumsum(Y[:m])/sqrt(m)
    axs[i].plot(xl, W)
    axs[i].axis([0, Tau, -2, 2])
    axs[i].set_title('m={}'.format(m))
'''

# 1 x 1
m = 1000
Y = np.random.normal(0,1,m)
Y = [0]+list(Y)

fig,axs = plt.subplots(1,1)
xl = np.linspace(0,Tau,m)
W = np.cumsum(Y[:m])/sqrt(m)
axs.plot(xl, W)
axs.axis([0, Tau, -2, 2])
#axs[i].set_title('m={}'.format(m))
axs.axis('off')



'''
# 2 x 4
m_seq = (2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10)
Y = np.random.normal(0,1,max(m_seq))

fig,axs = plt.subplots(2,4)
for i in range(len(m_seq)):
    m = m_seq[i]
    xl = np.linspace(0,Tau,m)
    W = np.cumsum(Y[:m])/sqrt(m)
    axs[i//4,i%4].plot(xl, W)
    axs[i//4,i%4].axis([0, Tau, -2, 2])
'''


plt.show()
