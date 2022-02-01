import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from keras.datasets import mnist

import logging
tf.get_logger().setLevel(logging.ERROR)

from sys import path
path.append(r"/Users/niklasbrunn/Desktop/CasADi_Python/casadi-osx-py39-v3.5.5")
from casadi import *
#x = MX.sym("x")
#print(jacobian(sin(x),x))


# Symbols/expressions
x = MX.sym('x')
y = MX.sym('y')
z = MX.sym('z')
f = x**2+100*z**2
g = z+(1-x)**2-y

nlp = {}                 # NLP declaration
nlp['x']= vertcat(x,y,z) # decision vars
nlp['f'] = f             # objective
nlp['g'] = g             # constraints

# Create solver instance
F = nlpsol('F','ipopt',nlp);

# Solve the problem using a guess
F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)
