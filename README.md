# Efficient Hessian-Free Optimization of Deep Neural Networks
This project which was developed as a part of the lecture Numerical Optimization, held by Prof. Dr. Moritz Diehl, in the winter term 21/22 at the Albert Ludwigs University of Freiburg.

We provide code in Python3, Tensorflow, for optimizing DNNs with a second-order optimization method.
Our Repository consists of two main files, the Hessian_free_MNIST.py where we implemented the method for the MNIST data set, 
and the Hessian_free_simple.py where we implemented the method for an self generated sin-data set (and also a very simple x^2-data set). The two main files are commented versions of our implementation. 

Also, for our benchmarks we used the two files model.py and train_steps.py where we implemented the Hessian-Free method using list comprehension for some extra computation speed.

The file code_graveyard contains older versions of our implementation and other codelines which may be usefull.
