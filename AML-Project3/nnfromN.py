#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import itertools as it

A = list(range(1, 9))
B = list(range(1, 9))
#
#    yv = tf.zeros(num_classes)
#    yv[y - 1] = 1

# Params
learning_rate = 0.1
horizon = 500
batch_size = 128

# Network Params
nph = 3 # neurons per hidden layers
num_hidden = 1
num_inputs = 8
num_classes = 8

# tf Graph input
#X = 1:8
#Y = 1:8

def pairs(vector):
    return zip(vector[:-1], vector[1:])

# Network topology
network = [num_inputs] + [nph for x in range(num_hidden)] + [num_classes]

# Weights (of each layer)
W  = [tf.Variable(tf.random_normal([p, q])) for p, q in pairs(network)]
dW = [tf.Variable(tf.random_normal([p, q])) for p, q in pairs(network)]

# Biases (of each layer)
b  = [tf.Variable(tf.random_normal([p]))    for p, q in pairs(network)]
db = [tf.Variable(tf.random_normal([p]))    for p, q in pairs(network)]

# X and activations
a  = [tf.Variable(tf.zeros([q]))            for p, q in pairs(network)]
da = [tf.Variable(tf.zeros([q]))            for p, q in pairs(network)]

print(network)
print(W)
print(b)
print(X)

# Prediction
def linear(x, A, b):
    tf.add(tf.matmul(x, A, b), b)

def forward(x):
    a[0] = linear(x, W[0], b[0]) # First hidden layer
    for i in range(len(network) - 1): # Subsequent hidden layers + output layer
        a[i + 1] = linear(sigm(a[i]), W[i + 1], b[i + 1])

def bprop(y):
    # Calculate delta and weight updates for output layer
    da[-1] = tf.negative(tf.subtract(y, sigm(X[-1]))) # 1xN(outputs)
    dW[-1] = tf.matmult(da[-1], tf.transpose(sigm(X[-1])))
    db[-1] = da[-1]
    W[-1]  = tf.add(W[-1], tf.multiply(learning_rate, dW[-1]))
    b[-1]  = tf.add(b[-1], tf.multiply(learning_rate, db[-1]))
    for i in range(len(network) - 1, -1, -1):
        # Calculate delta and weight updates for subsequent layers
        da[i] = tf.multiply(tf.matmult(tf.transpose(W[i + 1]), X[i + 1]), sigmd(X[i]))
	dW[i] = tf.matmult(da[i], tf.transpose(sigm(X[i])))
	db[i] = da[i]
	W[i]  = tf.add(W[i], tf.multiply(learning_rate, dW[i]))
	b[i]  = tf.add(b[i], tf.multiply(learning_Rate, db[i]))

# Activation functions
def sigm(a):
    return 1. / (1 + np.exp(np.negative(a)))

# Activation function derivatives
def sigmd(a):
    return np.multiply(sigm(a), (1 - sigm(a)))

# Loss functions
def cross_entropy(y, o):
    x0 = np.multiply(y, np.log(o))
    x1 = np.multiply(1 - y, np.log(1 - o))
    return -np.sum(x0, x1)