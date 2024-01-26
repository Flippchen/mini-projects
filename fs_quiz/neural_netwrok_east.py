import numpy as np

i1, i2 = 0.05, 0.1
w1, w2, w3, w4, w5, w6, w7, w8 = 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
b1, b2 = 0.3, 0.6


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


h1 = sigmoid(w1 * i1 + w2 * i2 + b1)
h2 = sigmoid(w3 * i1 + w4 * i2 + b1)

o1 = sigmoid(w5 * h1 + w6 * h2 + b2)
o2 = sigmoid(w7 * h1 + w8 * h2 + b2)

print(o1, o2)
