#!/usr/bin/env python

import numpy as np

n = 1000
p = 10
z = np.random.normal(size=(n, 1))

f1 = lambda x: x ** 2 + 2 * x
f2 = lambda x: x ** 3 - x

def poly(coefs):
    def f(x):
        result = np.zeros((x.shape[0], 1))
        for j, alpha in enumerate(coefs):
            result += alpha * x ** j
        return result
    return f

def add_noise(f, sigma=0.8):
    def g(x):
        n = x.shape[0]
        return f(x) + np.random.normal(0, sigma, (n, 1))
    return g

f1 = add_noise(poly([2, -2, 1]))
f2 = add_noise(poly([-1, 0, 1, 0.1]))
x = np.hstack([z, f1(z), f2(z), np.random.normal(size=(n, 5))])
