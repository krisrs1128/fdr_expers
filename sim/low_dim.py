#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset


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


class XSim(Dataset):
    def __init__(self, x):
        super(XSim, self).__init__()
        self.x = torch.Tensor(x).float()
        self.J = x.shape[1]

    def __getitem__(self, ix):
        return self.x[ix, :]

    def __len__(self):
        return self.x.shape[0]
