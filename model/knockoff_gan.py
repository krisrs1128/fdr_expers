#!/usr/bin/env python
"""
Implements the Baseline Knockoff Gan
"""
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, K=20, H=20, J=10):
        super(Generator, self).__init__()
        self.J = J
        self.layers = nn.Sequential(
            nn.Linear(K, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, J)
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, J=10, H=20):
        super(Discriminator, self).__init__()
        self.J = J
        self.layers = nn.Sequential(
            nn.Linear(J, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 1)
        )

    def forward(self, x):
        return self.layers(x)
