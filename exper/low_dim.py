#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

f1 = add_noise(poly([2, -2, 1]))
f2 = add_noise(poly([-1, 0, 1, 0.1]))
x = np.hstack([z, f1(z), f2(z), np.random.normal(size=(n, 7))])
n_epochs = 100

models = {
    "D": Discriminator(),
    "G": Generator()
}

optimizers = {
    "D": torch.optim.Adam(models["D"].parameters()),
    "G": torch.optim.Adam(models["G"].parameters())
}

xset = XSim(x)
iterator = DataLoader(xset, batch_size=32)
losses = {}

for epoch in range(n_epochs):
    models, optimizers, losses_i = train(iterator, optimizers, models, loss_fun)
    losses[epoch] = losses_i
