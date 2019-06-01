#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
import pandas as pd

# simulate some data
n = 1024 * 10
p = 10
z = np.random.normal(size=(n, 1))

f1 = add_noise(poly([2, -2, 1]))
f2 = add_noise(poly([-1, 0, 1, 0.1]))
x = np.hstack([z, f1(z), f2(z), np.random.normal(size=(n, 7))])
n_epochs = 100

models = {"D": Discriminator(), "G": Generator()}

xset = XSim(x)
iterator = DataLoader(xset, batch_size=32)

# test using the extra sgd
optimizers = {
    "D": ExtraSGD(models["D"].parameters(), lr=1e-4),
    "G": ExtraSGD(models["G"].parameters(), lr=1e-4)
}

losses = []
for epoch in range(n_epochs):
    models, optimizers, losses_i = train(iterator, optimizers, models, loss_fun)
    losses_i = pd.DataFrame(losses_i)
    losses_i["epoch"] = epoch
    losses.append(losses_i)

losses = pd.concat(losses)
losses.columns = ["D0", "D1", "G", "epoch"]
losses.to_csv("losses.csv", index=False)
