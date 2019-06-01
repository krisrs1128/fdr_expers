#!/usr/bin/env python
import torch

def train(iterator, optimizers, models, loss_fun, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim_z = models["G"].layers[0].in_features
    losses = []

    models["G"].zero_grad()
    models["D"].zero_grad()

    for i, data in enumerate(iterator):
        ############################
        # (1) Update D network: minimize -(E[log(D(x))] + E[log(1 - D(G(z)))])
        ###########################
        # train with real
        y = torch.full((iterator.batch_size,), 1, device=device)
        l0 = loss_backward(models["D"], loss_fun, data, y)

        # train with fake
        z = torch.randn(iterator.batch_size, dim_z, device=device)
        x_tilde = models["G"](z)
        l1 = loss_backward(models["D"], loss_fun, x_tilde.detach(), y.fill_(0))
        extragradient_step(optimizers["D"], models["D"], i)

        ############################
        # (2) Update G network: minimize -E[log(1 - D(G(z)))]
        ###########################
        l2 = loss_backward(models["D"], loss_fun, x_tilde, y.fill_(1))
        extragradient_step(optimizers["G"], models["G"], i)

        if i % 50 == 0:
            losses.append([l0.item(), l1.item(), l2.item()])
            print("Loss {}: {}".format(i, losses[-1]))

    return models, optimizers, np.array(losses)


def loss_backward(model, loss_fun, x, y):
    y_hat = model(x)
    loss = loss_fun(y_hat.squeeze(), y)
    loss.backward()
    return loss

def extragradient_step(optimizer, model, i):
    if i % 2 == 0:
        optimizer.extrapolate()
    else:
        optimizer.step()
        model.zero_grad()
