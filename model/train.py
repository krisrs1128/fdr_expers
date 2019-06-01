#!/usr/bin/env python
import torch

def train(iterator, optimizers, models, loss_fun, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim_z = models["G"].layers[0].in_features
    losses = np.zeros((len(iterator.dataset), 3))


    for i, data in enumerate(iterator):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        models["D"].zero_grad()
        y = torch.full((iterator.batch_size,), 1, device=device)
        losses[i][0] = loss_backward(models["D"], loss_fun, data, y)

        # train with fake
        z = torch.randn(iterator.batch_size, dim_z, device=device)
        x_tilde = models["G"](z)
        losses[i][1] = loss_backward(models["D"], loss_fun, x_tilde.detach(), y.fill_(0))
        optimizers["D"].step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        models["G"].zero_grad()
        losses[i][2] = loss_backward(models["D"], loss_fun, x_tilde, y.fill_(1))
        optimizers["G"].step()

        # if i % 0 == 0:
        print("Loss {}: {}".format(i, losses[i]))

    return models, optimizers, losses


def loss_backward(model, loss_fun, x, y):
    y_hat = model(x)
    loss = loss_fun(y_hat.squeeze(), y)
    loss.backward()
    return loss
