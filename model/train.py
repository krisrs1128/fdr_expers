
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(iterator, optimizers, models, loss_fun, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, J = next(iter(iterator)).shape

    for i, data in enumerate(iterator):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        models["D"].zero_grad()
        label = torch.full((batch_size,), 1, device=device)

        y_hat = models["D"](real_cpu)
        loss = loss_fun(y_hat, label)
        loss.backward()

        # train with fake
        z = torch.randn(batch_size, dim_z, device=device)
        x_tilde = models["G"](noise)
        label.fill_(0)
        y_hat = netD(z.detach())
        loss = loss_fun(y_hat, label)
        loss.backward()
        optimizers["D"].step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        models["G"].zero_grad()
        label.fill_(0)  # fake labels are real for generator cost
        y_hat = models["D"](fake)
        loss = loss_fun(output, label)
        loss.backward()
        optimizers["G"].step()
