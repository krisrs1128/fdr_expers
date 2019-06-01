#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
from torch.optim import Optimizer, SGD

class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.
        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn"t specify them).
    """
    def __init__(self, params):
        super(Extragradient, self).__init__(params)
        self.params_copy = []

    def update(self, p, group):
        raise NotImplementedError

    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current
          parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group["params"]:
                u = self.update(p, group)
                if is_empty:
                    self.params_copy.append(p.data.clone())

                p.data.add_(u)

    def step(self):
        """Performs a single optimization step.
        """
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                u = self.update(p, group)
                p.data = self.params_copy[i].add_(u)
                i += 1

        # Free the old parameters
        self.params_copy = []


class ExtraSGD(Extragradient):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).
    Nesterov momentum is based on the formula from
    """
    def __init__(self,
                 params,
                 lr,
                 weight_decay=0):
        super(ExtraSGD, self).__init__(params)
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay
        }

    def update(self, param, group):
        grad = param.grad.data
        if weight_decay != 0:
            grad += group["weight_decay"] * param.data

        return -group["lr"] * grad
