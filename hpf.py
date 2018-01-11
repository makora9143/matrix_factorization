"""
Scalable Recommendation with Poisson Factorization
https://github.com/premgopalan/hgaprec

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pyro
from pyro.optim import Adam
from pyro.infer import SVI
import pyro.distributions as dist

class PoissonMF(nn.Module):
    def __init__(self, data_size, K):
        super(PoissonMF, self).__init__()
        self.data_size = data_size
        self.K = K

    def model(self, data):
        a_tilda = Variable(torch.ones(self.data_size[0]))
        b_tilda = Variable(torch.ones(self.data_size[0]))

        eps = pyro.sample('eps', dist.gamma, a_tilda, a_tilda / b_tilda)
        eps = torch.cat([eps]*self.K, 1)

        a = Variable(torch.ones(self.data_size[0], self.K))
        theta = pyro.sample('theta', dist.gamma, a, eps)

        c_tilda = Variable(torch.ones(self.data_size[1]))
        d_tilda = Variable(torch.ones(self.data_size[1]))

        eta = pyro.sample('eta', dist.gamma, c_tilda, c_tilda / d_tilda)
        eta = torch.cat([eta] * self.K, 1)

        c = Variable(torch.ones(self.data_size[1], self.K))
        beta = pyro.sample('beta', dist.gamma, c, eta)

        zeta = pyro.sample('zeta', dist.poisson, torch.matmul(theta, torch.t(beta)))

        for i in range(self.data_size[0]):
            for j in range(self.data_size[1]):
                if data[i, j] == 0:
                    continue
                pyro.observe("obs_{}{}".format(i, j),
                             dist.poisson,
                             data[i, j],
                             zeta[i, j]
                             )

    def guide(self, data):
        lamb  = Variable(torch.randn(self.data_size[1], self.K), requires_grad=True)
        gamma = Variable(torch.randn(self.data_size[0], self.K), requires_grad=True)
        kappa = Variable(torch.randn(self.data_size[0]), requires_grad=True)
        tau   = Variable(torch.randn(self.data_size[1]), requires_grad=True)
        phi   = Variable(torch.ones(self.data_size) / self.K, requires_grad=True)

        lamb  = pyro.param('lamb', lamb)
        gamma = pyro.param('gamma', gamma)
        kappa = pyro.param('kappa', kappa)
        tau   = pyro.param('tau', tau)
        phi   = pyro.param('phi', phi)

        pyro.sample('beta', dist.gamma, lamb, lamb)
        pyro.sample('theta', dist.gamma, gamma, gamma)
        pyro.sample('eps', dist.gamma, kappa, kappa)
        pyro.sample('eta', dist.gamma, tau, tau)
        # https://qiita.com/gilbert_yumu/items/5ed30b695dc42985445b
        pyro.sample('zeta', dist.multinomial, phi)


if __name__ == '__main__':
    R = Variable(torch.FloatTensor([
                [5, 3, 0, 1],
                [4, 0, 0, 1],
                [1, 1, 0, 5],
                [1, 0, 0, 4],
                [0, 1, 5, 4],
                ]
            ))

    mf = PoissonMF(R.size(), 5)

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)
    epochs = 10000

    svi = SVI(mf.model, mf.guide, optimizer, loss="ELBO", num_particles=10)

    losses = list()
    for step in range(epochs):
        loss = svi.step(R)
        losses.append(loss)
        if step % 100 == 0:
            print(step, loss)

    qu = pyro.param('qu_mu')
    qv = pyro.param('qv_mu')

    print(R)
    print(torch.matmul(qu, torch.t(qv)))
    import matplotlib.pyplot as plt

    plt.plot(list(range(epochs)), losses)
    plt.show()
