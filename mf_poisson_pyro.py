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
        u_a0 = Variable(torch.ones(self.data_size[0], self.K))
        u_b0 = Variable(torch.ones(self.data_size[0], self.K))

        v_a0 = Variable(torch.ones(self.data_size[1], self.K))
        v_b0 = Variable(torch.ones(self.data_size[1], self.K))

        u = pyro.sample('u', dist.gamma, u_a0, u_b0)
        v = pyro.sample('v', dist.gamma, v_a0, v_b0)
        s = pyro.sample('s', dist.poisson, F.relu(torch.matmul(u, torch.t(v))))

        for i in pyro.irange("row_loop", self.data_size[0]):
            for j in pyro.irange("col_loop", self.data_size[1]):
                if data[i, j] == 0:
                    continue
                pyro.observe("obs_{}{}".format(i, j),
                             dist.poisson,
                             data[i, j],
                             s[i, j]
                             )

    def guide(self, data):
        log_qu_a0 = Variable(torch.randn(self.data_size[0], self.K), requires_grad=True)
        log_qu_b0 = Variable(torch.randn(self.data_size[0], self.K), requires_grad=True)

        log_qv_a0 = Variable(torch.randn(self.data_size[1], self.K), requires_grad=True)
        log_qv_b0 = Variable(torch.randn(self.data_size[1], self.K), requires_grad=True)

        log_qu_a0 = pyro.param('log_qu_a0', log_qu_a0)
        log_qu_b0 = pyro.param('log_qu_b0', log_qu_b0)

        log_qv_a0 = pyro.param('log_qv_a0', log_qv_a0)
        log_qv_b0 = pyro.param('log_qv_b0', log_qv_b0)

        qu_a0, qu_b0 = torch.exp(log_qu_a0), torch.exp(log_qu_b0)
        qv_a0, qv_b0 = torch.exp(log_qv_a0), torch.exp(log_qv_b0)

        u = pyro.sample('u', dist.normal, qu_a0, qu_b0)
        v = pyro.sample('v', dist.normal, qv_a0, qv_b0)
        pyro.sample('s', dist.poisson, F.softplus(torch.matmul(u, torch.t(v))))


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
