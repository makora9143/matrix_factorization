import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyro
from pyro.optim import Adam
from pyro.infer import SVI
import pyro.distributions as dist

class GaussianMF(nn.Module):
    def __init__(self, data_size, K):
        super(GaussianMF, self).__init__()
        self.data_size = data_size
        self.K = K

    def model(self, data):
        u_mu0 = Variable(torch.zeros(self.data_size[0], self.K))
        u_sigma20 = Variable(torch.ones(self.data_size[0], self.K))

        v_mu0 = Variable(torch.zeros(self.data_size[1], self.K))
        v_sigma20 = Variable(torch.ones(self.data_size[1], self.K))

        u = pyro.sample('u', dist.normal, u_mu0, u_sigma20)
        v = pyro.sample('v', dist.normal, v_mu0, v_sigma20)

        for i in pyro.irange("row_loop", self.data_size[0]):
            for j in pyro.irange("col_loop", self.data_size[1]):
                if data[i, j] == 0:
                    continue
                pyro.observe("obs_{}{}".format(i, j), dist.normal,
                data[i, j],
                torch.dot(u[i, :], v[j, :]),
                Variable(torch.ones(1)))

    def guide(self, data):
        qu_mu0 = Variable(torch.randn(self.data_size[0], self.K), requires_grad=True)
        log_qu_sigma20 = Variable(torch.abs(torch.randn(self.data_size[0], self.K)), requires_grad=True)

        qv_mu0 = Variable(torch.randn(self.data_size[1], self.K), requires_grad=True)
        log_qv_sigma20 = Variable(torch.randn(self.data_size[1], self.K), requires_grad=True)

        qu_mu = pyro.param('qu_mu', qu_mu0)
        log_qu_sigma2 = pyro.param('log_qu_sigma2', log_qu_sigma20)

        qv_mu = pyro.param('qv_mu', qv_mu0)
        log_qv_sigma2 = pyro.param('log_qv_sigma2', log_qv_sigma20)

        qu_sigma2, qv_sigma2 = torch.exp(log_qu_sigma2), torch.exp(log_qv_sigma2)

        pyro.sample('u', dist.normal, qu_mu, qu_sigma2)
        pyro.sample('v', dist.normal, qv_mu, qv_sigma2)


if __name__ == '__main__':
    R = Variable(torch.FloatTensor([
                [5, 3, 0, 1],
                [4, 0, 0, 1],
                [1, 1, 0, 5],
                [1, 0, 0, 4],
                [0, 1, 5, 4],
                ]
            ))

    mf = GaussianMF(R.size(), 5)

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
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
