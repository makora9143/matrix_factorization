# Probabilistic Matrix Factorization

This repository is implemented with Pytorch or Pyro.

- Gaussian Matrix Factoriazation (Pytorch, [WIP]Pyro)
- Poisson Matrix Factoriazation (Pytorch)


## Gaussian Matrix Factorization
<img src="https://latex.codecogs.com/gif.latex?\mathbf{X}&space;$\sim$&space;p(\mathbf{X}&space;|&space;\mathbf{U},&space;\mathbf{V})&space;\\&space;p(\mathbf{X}&space;|&space;\mathbf{U},&space;\mathbf{V})&space;$=$&space;\mathcal{N}(\mathbf{X}&space;|&space;\mathbf{U}^T&space;\mathbf{V},&space;\mathbf{I})">

### ELBO against Gaussian via Pytorch

![ELBO_gaussian_pytorch](https://raw.githubusercontent.com/makora9143/pytorch_pyro_pmf/images/mf_gaussian_pytorch.png)

### ELBO against Gaussian via Pyro

![ELBO_gaussian_pyro](https://raw.githubusercontent.com/makora9143/probabilistic_matrix_factorization/images/Gaussian_MF_ELBO.png)

### ELBO against Poisson via Pytorch

![ELBO_poisson_pytorch](https://raw.githubusercontent.com/makora9143/pytorch_pyro_pmf/images/mf_poisson_pytorch.png)
