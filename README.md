# Probabilistic Matrix Factoriazation

This repository is implemented with Pytorch or Pyro.

- Gaussian Matrix Factoriazation (Pytorch, [WIP]Pyro)
- Poisson Matrix Factoriazation (Pytorch)


## Gaussian Matrix Factorization
<img src="https://latex.codecogs.com/gif.latex?\mathbf{X}&space;$\sim$&space;p(\mathbf{X}&space;|&space;\mathbf{U},&space;\mathbf{V})&space;\\&space;p(\mathbf{X}&space;|&space;\mathbf{U},&space;\mathbf{V})&space;$=$&space;\mathcal{N}(\mathbf{X}&space;|&space;\mathbf{U}^T&space;\mathbf{V},&space;\mathbf{I})">

![ELBO](https://raw.githubusercontent.com/makora9143/probabilistic_matrix_factorization/images/Gaussian_MF_ELBO.png)
