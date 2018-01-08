# Probabilistic Matrix Factoriazation

This repository is implemented with Pytorch or Pyro.

- Gaussian Matrix Factoriazation (Pytorch, [WIP]Pyro)
- Poisson Matrix Factoriazation (Pytorch)


## Gaussian Matrix Factorization

```math
\begin{eqnarray}
\mathbf{X} $\sim$ p(\mathbf{X} | \mathbf{U}, \mathbf{V}) \\
$=$ \mathcal{N}(\mathbf{X} | \mathbf{U}^T \mathbf{V}, \mathbf{I})
\end{eqnarray}
```

![ELBO](https://raw.githubusercontent.com/makora9143/probabilistic_matrix_factorization/images/Gaussian_MF_ELBO.png)
