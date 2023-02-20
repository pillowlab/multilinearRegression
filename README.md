# bilinearRegressionFitting

Matlab code for bilinear and trilinear least-squares regression

-----

This repository contains code for least squares and ridge regression problems where all or part of the regression weights are parametrized bilinearly or trilinearly.  Formally, bilinear regression solves the problem:

$\hat w = \arg \min_{\vec w} || \vec Y - X \vec w||^2_2 + \lambda ||\vec w||^2_2$, 

subject to the constraint that $\vec w = \mathrm{vec}(UV^\top)$, for some matrices $U$ and $V$, where $\vec Y$ are the outputs, $X$ is the design matrix, and $\lambda$ is the ridge parameter.


### Problem settings ###

- "mixed" bilinear regression, where we allow some coefficients of the regression weights $\vec w$ to be parametrized bilinearly, while others are parametrized linearly.

- "multiple" bilinear regression, where we allow different bilinear parametrizations (e.g., with different rank) for different segments of the regression weights

- trilinear regression, where the regression weights are parametrized by a low-rank 3rd order tensor.

### Algorithms ###

There are implementations of two different methods for solving the optimization problem for $\hat w$:

- **Alternating coordinate ascent** - this involves alternating between closed-form updates for $U$ and $V$ until convergence.

- **Joint ascent** - direct simultaneous gradient ascent on $U$ and $V$. 

See `demo1_testBilinearRegress.m` for a speed comparison; the optimal method seems to depend on the choice of dimensions and rank.
