# bilinearRegressionFitting

Matlab code for bilinear least-squares regression

-----

This repo contains code for least squares and ridge regression problems where all or part of the regression weights are parametrized linearly.  Formally this solves the problem:

$\hat w = \arg \min_{\vec w} || \vec Y - X \vec w||^2_2 + \lambda ||\vec w||^2_2$, 

subject to the constraint that $\vec w = \mathrm{vec}(UV^\top)$, for some matrices $U$ and $V$, where $Y$ are the outputs, $X$ is the design matrix, and $\lambda$ is the ridge parameter.

It also allows for "mixed" bilinear regression, where we allow some coefficients of the regression weights $\vec w$ to be parametrized bilinearly, while others are parametrized linearly.


