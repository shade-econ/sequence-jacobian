from collections import OrderedDict

import numpy as np
from numpy import random
from scipy import stats


## BASE CLASS #################################################################


class Distribution:
    """
    Base class for probability distributions
    """

    dim = 1

    def logpdf(self, x):
        raise NotImplementedError
    
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    
    def rand(self, size=None):
        return NotImplementedError


## UNIVARIATE DISTRIBUTIONS ###################################################


class Normal(Distribution):
    """
    Normal distribution class, parameterized by mean and squared deviation
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        return stats.norm.logpdf(
            x,
            loc = self.mu,
            scale = self.sigma
        )

    def rand(self, size=None):
        return random.normal(
            loc = self.mu,
            scale = self.sigma,
            size = (size,)
        )


class Gamma(Distribution):
    """
    Gamma distribution class, parameterized by shape and scale
    """
    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta

    def logpdf(self, x):
        return stats.gamma.logpdf(
            x,
            self.alpha,
            scale = self.theta
        )

    def rand(self, size=None):
        return random.gamma(
            self.alpha,
            scale = self.theta,
            size = size
        )

    
class Uniform(Distribution):
    """
    Uniform distribution, parameterized by an upper and lower bound
    """
    def __init__(self, lb=0.0, ub=1.0):
        self.lb = lb
        self.ub = ub

    def logpdf(self, x):
        return stats.uniform.logpdf(
            x,
            loc = self.lb,
            scale = self.ub - self.lb
        )
    
    def rand(self, size=None):
        return random.uniform(
            low = self.lb,
            high = self.ub,
            size = size
        )


## MULTIVARIATE DISTRIBUTIONS #################################################


class Product(Distribution):
    """
    Product of univariate distributions
    """
    def __init__(self, *dists):
        self.dists = dists
        self.dim = len(dists)

    def logpdf(self, x):
        return sum([
            dist.logpdf(x[..., i]) for i, dist in enumerate(self.dists)
        ])
    
    def rand(self, size=None):
        return np.stack(
            [dist.rand(size=size) for dist in self.dists],
            axis = 1
        )


def IID(dist, k):
    """
    Independent and identically drawn from a common density function
    """
    return Product(
        *[dist for _ in range(k)]
    )


## STRUCTURED DISTRIBUTIONS ###################################################


class Conditional(Distribution):
    """
    Conditional distribution based on Nicolas Chopin's python module
    """
    def __init__(self, dist, dim=1, dtype="float64"):
        self.dim = dim
        self.dist = dist
        self.dtype = dtype

    def __call__(self, x):
        return self.dist(x)


class Prior(Distribution):
    """
    Construct prior distribution
    """
    def __init__(self, dists):
        if isinstance(dists, OrderedDict):
            self.dists = dists
        elif isinstance(dists, dict):
            self.dists = OrderedDict(
                [(key, dists[key]) for key in sorted(dists.keys())]
            )
        else:
            raise TypeError("must be recast as a dictionary")

    def _rand(self):
        out = {}
        for param, dist in self.dists.items():
            cond_dist = dist(out) if callable(dist) else dist
            out[param] = cond_dist.rand(size=1)[0]

        return out
    
    def rand(self, size=None):
        if size is None:
            return self._rand()
        else:
            return [self._rand() for _ in range(size)]
        
    def logpdf(self, x):
        logprob = 0.0
        for param, dist in self.dists.items():
            cond_dist = dist(x) if callable(dist) else dist
            logprob += cond_dist.logpdf(x[param])

        return logprob