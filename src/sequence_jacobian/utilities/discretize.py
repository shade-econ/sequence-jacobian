"""Grids and Markov chains"""

import numpy as np
from scipy.stats import norm


def asset_grid(amin, amax, n):
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1


def agrid(amax, n, amin=0):
    """Create grid between amin and amax that is equidistant in logs."""
    pivot = np.abs(amin) + 0.25
    a_grid = np.geomspace(amin + pivot, amax + pivot, n) - pivot
    a_grid[0] = amin  # make sure *exactly* equal to amin
    return a_grid


# TODO: Temporarily include the old way of constructing grids from ikc_old for comparability of results
def agrid_old(amax, N, amin=0, frac=1/25):
    """crappy discretization method we've been using, generates N point
    log-spaced grid between bmin and bmax, choosing pivot such that 'frac' of
    total log space between log(1+amin) and log(1+amax) beneath it"""
    apivot = (1+amin)**(1-frac)*(1+amax)**frac - 1
    a = np.geomspace(amin+apivot,amax+apivot,N) - apivot
    a[0] = amin
    return a


def nonlinspace(amax, n, phi, amin=0):
    """Create grid between amin and amax. phi=1 is equidistant, phi>1 dense near amin. Extra flexibility may be useful in non-convex problems in which policy functions have nonlinear (even non-monotonic) sections far from the borrowing limit."""
    a_grid = np.zeros(n)
    a_grid[0] = amin
    for i in range(1, n):
        a_grid[i] = a_grid[i-1] + (amax - a_grid[i-1]) / (n-i)**phi 
    return a_grid


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def mean(x, pi):
    """Mean of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * x)


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)


def std(x, pi):
    """Standard deviation of discretized random variable with support x and probability mass function pi."""
    return np.sqrt(variance(x, pi))


def cov(x, y, pi):
    """Covariance of two discretized random variables with supports x and y common probability mass function pi."""
    return np.sum(pi * (x - mean(x, pi)) * (y - mean(y, pi)))


def corr(x, y, pi):
    """Correlation of two discretized random variables with supports x and y common probability mass function pi."""
    return cov(x, y, pi) / (std(x, pi) * std(y, pi))


def markov_tauchen(rho, sigma, N=7, m=3, normalize=True):
    """Tauchen method discretizing AR(1) s_t = rho*s_(t-1) + eps_t.

    Parameters
    ----------
    rho   : scalar, persistence
    sigma : scalar, unconditional sd of s_t
    N     : int, number of states in discretized Markov process
    m     : scalar, discretized s goes from approx -m*sigma to m*sigma

    Returns
    ----------
    y  : array (N), states proportional to exp(s) s.t. E[y] = 1
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    """

    # make normalized grid, start with cross-sectional sd of 1
    s = np.linspace(-m, m, N)
    ds = s[1] - s[0]
    sd_innov = np.sqrt(1 - rho ** 2)

    # standard Tauchen method to generate Pi given N and m
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sd_innov) -
                    norm.cdf(s[j] - rho * s - ds / 2, scale=sd_innov))

    # invariant distribution and scaling
    pi = stationary(Pi)
    s *= (sigma / np.sqrt(variance(s, pi)))
    if normalize:
        y = np.exp(s) / np.sum(pi * np.exp(s))
    else:
        y = s

    return y, pi, Pi


def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""

    # Explicitly typecast N as an integer, since when the grid constructor functions
    # (e.g. the function that makes a_grid) are implemented as blocks, they interpret the integer-valued calibration
    # as a float.
    N = int(N)

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi
