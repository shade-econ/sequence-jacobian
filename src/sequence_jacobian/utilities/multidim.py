import numpy as np


def multiply_ith_dimension(Pi, i, X):
    """If Pi is a square matrix, multiply Pi times the ith dimension of X and return"""
    X = X.swapaxes(0, i)
    shape = X.shape
    X = X.reshape((shape[0], -1))

    # iterate forward using Pi
    X = Pi @ X

    # reverse steps
    X = X.reshape(shape)
    return X.swapaxes(0, i)


def outer(pis):
    """Return n-dimensional outer product of list of n vectors"""
    pi = pis[0]
    for pi_i in pis[1:]:
        pi = np.kron(pi, pi_i)
    return pi.reshape(*(len(pi_i) for pi_i in pis))


def batch_multiply_ith_dimension(P, i, X):
    """If P is (D, X.shape) array, multiply P and X along ith dimension of X."""
    # standardize arrays
    P = P.swapaxes(1, 1+i)
    X = X.swapaxes(0, i)
    Pshape = P.shape
    P = P.reshape((*Pshape[:2], -1))
    X = X.reshape((X.shape[0], -1))

    # P[i, j, ...] @ X[j, ...]
    X = np.einsum('ijb,jb->ib', P, X)

    # original shape and order
    X = X.reshape(Pshape[0], *Pshape[-2:])
    return X.swapaxes(0, i)


def logsum():
