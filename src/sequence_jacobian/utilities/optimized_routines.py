"""Njitted routines to speed up some steps in backward iteration or aggregation"""

import numpy as np
from numba import njit


@njit
def setmin(x, xmin):
    """Set 2-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                x[i, j] = xmin
            else:
                break


@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


@njit
def fast_aggregate(X, Y):
    """If X has dims (T, ...) and Y has dims (T, ...), do dot product for each T to get length-T vector.

    Identical to np.sum(X*Y, axis=(1,...,X.ndim-1)) but avoids costly creation of intermediates,
    useful for speeding up aggregation in td by factor of 4 to 5."""
    T = X.shape[0]
    Xnew = X.reshape(T, -1)
    Ynew = Y.reshape(T, -1)
    Z = np.empty(T)
    for t in range(T):
        Z[t] = Xnew[t, :] @ Ynew[t, :]
    return Z
