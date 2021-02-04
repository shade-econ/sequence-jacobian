"""Various lower-level functions to support the computation of Jacobians"""

import numpy as np
from numba import njit

from .. import asymptotic


# For supporting SimpleSparse
def multiply_basis(t1, t2):
    """Matrix multiplication operation mapping two sparse basis elements to another."""
    # equivalent to formula in Proposition 2 of Sequence Space Jacobian paper, but with
    # signs of i and j flipped to reflect different sign convention used here
    i, m = t1
    j, n = t2
    k = i + j
    if i >= 0:
        if j >= 0:
            l = max(m, n - i)
        elif k >= 0:
            l = max(m, n - k)
        else:
            l = max(m + k, n)
    else:
        if j <= 0:
            l = max(m + j, n)
        else:
            l = max(m, n) + min(-i, j)
    return k, l


def multiply_rs_rs(s1, s2):
    """Matrix multiplication operation on two SimpleSparse objects."""
    # iterate over all pairs (i, m) -> x and (j, n) -> y in objects,
    # add all pairwise products to get overall product
    elements = {}
    for im, x in s1.elements.items():
        for jn, y in s2.elements.items():
            kl = multiply_basis(im, jn)
            if kl in elements:
                elements[kl] += x * y
            else:
                elements[kl] = x * y
    return elements


@njit
def multiply_rs_matrix(indices, xs, A):
    """Matrix multiplication of SimpleSparse object ('indices' and 'xs') and matrix A.
    Much more computationally demanding than multiplying two SimpleSparse (which is almost
    free with simple analytical formula), so we implement as jitted function."""
    n = indices.shape[0]
    T = A.shape[0]
    S = A.shape[1]
    Aout = np.zeros((T, S))

    for count in range(n):
        # for Numba to jit easily, SimpleSparse with basis elements '(i, m)' with coefs 'x'
        # was stored in 'indices' and 'xs'
        i = indices[count, 0]
        m = indices[count, 1]
        x = xs[count]

        # loop faster than vectorized when jitted
        # directly use def of basis element (i, m), displacement of i and ignore first m
        if i == 0:
            for t in range(m, T):
                for s in range(S):
                    Aout[t, s] += x * A[t, s]
        elif i > 0:
            for t in range(m, T - i):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
        else:
            for t in range(m - i, T):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
    return Aout


def pack_asymptotic_jacobians(jacdict, inputs, outputs, tau):
    """If we have -(tau-1),...,(tau-1) AsymptoticTimeInvariant Jacobians (or SimpleSparse) from
    nI inputs to nO outputs in jacdict, combine into (2*tau-1,nO,nI) array A"""
    nI, nO = len(inputs), len(outputs)
    A = np.empty((2*tau-1, nI, nO))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            if inputs[iI] in subdict:
                A[:, iO, iI] = make_ATI_v(jacdict[outputs[iO]][inputs[iI]], tau)
            else:
                A[:, iO, iI] = 0
    return A


def unpack_asymptotic_jacobians(A, inputs, outputs, tau):
    """If we have (2*tau-1, nO, nI) array A where each A[:,o,i] is vector for AsymptoticTimeInvariant
    Jacobian mapping output o to output i, output nested dict of AsymptoticTimeInvariant objects"""
    nI, nO = len(inputs), len(outputs)

    jacdict = {}
    for iO in range(nO):
        jacdict[outputs[iO]] = {}
        for iI in range(nI):
            jacdict[outputs[iO]][inputs[iI]] = asymptotic.AsymptoticTimeInvariant(A[:, iO, iI])
    return jacdict


def pack_vectors(vs, names, T):
    v = np.zeros(len(names)*T)
    for i, name in enumerate(names):
        if name in vs:
            v[i*T:(i+1)*T] = vs[name]
    return v


def unpack_vectors(v, names, T):
    vs = {}
    for i, name in enumerate(names):
        vs[name] = v[i*T:(i+1)*T]
    return vs


def make_matrix(A, T):
    """If A is not an outright ndarray, e.g. it is SimpleSparse, call its .matrix(T) method
    to convert it to T*T array."""
    if not isinstance(A, np.ndarray):
        return A.matrix(T)
    else:
        return A


def make_ATI_v(x, tau):
    """If x is either a AsymptoticTimeInvariant or something that can be converted to it, e.g.
    SimpleSparse, report the underlying length 2*tau-1 vector with entries -(tau-1),...,(tau-1)"""
    if not isinstance(x, asymptotic.AsymptoticTimeInvariant):
        return x.asymptotic_time_invariant.changetau(tau).v
    else:
        return x.v
