"""Forward iteration of distribution on grid and related functions.

        - forward_step_1d
        - forward_step_2d
            - apply law of motion for distribution to go from D_{t-1} to D_t

        - forward_step_shock_1d
        - forward_step_shock_2d
            - forward_step linearized, used in part 1 of fake news algorithm to get curlyDs

        - forward_step_transpose_1d
        - forward_step_transpose_2d
            - transpose of forward_step, used in part 2 of fake news algorithm to get curlyPs
"""

import numpy as np
from numba import njit


@njit
def forward_step_1d(D, Pi_T, x_i, x_pi):
    """Single forward step to update distribution using exogenous Markov transition Pi and
    policy x_i and x_pi for one-dimensional endogenous state.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of the endogenous
    part of Lam_{t-1}'.

    Note that it takes Pi_T, the transpose of Pi, as input rather than transposing itself;
    this is so that when it is applied repeatedly, we can precalculate a transpose stored in
    correct order rather than a view.

    Parameters
    ----------
    D : array (S*X), beginning-of-period distribution over s_t, x_(t-1)
    Pi_T : array (S*S), transpose Markov matrix that maps s_t to s_(t+1)
    x_i : int array (S*X), left gridpoint of endogenous policy
    x_pi : array (S*X), weight on left gridpoint of endogenous policy

    Returns
    ----------
    Dnew : array (S*X), beginning-of-next-period dist s_(t+1), x_t
    """

    # first update using endogenous policy
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            d = D[iz, ix]
            Dnew[iz, i] += d * pi
            Dnew[iz, i+1] += d * (1 - pi)

    # then using exogenous transition matrix
    return Pi_T @ Dnew


def forward_step_2d(D, Pi_T, x_i, y_i, x_pi, y_pi):
    """Like forward_step_1d but with two-dimensional endogenous state, policies given by x and y"""
    Dmid = forward_step_endo_2d(D, x_i, y_i, x_pi, y_pi)
    nZ, nX, nY = Dmid.shape
    return (Pi_T @ Dmid.reshape(nZ, -1)).reshape(nZ, nX, nY)


@njit
def forward_step_endo_2d(D, x_i, y_i, x_pi, y_pi):
    """Endogenous update part of forward_step_2d"""
    nZ, nX, nY = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i[iz, ix, iy]
                iyp = y_i[iz, ix, iy]
                beta = x_pi[iz, ix, iy]
                alpha = y_pi[iz, ix, iy]

                Dnew[iz, ixp, iyp] += alpha * beta * D[iz, ix, iy]
                Dnew[iz, ixp+1, iyp] += alpha * (1 - beta) * D[iz, ix, iy]
                Dnew[iz, ixp, iyp+1] += (1 - alpha) * beta * D[iz, ix, iy]
                Dnew[iz, ixp+1, iyp+1] += (1 - alpha) * (1 - beta) * D[iz, ix, iy]
    return Dnew


@njit
def forward_step_shock_1d(Dss, Pi_T, x_i_ss, x_pi_shock):
    """forward_step_1d linearized wrt x_pi"""
    # first find effect of shock to endogenous policy
    nZ, nX = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i_ss[iz, ix]
            dshock = x_pi_shock[iz, ix] * Dss[iz, ix]
            Dshock[iz, i] += dshock
            Dshock[iz, i + 1] -= dshock

    # then apply exogenous transition matrix to update
    return Pi_T @ Dshock


def forward_step_shock_2d(Dss, Pi_T, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock):
    """forward_step_2d linearized wrt x_pi and y_pi"""
    Dmid = forward_step_shock_endo_2d(Dss, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock)
    nZ, nX, nY = Dmid.shape
    return (Pi_T @ Dmid.reshape(nZ, -1)).reshape(nZ, nX, nY)


@njit
def forward_step_shock_endo_2d(Dss, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock):
    """Endogenous update part of forward_step_shock_2d"""
    nZ, nX, nY = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i_ss[iz, ix, iy]
                iyp = y_i_ss[iz, ix, iy]
                alpha = x_pi_ss[iz, ix, iy]
                beta = y_pi_ss[iz, ix, iy]

                dalpha = x_pi_shock[iz, ix, iy] * Dss[iz, ix, iy]
                dbeta = y_pi_shock[iz, ix, iy] * Dss[iz, ix, iy]

                Dshock[iz, ixp, iyp] += dalpha * beta + alpha * dbeta
                Dshock[iz, ixp+1, iyp] += dbeta * (1-alpha) - beta * dalpha
                Dshock[iz, ixp, iyp+1] += dalpha * (1-beta) - alpha * dbeta
                Dshock[iz, ixp+1, iyp+1] -= dalpha * (1-beta) + dbeta * (1-alpha)
    return Dshock


@njit
def forward_step_transpose_1d(D, Pi, x_i, x_pi):
    """Transpose of forward_step_1d"""
    # first update using exogenous transition matrix
    D = Pi @ D

    # then update using (transpose) endogenous policy
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            Dnew[iz, ix] = pi * D[iz, i] + (1-pi) * D[iz, i+1]
    return Dnew


def forward_step_transpose_2d(D, Pi, x_i, y_i, x_pi, y_pi):
    """Transpose of forward_step_2d."""
    nZ, nX, nY = D.shape
    Dmid = (Pi @ D.reshape(nZ, -1)).reshape(nZ, nX, nY)
    return forward_step_transpose_endo_2d(Dmid, x_i, y_i, x_pi, y_pi)


@njit
def forward_step_transpose_endo_2d(D, x_i, y_i, x_pi, y_pi):
    """Endogenous update part of forward_step_transpose_2d"""
    nZ, nX, nY = D.shape
    Dnew = np.empty_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i[iz, ix, iy]
                iyp = y_i[iz, ix, iy]
                alpha = x_pi[iz, ix, iy]
                beta = y_pi[iz, ix, iy]

                Dnew[iz, ix, iy] = (alpha * beta * D[iz, ixp, iyp] + alpha * (1-beta) * D[iz, ixp, iyp+1] +
                                    (1-alpha) * beta * D[iz, ixp+1, iyp] +
                                    (1-alpha) * (1-beta) * D[iz, ixp+1, iyp+1])
    return Dnew
