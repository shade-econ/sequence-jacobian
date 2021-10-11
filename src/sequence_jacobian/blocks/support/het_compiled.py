import numpy as np
from numba import njit

@njit
def forward_policy_1d(D, x_i, x_pi):
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            d = D[iz, ix]

            Dnew[iz, i] += d * pi
            Dnew[iz, i+1] += d * (1 - pi)
    
    return Dnew


@njit
def expectation_policy_1d(X, x_i, x_pi):
    nZ, nX = X.shape
    Xnew = np.zeros_like(X)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            Xnew[iz, ix] = pi * X[iz, i] + (1-pi) * X[iz, i+1]
    return Xnew


@njit
def forward_policy_shock_1d(Dss, x_i_ss, x_pi_shock):
    """forward_step_1d linearized wrt x_pi"""
    nZ, nX = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i_ss[iz, ix]
            dshock = x_pi_shock[iz, ix] * Dss[iz, ix]
            Dshock[iz, i] += dshock
            Dshock[iz, i + 1] -= dshock

    return Dshock


@njit
def forward_policy_2d(D, x_i, y_i, x_pi, y_pi):
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
def expectation_policy_2d(X, x_i, y_i, x_pi, y_pi):
    nZ, nX, nY = X.shape
    Xnew = np.empty_like(X)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i[iz, ix, iy]
                iyp = y_i[iz, ix, iy]
                alpha = x_pi[iz, ix, iy]
                beta = y_pi[iz, ix, iy]

                Xnew[iz, ix, iy] = (alpha * beta * X[iz, ixp, iyp] + alpha * (1-beta) * X[iz, ixp, iyp+1] +
                                    (1-alpha) * beta * X[iz, ixp+1, iyp] +
                                    (1-alpha) * (1-beta) * X[iz, ixp+1, iyp+1])
    return Xnew


@njit
def forward_policy_shock_2d(Dss, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock):
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


def logsum(V, scale):
    """Logsum formula along 0th dimension"""
    const = V[0, ...]
    Vnorm = V - const
    EV = const + scale * np.log(np.exp(Vnorm / scale).sum(axis=0))
    return EV 