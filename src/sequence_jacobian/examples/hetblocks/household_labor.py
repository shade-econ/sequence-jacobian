'''Standard Incomplete Market model with Endogenous Labor Supply'''

import numpy as np
from numba import vectorize, njit

from ...blocks.het_block import het
from ... import utilities as utils


def household_init(a_grid, e_grid, r, w, eis, T):
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return fininc, Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi):
    '''Single backward step via EGM.'''
    # TODO: ws shoukld be hetinput
    ws = w * e_grid
    uc_nextgrid = beta * Va_p
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, ws[:, np.newaxis], eis, frisch, vphi)

    lhs = c_nextgrid - ws[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
    rhs = (1 + r) * a_grid
    c = utils.interpolate.interpolate_y(lhs, rhs, c_nextgrid)
    n = utils.interpolate.interpolate_y(lhs, rhs, n_nextgrid)

    a = rhs + ws[:, np.newaxis] * n + T[:, np.newaxis] - c
    iconst = np.nonzero(a < a_grid[0])
    a[iconst] = a_grid[0]

    if iconst[0].size != 0 and iconst[1].size != 0:
        c[iconst], n[iconst] = solve_cn(ws[iconst[0]],
                                        rhs[iconst[1]] + T[iconst[0]] - a_grid[0],
                                        eis, frisch, vphi, Va_p[iconst])

    Va = (1 + r) * c ** (-1 / eis)
    
    # TODO: make hetoutput
    n_e = e_grid[:, np.newaxis] * n

    return Va, a, c, n, n_e


'''Supporting functions for HA block'''

@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


@vectorize
def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc = np.log(uc_seed)
    for i in range(30):
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        if abs(ne) < 1E-11:
            break
        else:
            log_uc -= ne / ne_p
    else:
        raise ValueError("Cannot solve constrained household's problem: No convergence after 30 iterations!")

    return np.exp(log_uc)


@njit
def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc
