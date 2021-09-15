'''Standard Incomplete Market model'''

import numpy as np

from ...blocks.het_block import het
from ... import utilities as utils


def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


def income_state_vars(rho, sigma, nS):
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)
    return e_grid, Pi


def asset_state_vars(amax, nA):
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return a_grid


@het(exogenous='Pi', policy='a', backward='Va',
     hetinputs=[income_state_vars, asset_state_vars], backward_init=household_init)
def household(Va_p, a_grid, e_grid, r, w, beta, eis):
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = utils.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c
    