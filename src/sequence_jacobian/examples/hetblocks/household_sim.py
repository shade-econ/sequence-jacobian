'''Standard Incomplete Market model'''

import numpy as np

from ...blocks.het_block import het
from ... import utilities as utils


def household_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = utils.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c
    