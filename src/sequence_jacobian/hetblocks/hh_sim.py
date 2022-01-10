'''Standard Incomplete Market model'''

import numpy as np

from ..blocks.het_block import het
from .. import interpolate, misc, grids

'''Core HetBlock'''

def hh_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c
    

'''Extended HetBlock with grid and income process inputs added, and example calibration'''

def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, _, Pi = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid


def income(w, e_grid):
    y = w * e_grid
    return y


hh_extended = hh.add_hetinputs([income, make_grids])


def example_calibration():
    return dict(min_a=0, max_a=1000, rho_e=0.975, sd_e=0.7, n_a=200, n_e=7,
                w=1, r=0.01/4, beta=1-0.08/4, eis=1)
