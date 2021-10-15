import numpy as np
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.examples.hetblocks.household_sim import household
from sequence_jacobian import markov_rouwenhorst, agrid 
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker
from sequence_jacobian import utilities as utils

def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, e_dist, Pi = markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return e_grid, e_dist, Pi, a_grid

def income(atw, N, e_grid, transfer):
    y = atw * N * e_grid + transfer
    return y

# copy original household hetblock but get rid of _p on Va
def household_new(Va, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = utils.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c

het_stage = Continuous1D(backward='Va', policy='a', f=household_new, name='consumption_savings')
stage_block = StageBlock([ExogenousMaker('Pi', 0, 'income_shock'), het_stage], name='household')

def test_ss_equivalence():
    hh = household.add_hetinputs([make_grids, income])
    calibration = {'r': 0.005, 'eis': 0.5, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3, 'amin': 0.0, 'amax': 200,
                    'nA': 100, 'transfer': 0.143, 'N': 1, 'atw': 1, 'beta': 0.97}
    ss1 = hh.steady_state(calibration)
    ss2 = stage_block._steady_state(ss1)

    assert np.isclose(ss1['A'], ss2['A'])
    assert np.isclose(ss1['C'], ss2['C'])
    assert np.allclose(ss1.internals['household']['Dbeg'], ss2.internals['household']['income_shock']['D'])
    assert np.allclose(ss1.internals['household']['a'], ss2.internals['household']['consumption_savings']['a'])
    assert np.allclose(ss1.internals['household']['c'], ss2.internals['household']['consumption_savings']['c'])

