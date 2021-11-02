import numpy as np

from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.examples.hetblocks.household_sim import household, household_init
from sequence_jacobian import markov_rouwenhorst, agrid 
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker
from sequence_jacobian import utilities as utils
from sequence_jacobian.classes import ImpulseDict

def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, e_dist, Pi_ss = markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return e_grid, e_dist, Pi_ss, a_grid

def alter_Pi(Pi_ss, shift):
    Pi = Pi_ss.copy()
    Pi[:, 0] -= shift
    Pi[:, -1] += shift
    return Pi

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

het_stage = Continuous1D(backward='Va', policy='a', f=household_new, name='stage1')
hh2 = StageBlock([ExogenousMaker('Pi', 0, 'stage0'), het_stage], name='household',
                    backward_init=household_init, hetinputs=(make_grids, income, alter_Pi))

def test_equivalence():
    hh1 = household.add_hetinputs([make_grids, income, alter_Pi])
    calibration = {'r': 0.004, 'eis': 0.5, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3,
                   'amin': 0.0, 'amax': 200, 'nA': 100, 'transfer': 0.143, 'N': 1,
                   'atw': 1, 'beta': 0.97, 'shift': 0}
    ss1 = hh1.steady_state(calibration)
    ss2 = hh2.steady_state(calibration)

    # test steady-state equivalence
    assert np.isclose(ss1['A'], ss2['A'])
    assert np.isclose(ss1['C'], ss2['C'])
    assert np.allclose(ss1.internals['household']['Dbeg'], ss2.internals['household']['stage0']['D'])
    assert np.allclose(ss1.internals['household']['a'], ss2.internals['household']['stage1']['a'])
    assert np.allclose(ss1.internals['household']['c'], ss2.internals['household']['stage1']['c'])

    # find Jacobians...
    inputs = ['r', 'atw', 'shift']
    outputs = ['A', 'C']
    T = 200
    J1 = hh1.jacobian(ss1, inputs, outputs, T)
    J2 = hh2.jacobian(ss2, inputs, outputs, T)

    # test Jacobian equivalence
    for i in inputs:
        for o in outputs:
            assert np.allclose(J1[o, i], J2[o, i])

    # impulse linear
    shock = ImpulseDict({'r': 0.5 ** np.arange(20)})
    td_lin1 = hh1.impulse_linear(ss1, shock, outputs=['C'])
    td_lin2 = hh2.impulse_linear(ss2, shock, outputs=['C'])
    assert np.allclose(td_lin1['C'], td_lin2['C'])

    # impulse nonlinear
    td_nonlin1 = hh1.impulse_nonlinear(ss1, shock * 1E-4, outputs=['C'])
    td_nonlin2 = hh2.impulse_nonlinear(ss2, shock * 1E-4, outputs=['C'])
    assert np.allclose(td_nonlin1['C'], td_nonlin2['C'])
