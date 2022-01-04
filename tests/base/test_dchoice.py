'''
SIM model with labor force participation choice
- state space: (s, x, e, a)
    - s is employment
        - 0: employed, 1: unemployed, 2: out of labor force
    - x is matching
        - 0: matched, 1: unmatched
    - e is labor productivity
    - a is assets
'''
import numpy as np
from numba import njit
import sequence_jacobian

from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker, LogitChoice
from sequence_jacobian import markov_rouwenhorst, agrid
from sequence_jacobian.classes.impulse_dict import ImpulseDict 
from sequence_jacobian.utilities.misc import nonconcave
from sequence_jacobian.utilities.interpolate import interpolate_coord_njit, apply_coord_njit, interpolate_point


'''Setup: utility function, hetinputs, initializer'''


@njit
def util(c, eis):
    if eis == 1:
        u = np.log(c)
    else:
        u = c ** (1 - 1 / eis) / (1 - 1 / eis)
    return u


def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, e_dist, Pi_e = markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return e_grid, e_dist, Pi_e, a_grid


def labor_income(a_grid, e_grid, atw, b, s, f, r):
    y = e_grid[np.newaxis, :] * np.array([atw, b, b])[:, np.newaxis]
    coh = (1 + r) * a_grid[np.newaxis, np.newaxis, :] + y[..., np.newaxis]
    Pi_s = np.array([[1 - s, s], [f, (1 - f)], [0, 1]])
    return y, coh, Pi_s


def backward_init(coh, a_grid, eis):
    V = util(0.1 * coh, eis) / 0.01
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    return V, Va 


'''Consumption-savings stage: : (s, e, a) -> (s, e, a')'''


def consav(V, Va, a_grid, coh, y, r, beta, eis):
    """DC-EGM algorithm"""
    # EGM step
    W = beta * V
    uc_endo= beta * Va
    c_endo= uc_endo** (-eis)
    a_endo= (c_endo+ a_grid[np.newaxis, np.newaxis, :] - y[:, :, np.newaxis]) / (1 + r)

    # upper envelope step
    V, c = upper_envelope(Va, W, a_endo, c_endo, coh, a_grid, eis)

    # update Va, report asset policy
    uc = c ** (-1 / eis)
    Va = (1 + r) * uc
    a = coh - c

    return V, Va, a, c


def upper_envelope(Va, W, a_endo, c_endo, coh, a_grid, *args):
    # identify bounds of nonconcave region
    ilower, iupper = nonconcave(Va)
    
    # upper envelope
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    a_endo = a_endo.reshape((-1, shape[-1]))
    c_endo = c_endo.reshape((-1, shape[-1]))
    coh = coh.reshape((-1, shape[-1]))
    ilower = ilower.reshape(-1)
    iupper = iupper.reshape(-1)
    V, c = upper_envelope_core(ilower, iupper, W, a_endo, c_endo, coh, a_grid, *args)

    return V.reshape(shape), c.reshape(shape)


@njit
def upper_envelope_core(ilower, iupper, W, a_endo, c_endo, coh, a_grid, *args):
    """Interpolate value function and consumption to exogenous grid."""
    nB, nA = W.shape
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    for ib in range(nB):
        ilower_cur = ilower[ib]
        iupper_cur = iupper[ib]

        # Below nonconcave region: exploit monotonicity
        if ilower_cur > 0:
            ai, api = interpolate_coord_njit(a_endo[ib, :ilower_cur], a_grid[:ilower_cur])
            c0 = apply_coord_njit(ai, api, c_endo[ib, :ilower_cur])
            W0 = apply_coord_njit(ai, api, W[ib, :ilower_cur])
            c[ib, :ilower_cur] = c0 
            V[ib, :ilower_cur] = util(c0, *args) + W0

        # Nonconcave region: check everything
        for ia in range(ilower_cur, iupper_cur):
            acur = a_grid[ia]
            for ja in range(nA - 1):
                ap_low = a_endo[ib, ja]
                ap_high = a_endo[ib, ja + 1]
                
                interp = (ap_low <= acur <= ap_high) or (ap_low >= acur >= ap_high)
                extrap = (ja == nA - 2) and (acur > a_endo[ib, nA - 1])

                if interp or extrap:
                    c0 = interpolate_point(acur, ap_low, ap_high, c_endo[ib, ja], c_endo[ib, ja+1])
                    W0 = interpolate_point(acur, ap_low, ap_high, W[ib, ja], W[ib, ja + 1])
                    V0 = util(c0, *args) + W0

                    if V0 > V[ib, ia]:
                        V[ib, ia] = V0
                        c[ib, ia] = c0

        # Above nonconcave region: exploit monotonicity
        if iupper_cur > 0:
            ai, api = interpolate_coord_njit(a_endo[ib, iupper_cur:], a_grid[iupper_cur:])
            c0 = apply_coord_njit(ai, api, c_endo[ib, iupper_cur:])
            W0 = apply_coord_njit(ai, api, W[ib, iupper_cur:])
            c[ib, iupper_cur:] = c0 
            V[ib, iupper_cur:] = util(c0, *args) + W0

        # Enforce borrowing constraint
        ia = 0
        while ia < nA and a_grid[ia] <= a_endo[ib, 0]:
            c[ib, ia] = coh[ib, ia]
            V[ib, ia] = util(c[ib, ia], *args) + W[ib, 0]
            ia += 1

    return V, c


'''Logit choice stage: (x, z, a) -> (s, z, a)'''


def participation(V, vphi, chi):
    '''adjustments to flow utility associated with x -> s choice, implements constraints on discrete choice'''
    flow_u = np.zeros((3, 2,) + V.shape[-2:])   # (s, x, z, a)
    flow_u[0, ...] = -vphi                      # employed
    flow_u[1, ...] = -chi                       # unemployed
    flow_u[0, 1, ...] = -np.inf                 # unmatched -> employed
    return flow_u


'''Put stages together'''

consav_stage = Continuous1D(backward=['Va', 'V'], policy='a', f=consav, name='consav')
labsup_stage = LogitChoice(value='V', backward='Va', index=0, 
                           taste_shock_scale='taste_shock',
                           f=participation, name='dchoice')
search_stage = ExogenousMaker(markov_name='Pi_s', index=0, name='search_shock')
prod_stage = ExogenousMaker(markov_name='Pi_e', index=1, name='prod_shock')

hh = StageBlock([prod_stage, search_stage, labsup_stage, consav_stage],
                backward_init=backward_init, hetinputs=[make_grids, labor_income], name='household')

def test_runs():
    calibration = {'taste_shock': 0.01, 'r': 0.005, 'beta': 0.97, 'eis': 0.5,
                   'vphi': 0.3, 'chi': 0.3, 'rho_e': 0.95, 'sd_e': 0.5, 'nE': 7, 'amin': .0, 'amax': 200.0, 'nA': 200, 'atw': 1.0, 'b': 0.5, 's': 0.1, 'f': 0.4}

    ss1 = hh.steady_state(calibration)
    ss2 = hh.steady_state({**calibration,
                           'V': 0.9*ss1.internals['household']['consav']['V'],
                           'Va': 0.9*ss1.internals['household']['consav']['Va']})
    
    # test steady-state equivalence (from different starting point)
    assert np.isclose(ss1['A'], ss2['A'])
    assert np.isclose(ss1['C'], ss2['C'])
    assert np.allclose(ss1.internals['household']['consav']['D'], ss2.internals['household']['consav']['D'])
    assert np.allclose(ss1.internals['household']['consav']['a'], ss2.internals['household']['consav']['a'])
    assert np.allclose(ss1.internals['household']['consav']['c'], ss2.internals['household']['consav']['c'])

    inputs = ['r', 'atw', 'f']
    outputs = ['A', 'C']
    T = 50
    J = hh.jacobian(ss1, inputs, outputs, T)

    # impulse responses
    shock = ImpulseDict({'f': 0.5 ** np.arange(50)})
    td_lin = hh.impulse_linear(ss1, shock, outputs=['C'])
    td_nonlin = hh.impulse_nonlinear(ss1, shock * 1E-4, outputs=['C'])
    td_ghost = hh.impulse_nonlinear(ss1, shock * 0.0, outputs=['C'])
    td_nonlin = td_nonlin - td_ghost
    assert np.allclose(td_lin['C'], td_nonlin['C'] / 1E-4, atol=1E-5)
