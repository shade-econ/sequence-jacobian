import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.utilities.multidim import multiply_ith_dimension
from sequence_jacobian import het, simple, combine


def household_init(a_grid, y, r, sigma):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-sigma))
    return Va


def search_frictions(f, s):
    Pi_e = np.vstack(([1 - s, s], [f, 1 - f]))
    return Pi_e


def labor_income(z, w, b):
    y = np.vstack((w * z, b * w * z))
    return y 


@simple
def income_state_vars(rho_z, sd_z, nZ):
    z, _, Pi_z = sj.utilities.discretize.markov_rouwenhorst(rho=rho_z, sigma=sd_z, N=nZ)
    return z, Pi_z

@simple
def asset_state_vars(amin, amax, nA):
    a_grid = sj.utilities.discretize.agrid(amin=amin, amax=amax, n=nA)
    return a_grid


@het(exogenous=['Pi_e', 'Pi_z'], policy='a', backward='Va', backward_init=household_init)
def household_multidim(Va_p, a_grid, y, r, beta, sigma):
    c_nextgrid = (beta * Va_p) ** (-1 / sigma)
    coh = (1 + r) * a_grid + y[..., np.newaxis]
    a = sj.utilities.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # (x, xq, y)
    a = np.maximum(a, a_grid[0])
    c = coh - a
    uc = c ** (-sigma)
    Va = (1 + r) * uc

    return Va, a, c

@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household_onedim(Va_p, a_grid, y, r, beta, sigma):
    c_nextgrid = (beta * Va_p) ** (-1 / sigma)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.utilities.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # (x, xq, y)
    sj.utilities.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    uc = c ** (-sigma)
    Va = (1 + r) * uc

    return Va, a, c

def test_equivalence():
    calibration = dict(beta=0.95, r=0.01, sigma=2, a_grid = sj.utilities.discretize.agrid(1000, 50))

    e1, _, Pi1 = sj.utilities.discretize.markov_rouwenhorst(rho=0.7, sigma=0.7, N=3)
    e2, _, Pi2 = sj.utilities.discretize.markov_rouwenhorst(rho=0.3, sigma=0.5, N=3)
    e_multidim = np.outer(e1, e2)

    e_onedim = np.kron(e1, e2)
    Pi = np.kron(Pi1, Pi2)

    ss_multidim = household_multidim.steady_state({**calibration, 'y': e_multidim, 'Pi_e': Pi1, 'Pi_z': Pi2})
    ss_onedim = household_onedim.steady_state({**calibration, 'y': e_onedim, 'Pi': Pi})

    assert np.isclose(ss_multidim['A'], ss_onedim['A']) and np.isclose(ss_multidim['C'], ss_onedim['C'])

    D_onedim = ss_onedim.internal['household_onedim']['D']
    D_multidim = ss_multidim.internal['household_multidim']['D']

    assert np.allclose(D_onedim, D_multidim.reshape(*D_onedim.shape))

    J_multidim = household_multidim.jacobian(ss_multidim, inputs = ['r'], outputs=['A'], T=10)
    J_onedim = household_onedim.jacobian(ss_onedim, inputs = ['r'], outputs=['A'], T=10)

    assert np.allclose(J_multidim['A','r'], J_onedim['A','r'])


def test_pishock():
    calibration = dict(beta=0.95, r=0.01, sigma=2., f=0.4, s=0.1, w=1., b=0.5,
                       rho_z=0.9, sd_z=0.5, nZ=3, amin=0., amax=1000, nA=50) 
    
    household = household_multidim.add_hetinputs([search_frictions, labor_income])
    hh = combine([household, income_state_vars, asset_state_vars])

    ss = hh.steady_state(calibration)

    J = hh.jacobian(ss, inputs=['f', 's', 'r'], outputs=['C'], T=10)

    assert np.max(np.triu(J['C']['r'], 1)) <= 0  # low C before hike in r
    assert np.min(np.tril(J['C']['r'])) >= 0     # high C after hike in r

    assert np.all(J['C']['f'] > 0)  # high f increases C everywhere
    assert np.all(J['C']['s'] < 0)  # high s decreases C everywhere 

    return ss, J
