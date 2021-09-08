import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.utilities.multidim import multiply_ith_dimension
from sequence_jacobian import het


def household_init(a_grid, y, r, sigma):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-sigma))
    return Va


@het(exogenous=['Pi1', 'Pi2'], policy='a', backward='Va', backward_init=household_init)
def household_multidim(Va_p, Pi1_p, Pi2_p, a_grid, y, r, beta, sigma):
    Va_p = multiply_ith_dimension(beta * Pi1_p, 0, Va_p)
    uc_nextgrid = multiply_ith_dimension(Pi2_p, 1, Va_p)

    c_nextgrid = uc_nextgrid ** (-1 / sigma)
    coh = (1 + r) * a_grid + y[..., np.newaxis]
    a = sj.utilities.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # (x, xq, y)
    a = np.maximum(a, a_grid[0])
    c = coh - a
    uc = c ** (-sigma)
    Va = (1 + r) * uc

    return Va, a, c


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household_onedim(Va_p, Pi_p, a_grid, y, r, beta, sigma):
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-1 / sigma)
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

    ss_multidim = household_multidim.steady_state({**calibration, 'y': e_multidim, 'Pi1': Pi1, 'Pi2': Pi2})
    ss_onedim = household_onedim.steady_state({**calibration, 'y': e_onedim, 'Pi': Pi})

    assert np.isclose(ss_multidim['A'], ss_onedim['A']) and np.isclose(ss_multidim['C'], ss_onedim['C'])

    D_onedim = ss_onedim.internal['household_onedim']['D']
    D_multidim = ss_multidim.internal['household_multidim']['D']

    assert np.allclose(D_onedim, D_multidim.reshape(*D_onedim.shape))
