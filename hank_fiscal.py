import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.support.bijection import Bijection
from sequence_jacobian.steady_state.classes import SteadyStateDict


'''Part 1: Household block'''


def household_init(a_grid, y, rpost, sigma):
    c = np.maximum(1e-8, y[:, np.newaxis] + np.maximum(rpost, 0.04) * a_grid[np.newaxis, :])
    Va = (1 + rpost) * (c ** (-sigma))
    return Va


@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, Pi_p, a_grid, y, rpost, beta, sigma):
    """
    Backward step in simple incomplete market model. Assumes CRRA utility.

    Parameters
    ----------
    Va_p    : array (E, A), marginal value of assets tomorrow (backward iterator)
    Pi_p    : array (E, E), Markov matrix for skills tomorrow
    a_grid  : array (A), asset grid
    y       : array (E), non-financial income
    rpost   : scalar, ex-post return on assets
    beta    : scalar, discount factor
    sigma   : scalar, utility parameter

    Returns
    -------
    Va    : array (E, A), marginal value of assets today
    a     : array (E, A), asset policy today
    c     : array (E, A), consumption policy today
    """
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-1 / sigma)
    coh = (1 + rpost) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.utilities.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # (x, xq, y)
    sj.utilities.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    uc = c ** (-sigma)
    Va = (1 + rpost) * uc

    return Va, a, c


def get_mpcs(c, a, a_grid, rpost):
    """Approximate mpc, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpcs_ = np.empty_like(c)
    post_return = (1 + rpost) * a_grid

    # symmetric differences away from boundaries
    mpcs_[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])

    # asymmetric first differences at boundaries
    mpcs_[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpcs_[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])

    # special case of constrained
    mpcs_[a == a_grid[0]] = 1

    return mpcs_


def income(tau, Y, e_grid, e_dist, Gamma, transfer):
    """Labor income on the grid."""
    gamma = e_grid ** (Gamma * np.log(Y)) / np.vdot(e_dist, e_grid ** (1 + Gamma * np.log(Y)))
    y = (1 - tau) * Y * gamma * e_grid + transfer
    return y


@sj.simple
def income_state_vars(rho_e, sd_e, nE):
    e_grid, e_dist, Pi = sj.utilities.discretize.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    return e_grid, e_dist, Pi


@sj.simple
def asset_state_vars(amin, amax, nA):
    a_grid = sj.utilities.discretize.agrid(amin=amin, amax=amax, n=nA)
    return a_grid


@sj.hetoutput()
def mpcs(c, a, a_grid, rpost):
    """MPC out of lump-sum transfer."""
    mpc = get_mpcs(c, a, a_grid, rpost)
    return mpc


household.add_hetinput(income, verbose=False)
household.add_hetoutput(mpcs, verbose=False)


'''Part 2: rest of the model'''


@sj.simple
def interest_rates(r):
    rpost = r(-1)  # household ex-post return
    rb = r(-1)     # rate on 1-period real bonds
    return rpost, rb


@sj.solved(unknowns={'B': (0.0, 10.0)}, targets=['B_rule'], solver='brentq')
def fiscal(B, G, rb, rho_B, Y, transfer):
    B_rule = B.ss + rho_B * (B(-1) - B.ss) - B
    rev = (1 + rb) * B(-1) + G + transfer - B   # revenue to be raised
    tau = rev / Y
    return B_rule, rev, tau


@sj.simple
def fiscal_dis(B, G, rb, rho_B, Y, transfer):
    B_rule = B.ss + rho_B * (B(-1) - B.ss) - B
    rev = (1 + rb) * B(-1) + G + transfer - B   # revenue to be raised
    tau = rev / Y
    return B_rule, rev, tau


@sj.simple
def mkt_clearing(A, B, C, Y, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt


'''Try simple block'''


@sj.simple
def alma(a, b):
    c = a + b
    return c


alma2 = alma.remap({k: k + '_out' for k in ['a', 'b', 'c']})
cali2 = SteadyStateDict({'a_out': 1, 'b_out': 2}, internal={})
ss2 = alma2.steady_state(cali2)
jac2 = alma2.jacobian(cali2, ['a_out'])

mymap = Bijection({'a': 'A', 'c': 'C'})

cali = SteadyStateDict({'a': 1, 'b': 2}, internal={})
jac = alma.jacobian(cali, ['a'])


'''More serious'''

# household_remapped = household.remap({'beta': 'beta1', 'Mpc': 'Mpc1'})
# hh = sj.create_model([household_remapped, income_state_vars, asset_state_vars], name='Household')
# dag = sj.create_model([hh, interest_rates, fiscal_dis, mkt_clearing], name='HANK')
#
# calibration = {'Y': 1.0, 'r': 0.005, 'sigma': 2.0, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 11,
#                'amin': 0.0, 'amax': 1000, 'nA': 500, 'Gamma': 0.0, 'rho_B': 0.9, 'transfer': 0.143}
#
# ss = dag.solve_steady_state(calibration,
#                             unknowns={'beta1': .95, 'G': 0.2, 'B': 2.0},
#                             targets={'asset_mkt': 0.0, 'tau': 0.334, 'Mpc1': 0.25},
#                             solver='hybr')

# td_lin = dag.solve_impulse_linear(ss, {'r': 0.001*0.9**np.arange(300)},
#                                   unknowns=['B', 'Y'], targets=['asset_mkt', 'B_rule'])
#
#
# td_nonlin = dag.solve_impulse_nonlinear(ss, {'r': 0.001*0.9**np.arange(300)},
#                                         unknowns=['B', 'Y'], targets=['asset_mkt', 'B_rule'])



