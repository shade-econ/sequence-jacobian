import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import het, simple, hetoutput, combine, solved, create_model, get_H_U
from sequence_jacobian.jacobian.classes import JacobianDict, ZeroMatrix
from sequence_jacobian.blocks.support.impulse import ImpulseDict


'''Part 1: Household block'''


def household_init(a_grid, y, rpost, sigma):
    c = np.maximum(1e-8, y[:, np.newaxis] +
                   np.maximum(rpost, 0.04) * a_grid[np.newaxis, :])
    Va = (1 + rpost) * (c ** (-sigma))
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
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
    mpcs_[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / \
        (post_return[2:] - post_return[:-2])

    # asymmetric first differences at boundaries
    mpcs_[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpcs_[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])

    # special case of constrained
    mpcs_[a == a_grid[0]] = 1

    return mpcs_


def income(tau, Y, e_grid, e_dist, Gamma, transfer):
    """Labor income on the grid."""
    gamma = e_grid ** (Gamma * np.log(Y)) / np.vdot(e_dist,
                                                    e_grid ** (1 + Gamma * np.log(Y)))
    y = (1 - tau) * Y * gamma * e_grid + transfer
    return y


@simple
def income_state_vars(rho_e, sd_e, nE):
    e_grid, e_dist, Pi = sj.utilities.discretize.markov_rouwenhorst(
        rho=rho_e, sigma=sd_e, N=nE)
    return e_grid, e_dist, Pi


@simple
def asset_state_vars(amin, amax, nA):
    a_grid = sj.utilities.discretize.agrid(amin=amin, amax=amax, n=nA)
    return a_grid


@hetoutput()
def mpcs(c, a, a_grid, rpost):
    """MPC out of lump-sum transfer."""
    mpc = get_mpcs(c, a, a_grid, rpost)
    return mpc


household.add_hetinput(income, verbose=False)
household.add_hetoutput(mpcs, verbose=False)


'''Part 2: rest of the model'''


@simple
def interest_rates(r):
    rpost = r(-1)  # household ex-post return
    rb = r(-1)     # rate on 1-period real bonds
    return rpost, rb


# @simple
# def fiscal(B, G, rb, Y, transfer):
#     rev = rb * B + G + transfer   # revenue to be raised
#     tau = rev / Y
#     return rev, tau

@simple
def fiscal(B, G, rb, Y, transfer, rho_B):
    B_rule = B.ss + rho_B * (B(-1) - B.ss + G - G.ss) - B
    rev = (1 + rb) * B(-1) + G + transfer - B   # revenue to be raised
    tau = rev / Y
    return B_rule, rev, tau


@simple
def mkt_clearing(A, B, C, Y, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt


'''try this'''

# flat dag
# dag = sj.create_model([household, income_state_vars, asset_state_vars, interest_rates, fiscal, mkt_clearing],
#                       name='HANK')


# nested dag
hh = combine([household, income_state_vars, asset_state_vars], name='HH')
dag = sj.create_model([hh, interest_rates, fiscal, mkt_clearing], name='HANK')

calibration = {'Y': 1.0, 'r': 0.005, 'sigma': 2.0, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3,
               'amin': 0.0, 'amax': 1000, 'nA': 100, 'Gamma': 0.0, 'transfer': 0.143}
calibration['rho_B'] = 0.8

# ss0 = dag.solve_steady_state(calibration, solver='hybr',
#                             unknowns={'beta': .95, 'G': 0.2, 'B': 2.0},
#                             targets={'asset_mkt': 0.0, 'tau': 0.334, 'Mpc': 0.25})

#Js = dag.partial_jacobians(ss0, inputs=['Y', 'r'], T=10)

@solved(unknowns={'B': (0.0, 10.0)}, targets=['B_rule'], solver='brentq')
def fiscal_solved(B, G, rb, Y, transfer, rho_B):
    B_rule = B.ss + rho_B * (B(-1) - B.ss + G - G.ss) - B
    rev = (1 + rb) * B(-1) + G + transfer - B   # revenue to be raised
    tau = rev / Y
    return B_rule, rev, tau

dag = sj.create_model([hh, interest_rates, fiscal_solved, mkt_clearing], name='HANK')

ss = dag.solve_steady_state(calibration, dissolve=['fiscal_solved'], solver='hybr',
                            unknowns={'beta': .95, 'G': 0.2, 'B': 2.0},
                            targets={'asset_mkt': 0.0, 'tau': 0.334, 'Mpc': 0.25})

# assert all(np.allclose(ss0[k], ss[k]) for k in ss0)

# Partial Jacobians
# J_ir = interest_rates.jacobian(ss, ['r', 'tau'])
# J_ha = household.jacobian(ss, ['rpost', 'tau'], T=5)
# J1 = J_ha.compose(J_ir)

# Let's make a simple combined block
# hh = sj.combine([interest_rates, household], name='HH')
# J2 = hh.jacobian(ss, exogenous=['r', 'tau'], T=4)


'''Test H_U'''

unknowns = ['Y']
targets = ['asset_mkt']
exogenous = ['r']
T = 5

# J = dag.partial_jacobians(ss, inputs=unknowns + exogenous, T=5)

# HZ1 = dag.jacobian(ss, exogenous=exogenous, outputs=targets, T=5)
# HZ2 = get_H_U(dag.blocks, exogenous, targets, T, ss=ss)
# np.all(HZ1['asset_mkt']['r'] == HZ2)
#
# HU1 = dag.jacobian(ss, exogenous=unknowns, outputs=targets, T=5)
# HU2 = get_H_U(dag.blocks, unknowns, targets, T, ss=ss)
# np.all(HU1 == HU2)

# J_int = interest_rates.jacobian(ss, exogenous=['r'])
# J_hh = household.jacobian(ss, exogenous=['Y', 'rpost'], T=4)
#
# # This should have
# J_all1 = J_hh.compose(J_int)
# J_all2 = J_int.compose(J_hh)


# G = dag.solve_jacobian(ss0, inputs=['r'], outputs=['A', 'Y', 'asset_mkt', 'goods_mkt'], unknowns=['Y', 'B'], targets=['asset_mkt', 'B_rule'], T=300)

G = dag.solve_jacobian(ss, inputs=['r'], outputs=['A', 'Y', 'asset_mkt', 'goods_mkt'], unknowns=['Y'], targets=['asset_mkt'], T=300)

shock = ImpulseDict({'r': 0.001*0.9**np.arange(300)})

td1 = G @ shock 


td_lin = dag.solve_impulse_linear(ss, unknowns=['Y'], targets=['asset_mkt'],
                                  inputs=shock, outputs=['A', 'Y', 'asset_mkt', 'goods_mkt'])

# td_nonlin = dag.solve_impulse_nonlinear(ss, {'r': 0.001*0.9**np.arange(300)},
#                                         unknowns=['B', 'Y'], targets=['asset_mkt', 'B_rule'])
