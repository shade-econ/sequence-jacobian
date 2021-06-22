"""Simple model to test remapping & multiple hetblocks."""

import numpy as np
import copy

from sequence_jacobian import utilities as utils
from sequence_jacobian import create_model, hetoutput, het, simple


'''Part 1: HA block'''


def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, Pi_p, a_grid, e_grid, beta, r, w, eis):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Parameters
    ----------
    Va_p       : array (S, A), marginal value of assets tomorrow
    Pi_p       : array (S, S), Markov matrix for skills tomorrow
    a_grid     : array (A), asset grid
    e_grid     : array (S*B), skill grid
    beta       : scalar, discount rate today
    r          : scalar, ex-post real interest rate
    w          : scalar, wage
    eis        : scalar, elasticity of intertemporal substitution

    Returns
    ----------
    Va : array (S*B, A), marginal value of assets today
    a  : array (S*B, A), asset policy today
    c  : array (S*B, A), consumption policy today
    """
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = utils.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def get_mpcs(c, a, a_grid, rpost):
    """Approximate mpc, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpcs = np.empty_like(c)
    post_return = (1 + rpost) * a_grid

    # symmetric differences away from boundaries
    mpcs[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])

    # asymmetric first differences at boundaries
    mpcs[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpcs[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])

    # special case of constrained
    mpcs[a == a_grid[0]] = 1

    return mpcs


'''Part 2: Simple Blocks'''


@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt


@simple
def income_state_vars(rho, sigma, nS):
    e, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)
    return e, Pi


@simple
def asset_state_vars(amax, nA):
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return a_grid


@simple
def firm_steady_state_solution(r, delta, alpha):
    rk = r + delta
    Z = (rk / alpha) ** alpha  # normalize so that Y=1
    K = (alpha * Z / rk) ** (1 / (1 - alpha))
    Y = Z * K ** alpha
    w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))
    return Z, K, Y, w


'''Part 3: permanent heterogeneity'''

# remap method takes a dict and returns new copies of blocks
to_map = ['beta', *household.outputs]
hh_patient = household.remap({k: k + '_patient' for k in to_map}).rename('patient household')
hh_impatient = household.remap({k: k + '_impatient' for k in to_map}).rename('impatient household')


@simple
def aggregate(A_patient, A_impatient, C_patient, C_impatient, mass_patient):
    C = mass_patient * C_patient + (1 - mass_patient) * C_impatient
    A = mass_patient * A_patient + (1 - mass_patient) * A_impatient
    return C, A


'''Steady state'''

# DAG
# blocks = [hh_patient, hh_impatient, firm, mkt_clearing, income_state_vars, asset_state_vars, aggregate]
# helper_blocks = [firm_steady_state_solution]
# ks_model = create_model(blocks, name="Krusell-Smith")

# # Steady State
# calibration = {"eis": 1, "delta": 0.025, "alpha": 0.3, "rho": 0.966, "sigma": 0.5, "L": 1.0,
#                "nS": 11, "nA": 100, "amax": 1000, "r": 0.01, 'beta_impatient': 0.98}
# unknowns_ss = {"beta_patient": (0.98/1.01, 0.999/1.01), "Z": 0.85, "K": 3.}
# targets_ss = {"asset_mkt": 0., "Y": 1., "r": 0.01}
# ss = ks_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="brentq",
#                                  helper_blocks=helper_blocks, helper_targets=["Y", "r"])
