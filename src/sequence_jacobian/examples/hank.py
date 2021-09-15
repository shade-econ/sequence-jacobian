import numpy as np

from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.combined_block import create_model, combine
from .hetblocks import household_labor as hh


'''Part 1: Blocks'''


@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def mkt_clearing(A, N_E, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = N_E - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


#TODO: this works but seems wrong conceptually
@simple
def partial_ss_solution(B_Y, Y, mu, r, kappa, Z, pi):
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B

    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) - (1 + pi).apply(np.log)

    return B, w, Div, Tax, nkpc_res


'''Part 2: Embed HA block'''


# This cannot be a hetinput, bc `transfers` depends on it
@simple
def make_grids(rho_s, sigma_s, nS, amax, nA):
    e_grid, pi_e, Pi = utils.discretize.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div, Tax, e_grid):
    # default incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid

    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


'''Part 3: DAG'''

def dag():
    # Combine blocks
    household = hh.household.add_hetinputs([transfers])
    household = combine([make_grids, household], name='HH')
    blocks = [household, firm, monetary, fiscal, mkt_clearing, nkpc]
    helper_blocks = [partial_ss_solution]
    hank_model = create_model(blocks, name="One-Asset HANK")

    # Steady state
    calibration = {'r': 0.005, 'rstar': 0.005, 'eis': 0.5, 'frisch': 0.5, 'B_Y': 5.6,
                   'mu': 1.2, 'rho_s': 0.966, 'sigma_s': 0.5, 'kappa': 0.1, 'phi': 1.5,
                   'Y': 1, 'Z': 1, 'L': 1, 'pi': 0, 'nS': 2, 'amax': 150, 'nA': 10}
    unknowns_ss = {'beta': 0.986, 'vphi': 0.8, 'w': 0.8}
    targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'nkpc_res': 0.}
    ss = hank_model.solve_steady_state(calibration, unknowns_ss, targets_ss, 
                                       solver='broyden_custom',
                                       helper_blocks=helper_blocks,
                                       helper_targets=['nkpc_res'])

    # Transitional dynamics
    unknowns = ['pi', 'w', 'Y']
    targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']
    exogenous = ['rstar', 'Z']

    return hank_model, ss, unknowns, targets, exogenous
