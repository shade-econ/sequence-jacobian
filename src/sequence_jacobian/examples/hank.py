import numpy as np

from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.combined_block import create_model
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
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


@simple
def nkpc_ss(Z, mu):
    '''Solve (w) to hit targets for (nkpc_res)'''
    w = Z / mu
    return w


'''Part 2: Embed HA block'''


def make_grids(rho_s, sigma_s, nS, amax, nA):
    e_grid, pi_e, Pi = utils.discretize.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


def wages(w, e_grid):
    we = w * e_grid
    return we


def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne


'''Part 3: DAG'''

def dag():
    # Combine blocks
    household = hh.household.add_hetinputs([transfers, wages, make_grids])
    household = household.add_hetoutputs([labor_supply])
    blocks = [household, firm, monetary, fiscal, mkt_clearing, nkpc]
    blocks_ss = [household, firm, monetary, fiscal, mkt_clearing, nkpc_ss]
    hank_model = create_model(blocks, name="One-Asset HANK")
    hank_model_ss = create_model(blocks_ss, name="One-Asset HANK SS")

    # Steady state
    calibration = {'r': 0.005, 'rstar': 0.005, 'eis': 0.5, 'frisch': 0.5, 'B': 5.6,
                   'mu': 1.2, 'rho_s': 0.966, 'sigma_s': 0.5, 'kappa': 0.1, 'phi': 1.5,
                   'Y': 1., 'Z': 1., 'pi': 0., 'nS': 2, 'amax': 150, 'nA': 10}
    unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
    targets_ss = {'asset_mkt': 0., 'NE': 1.}
    cali = hank_model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, 
                                            solver='broyden_custom')
    ss = hank_model.steady_state(cali)

    # Transitional dynamics
    unknowns = ['w', 'Y', 'pi']
    targets = ['asset_mkt', 'goods_mkt', 'nkpc_res']
    exogenous = ['rstar', 'Z']

    return hank_model_ss, ss, hank_model, unknowns, targets, exogenous
