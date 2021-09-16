from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.combined_block import create_model, combine
from .hetblocks import household_sim as hh


'''Part 1: Blocks'''

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
def firm_ss_solution(r, Y, L, delta, alpha):
    '''Solve for (Z, K) given targets for (Y, r).'''
    rk = r + delta
    K = alpha * Y / rk
    Z = Y / K ** alpha / L ** (1 - alpha)
    return K, Z


'''Part 2: Embed HA block'''

@simple
def make_grids(rho, sigma, nS, amax, nA):
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return e_grid, Pi, a_grid


def income(w, e_grid):
    y = w * e_grid
    return y


'''Part 3: DAG'''

def dag():
    # Combine blocks
    household = hh.household.add_hetinputs([income])
    blocks = [household, firm, make_grids, mkt_clearing]
    helper_blocks = [firm_ss_solution]
    ks_model = create_model(blocks, name="Krusell-Smith")

    # Steady state
    calibration = {'eis': 1, 'delta': 0.025, 'alpha': 0.11, 'rho': 0.966, 'sigma': 0.5,
                   'L': 1.0, 'nS': 2, 'nA': 10, 'amax': 200, 'r': 0.01}
    unknowns_ss = {'beta': (0.98 / 1.01, 0.999 / 1.01), 'Z': 0.85, 'K': 3.}
    targets_ss = {'asset_mkt': 0., 'Y': 1., 'r': 0.01}
    ss = ks_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='brentq',
                                     helper_blocks=helper_blocks, helper_targets=['Y', 'r'])

    # Transitional dynamics
    inputs = ['Z']
    unknowns = ['K']
    targets = ['asset_mkt']

    return ks_model, ss, unknowns, targets, inputs


'''Part 3: Permanent beta heterogeneity'''

@simple
def aggregate(A_patient, A_impatient, C_patient, C_impatient, mass_patient):
    C = mass_patient * C_patient + (1 - mass_patient) * C_impatient
    A = mass_patient * A_patient + (1 - mass_patient) * A_impatient
    return C, A


def remapped_dag():
    # Create 2 versions of the household block using `remap`
    to_map = ['beta', *household.outputs]
    hh_patient = household.remap({k: k + '_patient' for k in to_map}).rename('hh_patient')
    hh_impatient = household.remap({k: k + '_impatient' for k in to_map}).rename('hh_impatient')
    blocks = [hh_patient, hh_impatient, firm, mkt_clearing, aggregate]
    ks_remapped = create_model(blocks, name='KS-beta-het')

    # Steady State
    calibration = {'eis': 1., 'delta': 0.025, 'alpha': 0.3, 'rho': 0.966, 'sigma': 0.5, 'L': 1.0,
                   'nS': 3, 'nA': 100, 'amax': 1000, 'beta_impatient': 0.985, 'mass_patient': 0.5}
    unknowns_ss = {'beta_patient': (0.98 / 1.01, 0.999 / 1.01), 'Z': 0.5, 'K': 8.}
    targets_ss = {'asset_mkt': 0., 'Y': 1., 'r': 0.01}
    helper_blocks = [firm_ss_solution]
    ss = ks_remapped.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='brentq',
                                        helper_blocks=helper_blocks, helper_targets=['Y', 'r'])

    # Transitional Dynamics/Jacobian Calculation
    unknowns = ['K']
    targets = ['asset_mkt']
    exogenous = ['Z']

    return ks_remapped, ss, unknowns, targets, ss, exogenous
