from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.combined_block import create_model
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
    I = K - (1 - delta) * K(-1)
    goods_mkt = Y - C - I
    return asset_mkt, goods_mkt, I


@simple
def firm_ss(r, Y, L, delta, alpha):
    '''Solve for (Z, K) given targets for (Y, r).'''
    rk = r + delta
    K = alpha * Y / rk
    Z = Y / K ** alpha / L ** (1 - alpha)
    w = (1 - alpha) * Z * (K / L) ** alpha
    return K, Z, w


'''Part 2: Embed HA block'''

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
    household = hh.household.add_hetinputs([income, make_grids])
    ks_model = create_model([household, firm, mkt_clearing], name="Krusell-Smith")
    ks_model_ss = create_model([household, firm_ss, mkt_clearing], name="Krusell-Smith SS")

    # Steady state
    calibration = {'eis': 1.0, 'delta': 0.025, 'alpha': 0.11, 'rho': 0.966, 'sigma': 0.5,
                   'Y': 1.0, 'L': 1.0, 'nS': 2, 'nA': 10, 'amax': 200, 'r': 0.01}
    unknowns_ss = {'beta': (0.98 / 1.01, 0.999 / 1.01)}
    targets_ss = {'asset_mkt': 0.}
    ss = ks_model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='brentq')

    # Transitional dynamics
    inputs = ['Z']
    unknowns = ['K']
    targets = ['asset_mkt']

    return ks_model_ss, ss, ks_model, unknowns, targets, inputs


'''Part 3: Permanent beta heterogeneity'''

@simple
def aggregate(A_patient, A_impatient, C_patient, C_impatient, mass_patient):
    C = mass_patient * C_patient + (1 - mass_patient) * C_impatient
    A = mass_patient * A_patient + (1 - mass_patient) * A_impatient
    return C, A


def remapped_dag():
    # Create 2 versions of the household block using `remap`
    household = hh.household.add_hetinputs([income, make_grids])
    to_map = ['beta', *household.outputs]
    hh_patient = household.remap({k: k + '_patient' for k in to_map}).rename('hh_patient')
    hh_impatient = household.remap({k: k + '_impatient' for k in to_map}).rename('hh_impatient')
    blocks = [hh_patient, hh_impatient, firm, mkt_clearing, aggregate]
    blocks_ss = [hh_patient, hh_impatient, firm_ss, mkt_clearing, aggregate]
    ks_remapped = create_model(blocks, name='KS-beta-het')
    ks_remapped_ss = create_model(blocks_ss, name='KS-beta-het')

    # Steady State
    calibration = {'eis': 1., 'delta': 0.025, 'alpha': 0.3, 'rho': 0.966, 'sigma': 0.5, 'Y': 1.0, 'L': 1.0,
                   'nS': 3, 'nA': 100, 'amax': 1000, 'beta_impatient': 0.985, 'mass_patient': 0.5}
    unknowns_ss = {'beta_patient': (0.98 / 1.01, 0.999 / 1.01)}
    targets_ss = {'asset_mkt': 0.}
    ss = ks_remapped_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='brentq')

    # Transitional Dynamics/Jacobian Calculation
    unknowns = ['K']
    targets = ['asset_mkt']
    exogenous = ['Z']

    return ks_remapped_ss, ss, ks_remapped, unknowns, targets, ss, exogenous
