import numpy as np
import scipy.optimize as opt
import utils
import het_block as het
from simple_block import simple


'''Part 1: HA block'''


def backward_iterate(Va_p, Pi_p, a_grid, e_grid, r, w, beta, eis):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Order of returns matters! backward_var, assets, others

    Parameters
    ----------
    Va_p : np.ndarray
        marginal value of assets tomorrow
    Pi_p : np.ndarray
        Markov transition matrix for skills tomorrow
    a_grid : np.ndarray
        asset grid
    e_grid : np.ndarray
        skill grid
    r : float
        ex-post interest rate
    w : float
        wage
    beta : float
        discount rate today
    eis : float
        elasticity of intertemporal substitution

    Returns
    ----------
    Va : np.ndarray, shape(nS, nA)
        marginal value of assets today
    a : np.ndarray, shape(nS, nA)
        asset policy today
    c : np.ndarray, shape(nS, nA)
        consumption policy today
    """
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = utils.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


household = het.HetBlock(backward_iterate, exogenous='Pi', policy='a', backward='Va')


'''Part 2: Simple Blocks'''


@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A):
    asset_mkt = K - A
    return asset_mkt


'''Part 3: Steady state'''


def ks_ss(lb=0.98, ub=0.999, r=0.01, eis=1, delta=0.025, alpha=0.11, rho=0.966, sigma=0.5, nS=7, nA=500, amax=200):
    """Solve steady state of full GE model. Calibrate beta to hit target for interest rate."""
    # set up grid
    a_grid = utils.agrid(amax=amax, n=nA)
    e_grid, _, Pi = utils.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)

    # solve for aggregates analytically
    rk = r + delta
    Z = (rk / alpha) ** alpha  # normalize so that Y=1
    K = (alpha * Z / rk) ** (1 / (1 - alpha))
    Y = Z * K ** alpha
    w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))

    # figure out initializer
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # solve for beta consistent with this
    beta_min = lb / (1 + r)
    beta_max = ub / (1 + r)
    beta, sol = opt.brentq(lambda bet: household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=bet, eis=eis,
                                                    Va=Va)['A'] - K, beta_min, beta_max, full_output=True)
    if not sol.converged:
        raise ValueError('Steady-state solver did not converge.')

    # extra evaluation to report variables
    ss = household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=beta, eis=eis, Va=Va)
    ss.update({'Z': Z, 'K': K, 'L': 1, 'Y': Y, 'alpha': alpha, 'delta': delta, 'goods_mkt': Y - ss['C'] - delta * K})

    return ss
