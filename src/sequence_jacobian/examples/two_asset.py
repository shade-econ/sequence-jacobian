import numpy as np

from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.solved_block import solved
from ..blocks.combined_block import create_model, combine
from .hetblocks import household_twoasset as hh


'''Part 1: Blocks'''

@simple
def pricing(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1 / mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) \
           / (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc


@simple
def arbitrage(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc


@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) -\
        (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) +\
        K(+1) / K * Q(+1) - (1 + r(+1)) * Q
    return inv, val


@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - w * N - I - psip
    return psip, I, div


@simple
def taylor(rstar, pi, phi):
    i = rstar + phi * pi
    return i


@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax


@simple
def finance(i, p, pi, r, div, omega, pshare):
    rb = r - omega
    ra = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher


@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw


@simple
def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw) + beta * \
            (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log)
    return wnkpc


@simple
def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + psip + omega * B - Y
    return asset_mkt, wealth, goods_mkt


@simple
def share_value(p, tot_wealth, Bh):
    pshare = p / (tot_wealth - Bh)
    return pshare


@solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1 / mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / \
           (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc


@solved(unknowns={'p': (5, 15)}, targets=['equity'], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
    """Solves for (mup, alpha, Z, w) to hit (tot_wealth, Y, K, pi)."""
    # 1. Solve for markup to hit total wealth
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc

    # 2. Solve for capital share to hit K
    alpha = (r + delta) * K / Y / mc

    # 3. Solve for TFP to hit Y
    Z = Y * K ** (-alpha) * N ** (alpha - 1)

    # 4. Solve for w such that piw = 0
    w = mc * (1 - alpha) * Y / N

    return p, mc, mup, alpha, Z, w


@simple
def union_ss(tax, w, UCE, N, muw, frisch):
    """Solves for (vphi) to hit (wnkpc)."""
    vphi = (1 - tax) * w * UCE / muw / N ** (1 + 1 / frisch)
    wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * UCE / muw
    return vphi, wnkpc


'''Part 2: Embed HA block'''

def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = utils.discretize.agrid(amax=bmax, n=nB)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    k_grid = utils.discretize.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid


'''Part 3: DAG'''

def dag():
    # Combine Blocks
    household = hh.household.add_hetinputs([income, make_grids])
    production = combine([labor, investment])
    production_solved = production.solved(unknowns={'Q': 1., 'K': 10.},
                                          targets=['inv', 'val'], solver='broyden_custom')
    blocks = [household, pricing_solved, arbitrage_solved, production_solved,
              dividend, taylor, fiscal, share_value, finance, wage, union, mkt_clearing]
    two_asset_model = create_model(blocks, name='Two-Asset HANK')

    # Steadt state DAG
    blocks_ss = [household, partial_ss,
                 dividend, taylor, fiscal, share_value, finance, union_ss, mkt_clearing]
    two_asset_model_ss = create_model(blocks_ss, name='Two-Asset HANK SS')

    # Steady State
    calibration = {'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125, 'tot_wealth': 14,
                   'delta': 0.02, 'pi': 0.,
                   'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8, 'G': 0.2, 'eis': 0.5,
                   'frisch': 1, 'chi0': 0.25, 'chi2': 2, 'epsI': 4, 'omega': 0.005,
                   'kappaw': 0.1, 'phi': 1.5, 'nZ': 3, 'nB': 10, 'nA': 16, 'nK': 4,
                   'bmax': 50, 'amax': 4000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92}
    unknowns_ss = {'beta': 0.976, 'chi1': 6.5}
    targets_ss = {'asset_mkt': 0., 'B': 'Bh'}
    cali = two_asset_model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='broyden_custom')
    ss =  two_asset_model.steady_state(cali)
    
    # Transitional Dynamics/Jacobian Calculation
    unknowns = ['r', 'w', 'Y']
    targets = ['asset_mkt', 'fisher', 'wnkpc']
    exogenous = ['rstar', 'Z', 'G']

    return two_asset_model_ss, ss, two_asset_model, unknowns, targets, exogenous
