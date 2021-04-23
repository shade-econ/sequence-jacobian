import numpy as np
from numba import vectorize, njit

from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.het_block import het


'''Part 1: HA block'''


def household_init(a_grid, e_grid, r, w, eis, T):
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return fininc, Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, Pi_p, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi):
    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""
    # this one is useful to do internally
    ws = w * e_grid

    # uc(z_t, a_t)
    uc_nextgrid = (beta * Pi_p) @ Va_p

    # c(z_t, a_t) and n(z_t, a_t)
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, ws[:, np.newaxis], eis, frisch, vphi)

    # c(z_t, a_{t-1}) and n(z_t, a_{t-1})
    lhs = c_nextgrid - ws[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
    rhs = (1 + r) * a_grid
    c = utils.interpolate.interpolate_y(lhs, rhs, c_nextgrid)
    n = utils.interpolate.interpolate_y(lhs, rhs, n_nextgrid)

    # test constraints, replace if needed
    a = rhs + ws[:, np.newaxis] * n + T[:, np.newaxis] - c
    iconst = np.nonzero(a < a_grid[0])
    a[iconst] = a_grid[0]

    # if there exist states/prior asset levels such that households want to borrow, compute the constrained
    # solution for consumption and labor supply
    if iconst[0].size != 0 and iconst[1].size != 0:
        c[iconst], n[iconst] = solve_cn(ws[iconst[0]], rhs[iconst[1]] + T[iconst[0]] - a_grid[0],
                                        eis, frisch, vphi, Va_p[iconst])

    # calculate marginal utility to go backward
    Va = (1 + r) * c ** (-1 / eis)

    # efficiency units of labor which is what really matters
    n_e = e_grid[:, np.newaxis] * n

    return Va, a, c, n, n_e


def transfers(pi_e, Div, Tax, e_grid):
    # default incidence rules are proportional to skill
    tax_rule, div_rule = e_grid, e_grid  # scale does not matter, will be normalized anyway

    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


household.add_hetinput(transfers, verbose=False)


@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


@vectorize
def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc = np.log(uc_seed)
    for i in range(30):
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        if abs(ne) < 1E-11:
            break
        else:
            log_uc -= ne / ne_p
    else:
        raise ValueError("Cannot solve constrained household's problem: No convergence after 30 iterations!")

    return np.exp(log_uc)


@njit
def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc


'''Part 2: Simple blocks and hetinput'''


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
def mkt_clearing(A, N_e, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = N_e - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


@simple
def income_state_vars(rho_s, sigma_s, nS):
    e_grid, pi_e, Pi = utils.discretize.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)
    return e_grid, pi_e, Pi


@simple
def asset_state_vars(amax, nA):
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    return a_grid


@simple
def partial_steady_state_solution(B_Y, Y, mu, r, kappa, Z, pi):
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B

    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) - (1 + pi).apply(np.log)

    return B, w, Div, Tax, nkpc_res


'''Part 3: Steady state'''


def hank_ss(beta_guess=0.986, vphi_guess=0.8, r=0.005, eis=0.5, frisch=0.5, mu=1.2, B_Y=5.6, rho_s=0.966, sigma_s=0.5,
            kappa=0.1, phi=1.5, nS=7, amax=150, nA=500):
    """Solve steady state of full GE model. Calibrate (beta, vphi) to hit target for interest rate and Y."""

    # set up grid
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    e_grid, pi_e, Pi = utils.discretize.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)

    # solve analytically what we can
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B
    T = transfers(pi_e, Div, Tax, e_grid)

    # initialize guess for policy function iteration
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # residual function
    def res(x):
        beta_loc, vphi_loc = x
        # precompute constrained c and n which don't depend on Va
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        out = household.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta_loc,
                           eis=eis, Div=Div, Tax=Tax, frisch=frisch, vphi=vphi_loc)
        return np.array([out['A'] - B, out['N_e'] - 1])

    # solve for beta, vphi
    (beta, vphi), _ = utils.solvers.broyden_solver(res, np.array([beta_guess, vphi_guess]), verbose=False)

    # extra evaluation for reporting
    ss = household.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta, eis=eis,
                      Div=Div, Tax=Tax, frisch=frisch, vphi=vphi)
    
    # check Walras's law
    goods_mkt = 1 - ss['C']
    assert np.abs(goods_mkt) < 1E-8
    
    # add aggregate variables
    ss.update({'B': B, 'phi': phi, 'kappa': kappa, 'Y': 1, 'rstar': r, 'Z': 1, 'mu': mu, 'L': 1, 'pi': 0,
               'rho_s': rho_s, 'labor_mkt': ss["N_e"] - 1, 'nA': nA, 'nS': nS, 'B_Y': B_Y, 'sigma_s': sigma_s,
               'goods_mkt': 1 - ss["C"], 'amax': amax, 'asset_mkt': ss["A"] - B, 'nkpc_res': kappa * (w - 1 / mu)})

    return ss
