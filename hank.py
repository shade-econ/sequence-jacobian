import numpy as np
from numba import vectorize, njit

import utils
import het_block as het
import simple_block as sim
from simple_block import simple
import jacobian as jac


'''Part 1: HA block'''


@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


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


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


def backward_iterate(Va_p, Pi_p, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, c_const, n_const, ssflag=False):
    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility.

    Order of returns matters! backward_var, assets, others
    """

    # this one is useful to do internally
    ws = w * e_grid

    # uc(z_t, a_t)
    uc_nextgrid = (beta * Pi_p) @ Va_p

    # c(z_t, a_t) and n(z_t, a_t)
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, ws[:, np.newaxis], eis, frisch, vphi)

    # c(z_t, a_{t-1}) and n(z_t, a_{t-1})
    lhs = c_nextgrid - ws[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
    rhs = (1 + r) * a_grid
    c = utils.interpolate_y(lhs, rhs, c_nextgrid)
    n = utils.interpolate_y(lhs, rhs, n_nextgrid)

    # test constraints, replace if needed
    a = rhs + ws[:, np.newaxis] * n + T[:, np.newaxis] - c
    iconst = np.nonzero(a < a_grid[0])
    a[iconst] = a_grid[0]

    if ssflag:
        # use precomputed values
        c[iconst] = c_const[iconst]
        n[iconst] = n_const[iconst]
    else:
        # have to solve again if in transition
        uc_seed = c_const[iconst] ** (-1 / eis)
        c[iconst], n[iconst] = solve_cn(ws[iconst[0]],
                                        rhs[iconst[1]] + T[iconst[0]] - a_grid[0], eis, frisch, vphi, uc_seed)

    # calculate marginal utility to go backward
    Va = (1 + r) * c ** (-1 / eis)

    # efficiency units of labor which is what really matters
    ns = e_grid[:, np.newaxis] * n

    return Va, a, c, n, ns


household = het.HetBlock(backward_iterate, exogenous='Pi', policy='a', backward='Va')


'''Part 2: simple blocks'''


@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * np.log(1+pi)**2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi):
    # i = rstar + phi * pi
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) - np.log(1 + pi)
    return nkpc_res


@simple
def mkt_clearing(A, NS, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NS - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * np.log(1+pi)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


'''Part 3: Steady state'''


def hank_ss(beta_guess=0.986, vphi_guess=0.8, r=0.005, eis=0.5, frisch=0.5, mu=1.2, B_Y=5.6, rho_s=0.966, sigma_s=0.5,
            kappa=0.1, phi=1.5, nS=7, amax=150, nA=500, tax_rule=None, div_rule=None):
    """Solve steady state of full GE model. Calibrate (beta, vphi) to hit target for interest rate and Y."""

    # set up grid
    a_grid = utils.agrid(amax=amax, n=nA)
    e_grid, pi_s, Pi = utils.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)

    # default incidence rule is proportional to skill
    if tax_rule is None:
        tax_rule = e_grid  # scale does not matter, will be normalized anyway
    if div_rule is None:
        div_rule = e_grid

    assert tax_rule.shape[0] == div_rule.shape[0] == nS, 'Incidence rules are inconsistent with income grid.'

    # solve analitically what we can
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B
    div = Div / np.sum(pi_s * div_rule) * div_rule
    tax = Tax / np.sum(pi_s * tax_rule) * tax_rule
    T = div - tax

    # figure out initializer
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # residual function
    def res(x):
        beta_loc, vphi_loc = x
        # precompute constrained c and n which don't depend on Va
        c_const_loc, n_const_loc = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi_loc, Va)
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        # out = household_labor_ss(Pi, a_grid, e_grid, T, w, r, beta_loc, eis, frisch, vphi_loc)
        out = household.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, T=T, w=w, r=r, beta=beta_loc, eis=eis,
                           frisch=frisch, vphi=vphi_loc, c_const=c_const_loc, n_const=n_const_loc, ssflag=True)
        return np.array([out['A'] - B, out['NS'] - 1])

    # solve for beta, vphi
    (beta, vphi), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess]), noisy=False)

    # extra evaluation to report variables
    c_const, n_const = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi, Va)
    ss = household.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, T=T, w=w, r=r, beta=beta, eis=eis,
                      frisch=frisch, vphi=vphi, c_const=c_const, n_const=n_const, ssflag=True)
    ss.update({'pi_s': pi_s, 'B': B, 'phi': phi, 'kappa': kappa, 'Y': 1, 'rstar': r, 'Z': 1, 'mu': mu, 'L': 1, 'pi': 0,
               'Div': Div, 'Tax': Tax, 'div': div, 'tax': tax, 'div_rule': div_rule, 'tax_rule': tax_rule,
               'goods_mkt': 1 - ss['C'], 'ssflag': False})
    return ss


'''Part 4: Nonlinear transition'''


def td_map(ss, **kwargs):
    # initialize results
    results = kwargs

    # blocks before HA
    for b in [firm, monetary, fiscal]:
        results.update(b.td(ss, **{k: results[k] for k in b.inputs if k in results}))

    # hetinput
    results['div'] = results['Div'][:, np.newaxis] / np.sum(ss['pi_s'] * ss['div_rule']) * ss['div_rule']
    results['tax'] = results['Tax'][:, np.newaxis] / np.sum(ss['pi_s'] * ss['tax_rule']) * ss['tax_rule']

    # HA block
    results.update(household.td(ss, monotonic=True, T=results['div']-results['tax'],
                                **{k: results[k] for k in household.inputs if k in results}))

    # blocks after HA
    for b in [nkpc, mkt_clearing]:
        results.update(b.td(ss, **{k: results[k] for k in b.inputs if k in results}))

    return results
