import numpy as np
from numba import vectorize, njit

import utils
import solvers
import het_block as het
import rec_block as rec
from rec_block import recursive
import jacobian as jac

"""Part 1: static problem at the boundary"""


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


"""Part 2: Steady state"""


def backward_iterate_labor(Va_p, Pi_p, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, c_const, n_const, ssflag):
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


def pol_labor_ss(Pi, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, Va_seed=None, tol=1E-8, maxit=5000):
    if Va_seed is None:
        coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
        Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    else:
        Va = Va_seed

    # precompute constrained c and n which don't depend on Va
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    c_const, n_const = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi, Va)

    # iterate until convergence of a policy by tol or reach max number of iterations
    a = np.empty_like(Va)
    for it in range(maxit):
        Va, anew, c, n, ns = backward_iterate_labor(Va, Pi, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi,
                                                    c_const, n_const, ssflag=True)

        if it % 10 == 1 and utils.within_tolerance(a, anew, tol):
            break
        a = anew
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')

    return Va, a, c, n, ns, c_const, n_const


def household_labor_ss(Pi, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, Va_seed=None, D_seed=None, pi_seed=None):
    """Solve for steady-state policies and distribution. Report results in dict."""
    # solve ha block
    Va, a, c, n, ns, c_const, n_const = pol_labor_ss(Pi, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, Va_seed)
    D = utils.dist_ss(a, Pi, a_grid, D_seed, pi_seed)

    # return handy dict with results and inputs
    inputs = {'Pi': Pi, 'a_grid': a_grid, 'e_grid': e_grid, 'T': T, 'w': w, 'r': r, 'beta': beta, 'eis': eis,
              'frisch': frisch, 'vphi': vphi}
    results = {'D': D, 'Va': Va, 'a': a, 'c': c, 'n': n, 'ns': ns, 'c_const': c_const, 'n_const': n_const,
               'A': np.vdot(D, a), 'C': np.vdot(D, c), 'N': np.vdot(D, n), 'NS': np.vdot(D, ns)}

    return {**inputs, **results}


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

    # residual function
    def res(x):
        beta_loc, vphi_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        out = household_labor_ss(Pi, a_grid, e_grid, T, w, r, beta_loc, eis, frisch, vphi_loc)
        return np.array([out['A'] - B, out['NS'] - 1])

    # solve for beta, vphi
    (beta, vphi), _ = solvers.broyden_solver(res, np.array([beta_guess, vphi_guess]), noisy=False)

    # extra evaluation to report variables
    ss = household_labor_ss(Pi, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi)
    ss.update({'pi_s': pi_s, 'B': B, 'phi': phi, 'kappa': kappa, 'Y': 1, 'rstar': r, 'Z': 1, 'mu': mu, 'L': 1, 'pi': 0,
               'Div': Div, 'Tax': Tax, 'div': div, 'tax': tax, 'div_rule': div_rule, 'tax_rule': tax_rule,
               'goods_mkt': 1 - ss['C'], 'ssflag': False})
    return ss


'''Part 3: linearized transition dynamics'''


@recursive
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * np.log(1+pi)**2 * Y
    return L, Div


@recursive
def monetary(pi, rstar, phi):
    # i = rstar + phi * pi
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@recursive
def fiscal(r, B):
    Tax = r * B
    return Tax


@recursive
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) - np.log(1 + pi)
    return nkpc_res


@recursive
def mkt_clearing(A, NS, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NS - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * np.log(1+pi)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


def get_J(ss, T):
    """Compute Jacobians along computational graph: for r, w, curlyK as functions of Z and K."""

    # jacobians for simple blocks
    J_firm = rec.all_Js(firm, ss, T)
    J_monetary = rec.all_Js(monetary, ss, T, ['pi', 'rstar'])
    J_fiscal = rec.all_Js(fiscal, ss, T, ['r'])
    J_nkpc = rec.all_Js(nkpc, ss, T, ['pi', 'w', 'Z', 'Y', 'r'])
    J_mkt = rec.all_Js(mkt_clearing, ss, T, ['A', 'NS', 'C', 'L', 'Y'])

    # jacobian of HA block
    T_div = ss['div_rule'] / np.sum(ss['pi_s'] * ss['div_rule'])
    T_tax = -ss['tax_rule'] / np.sum(ss['pi_s'] * ss['tax_rule'])
    J_ha = het.all_Js(backward_iterate_labor, ss, T, {'r': {'r': 1},
                                                      'w': {'w': 1},
                                                      'Div': {'T': T_div},
                                                      'Tax': {'T': T_tax}})

    # now combine all into a single jacobian, ORDER OF JACDICTS MATTERS
    J = jac.chain_jacobians(jacdicts=[J_firm, J_monetary, J_fiscal, J_ha, J_mkt, J_nkpc],
                            inputs=['w', 'Y', 'pi', 'rstar', 'Z'], T=T)

    return J


def get_G(J, T):
    exogenous = ['rstar', 'Z']
    unknowns = ['pi', 'w', 'Y']
    targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

    # assemble large Jacobian
    H_X = jac.pack_jacobians(J, unknowns, targets, T)

    # take -inverse, still shock-independent, and unpack to a jacdict
    J_unknowns = jac.unpack_jacobians(-np.linalg.inv(H_X), targets, unknowns, T)

    # get G for X by chaining this together with J's targets
    J_targets = {k: v for k, v in J.items() if k in targets}
    G = jac.chain_jacobians(jacdicts=[J_targets, J_unknowns, J], inputs=exogenous, T=T)

    return G


def td_linear(G, shockdict, outputs=None):
    if outputs is not None:
        G = {k: G[k] for k in outputs}
    tdout = jac.apply_jacobians(G, shockdict)
    return {**tdout, **shockdict}


'''Part 2b: Nonlinear starts here'''


def household_td(back_it_fun, ss, **kwargs):
    """Calculate partial equilibrium response of household to shocks to any of its inputs given in kwargs.

    Not allowed to shock transition matrix or a_grid.
    """
    # infer T from kwargs, check that all shocks have same length
    shock_lengths = [x.shape[0] for x in kwargs.values()]
    assert shock_lengths[1:] == shock_lengths[:-1], 'Shocks with different length.'
    T = shock_lengths[0]

    # get steady state inputs
    ssinput_dict, _, _, _ = het.extract_info(back_it_fun, ss)

    # make new dict of all the ss that are not shocked
    fixed_inputs = {k: v for k, v in ssinput_dict.items() if k not in kwargs}

    # allocate empty arrays to store results
    Va_path, a_path, c_path, n_path, ns_path, D_path = (np.empty((T,) + ss['a'].shape) for _ in range(6))

    # backward iteration
    for t in reversed(range(T)):
        if t == T - 1:
            Va_p = ssinput_dict['Va_p']
        else:
            Va_p = Va_path[t + 1, ...]

        backward_inputs = {**fixed_inputs, **{k: v[t, ...] for k, v in kwargs.items()}, 'Va_p': Va_p}  # order matters
        Va_path[t, ...], a_path[t, ...], c_path[t, ...], n_path[t, ...], ns_path[t, ...] = backward_iterate_labor(
            **backward_inputs)

    # forward iteration
    Pi_T = ss['Pi'].T.copy()
    D_path[0, ...] = ss['D']
    for t in range(T):
        a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], a_path[t, ...])
        if t < T-1:
            D_path[t+1, ...] = utils.forward_step(D_path[t, ...], Pi_T, a_pol_i, a_pol_pi)

    # return paths and aggregates
    return {'Va': Va_path, 'a': a_path, 'c': c_path, 'n': n_path, 'ns': ns_path, 'D': D_path,
            'A': np.sum(D_path * a_path, axis=(1, 2)), 'C': np.sum(D_path * c_path, axis=(1, 2)),
            'N': np.sum(D_path * n_path, axis=(1, 2)), 'NS': np.sum(D_path * ns_path, axis=(1, 2))}


def td_map(ss, pi, w, Y, rstar, Z):
    # simple blocks
    L, Div = firm.td(ss, Y=Y, w=w, Z=Z, pi=pi)
    r = monetary.td(ss, pi=pi, rstar=rstar)
    Tax = fiscal.td(ss, r=r)
    div = Div[:, np.newaxis] / np.sum(ss['pi_s'] * ss['div_rule']) * ss['div_rule']
    tax = Tax[:, np.newaxis] / np.sum(ss['pi_s'] * ss['tax_rule']) * ss['tax_rule']

    # ha block
    td = household_td(backward_iterate_labor, ss, T=div-tax, w=w, r=r)
    td.update({'pi': pi, 'w': w, 'Y': Y, 'L': L, 'r': r, 'Div': Div, 'Tax': Tax, 'rstar': rstar, 'Z': Z})

    # nkpc and market clearing
    nkpc_res = nkpc.td(ss, pi=pi, w=w, Y=Y, Z=Z, r=r)
    asset_mkt, labor_mkt, td['goods_mkt'] = mkt_clearing.td(ss, **{k: td[k] for k in ('A', 'NS', 'C', 'L', 'Y', 'pi')})

    return nkpc_res, asset_mkt, labor_mkt, td


def td_nonlinear(ss, J, rstar, Z, tol=1E-8, maxit=30, noisy=True):
    T = rstar.shape[0]

    # assemble large jacobian
    H_X = jac.pack_jacobians(J, inputs=['pi', 'w', 'Y'], outputs=['nkpc_res', 'asset_mkt', 'labor_mkt'], T=T)
    H_X_inv = np.linalg.inv(H_X)

    # initialize guess at ss
    pi = np.full(T, 0)
    w = np.full(T, ss['w'])
    Y = np.full(T, ss['Y'])
    guesses = np.concatenate((pi, w, Y))

    # iterate until convergence
    for it in range(maxit):
        pi, w, Y = guesses[:T], guesses[T:2*T], guesses[2*T:],
        nkpc_res, asset_mkt, labor_mkt, td = td_map(ss, pi, w, Y, rstar, Z)
        residual = np.concatenate((nkpc_res, asset_mkt, labor_mkt))
        error = np.max(np.abs(residual))
        if noisy:
            print(f'Max error {error:.2E} on iteration {it}')
        if error < tol:
            break
        else:
            guesses -= H_X_inv @ residual
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')

    return td
