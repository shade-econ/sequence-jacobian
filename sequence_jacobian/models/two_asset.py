# pylint: disable=E1120
import numpy as np
from numba import guvectorize

from .. import utilities as utils
from ..blocks.simple_block import simple
from ..blocks.het_block import het, hetoutput
from ..blocks.solved_block import solved
from ..blocks.support.simple_displacement import apply_function

'''Part 1: HA block'''


def household_init(b_grid, a_grid, e_grid, eis, tax, w):
    z_grid = income(e_grid, tax, w, 1)
    Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))

    return z_grid, Va, Vb


@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'], backward_init=household_init)  # order as in grid!
def household(Va_p, Vb_p, Pi_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    # require that k is decreasing (new)
    assert k_grid[1] < k_grid[0], 'kappas in k_grid must be decreasing!'

    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, np.newaxis],
                             a_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]

    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = matrix_times_first_dim(beta * Pi_p, Vb_p)
    Wa = matrix_times_first_dim(beta * Pi_p, Va_p)
    W_ratio = Wa / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1+Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_unc = utils.interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = utils.interpolate.apply_coord(i, pi, Wb) ** (-eis)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + a_endo_unc + addouter(-z_grid, b_grid, -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = utils.interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    a_unc = utils.interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = utils.interpolate.apply_coord(i, pi, b_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_con = utils.interpolate.apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * utils.interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_grid[0]), -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = utils.interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_grid,
                                            a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c = addouter(z_grid, (1 + rb) * b_grid, (1 + ra) * a_grid) - Psi - a - b
    uc = c ** (-1 / eis)

    # for GE wage Phillips curve we'll need endowment-weighted utility too
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    return Va, Vb, a, b, c, u


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid


# A potential hetoutput to include with the above HetBlock
@hetoutput()
def adjustment_costs(a, a_grid, r, chi0, chi1, chi2):
    chi, _, _ = apply_function(get_Psi_and_deriv, a, a_grid, r, chi0, chi1, chi2)
    return chi


household.add_hetinput(income, verbose=False)
household.add_hetoutput(adjustment_costs, verbose=False)


"""Supporting functions for HA block"""


def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """Adjustment cost Psi(ap, a) and its derivatives with respect to
    first argument (ap) and second argument (a)"""
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)

    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)

    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
    return Psi, Psi1, Psi2


def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, np.newaxis, np.newaxis] + b[:, np.newaxis] + a


@guvectorize(['void(float64[:], float64[:,:], uint32[:], float64[:])'], '(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs, rhs, iout, piout):
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

    lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]

    i.e. where given j, lhs == rhs in between i and i+1.

    Also return the pi such that

    pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.

    If lhs[0] < rhs[0,j] already, just return u=0 and pi=1.

    ***IMPORTANT: Assumes that solution i is monotonically increasing in j
    and that lhs - rhs is monotonically decreasing in i.***
    """

    ni, nj = rhs.shape
    assert len(lhs) == ni

    i = 0
    for j in range(nj):
        while True:
            if lhs[i] < rhs[i, j]:
                break
            elif i < nj - 1:
                i += 1
            else:
                break

        if i == 0:
            iout[j] = 0
            piout[j] = 1
        else:
            iout[j] = i - 1
            err_upper = rhs[i, j] - lhs[i]
            err_lower = rhs[i - 1, j] - lhs[i - 1]
            piout[j] = err_upper / (err_upper - err_lower)


'''Part 2: Simple blocks'''


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
    val = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) - (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (
                                                                2 * delta * epsI)) + K(+1) / K * Q(+1) - (1 + r(+1)) * Q
    return inv, val


@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - w * N - I
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
def union(piw, N, tax, w, U, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * U / muw) + beta * \
            (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log)
    return wnkpc


@simple
def mkt_clearing(p, A, B, Bg, C, I, G, Chi, psip, omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + Chi + psip + omega * B - Y
    return asset_mkt, wealth, goods_mkt


@simple
def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = utils.discretize.agrid(amax=bmax, n=nB)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    k_grid = utils.discretize.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    return b_grid, a_grid, k_grid, e_grid, Pi


@simple
def share_value(p, tot_wealth, Bh):
    pshare = p / (tot_wealth - Bh)
    return pshare


@simple
def partial_ss_step1(Y, N, K, r, tot_wealth, Bg, delta):
    """Solves for (mup, alpha, Z, w) to hit (tot_wealth, N, K, pi)."""
    # 1. Solve for markup to hit total wealth
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc
    wealth = tot_wealth

    # 2. Solve for capital share to hit K
    alpha = (r + delta) * K / Y / mc

    # 3. Solve for TFP to hit N (Y cannot be used, because it is an unknown of the DAG)
    Z = Y * K ** (-alpha) * N ** (alpha - 1)

    # 4. Solve for w such that piw = 0
    w = mc * (1 - alpha) * Y / N
    piw = 0

    return p, mc, mup, wealth, alpha, Z, w, piw


@simple
def partial_ss_step2(tax, w, U, N, muw, frisch):
    """Solves for (vphi) to hit (wnkpc)."""
    vphi = (1 - tax) * w * U / muw / N ** (1 + 1 / frisch)
    wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * U / muw
    return vphi, wnkpc


'''Part 3: Steady state'''


def two_asset_ss(beta_guess=0.976, chi1_guess=6.5, r=0.0125, tot_wealth=14, K=10, delta=0.02, kappap=0.1,
                 muw=1.1, Bh=1.04, Bg=2.8, G=0.2, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
                 phi=1.5, nZ=3, nB=50, nA=70, nK=50, bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92,
                 verbose=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.discretize.agrid(amax=bmax, n=nB)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    k_grid = utils.discretize.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    # solve analytically what we can
    I = delta * K
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    mup = 1 / mc
    Z = K ** (-alpha)
    w = (1 - alpha) * mc
    tax = (r * Bg + G) / w
    div = 1 - w - I
    p = div / r
    ra = r
    rb = r - omega

    # figure out initializer
    z_grid = income(e_grid, tax, w, 1)
    Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))

    # residual function
    def res(x):
        beta_loc, chi1_loc = x
        if beta_loc > 0.999 / (1 + r) or chi1_loc < 0.5:
            raise ValueError('Clearly invalid inputs')
        out = household.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                           k_grid=k_grid, beta=beta_loc, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2)
        asset_mkt = out['A'] + out['B'] - p - Bg
        return np.array([asset_mkt, out['B'] - Bh])

    # solve for beta, vphi, omega
    (beta, chi1), _ = utils.solvers.broyden_solver(res, np.array([beta_guess, chi1_guess]), verbose=verbose)

    # extra evaluation to report variables
    ss = household.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                      k_grid=k_grid, beta=beta, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2)

    # other things of interest
    vphi = (1 - tax) * w * ss['U'] / muw
    pshare = p / (tot_wealth - Bh)

    # calculate aggregate adjustment cost and check Walras's law
    chi = get_Psi_and_deriv(ss.internal["household"]['a'], a_grid, r, chi0, chi1, chi2)[0]
    Chi = np.vdot(ss.internal["household"]['D'], chi)
    goods_mkt = ss['C'] + I + G + Chi + omega * ss['B'] - 1

    ss.internal["household"].update({"chi": chi})
    ss.update({'pi': 0, 'piw': 0, 'Q': 1, 'Y': 1, 'N': 1, 'mc': mc, 'K': K, 'Z': Z, 'I': I, 'w': w, 'tax': tax,
               'div': div, 'p': p, 'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi, 'phi': phi, 'wealth': tot_wealth,
               'beta': beta, 'vphi': vphi, 'omega': omega, 'alpha': alpha, 'delta': delta, 'mup': mup, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z_grid, 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'pshare': pshare, 'rstar': r, 'i': r,
               'tot_wealth': tot_wealth, 'fisher': 0, 'nZ': nZ, 'Bh': Bh, 'psip': 0, 'inv': 0, 'goods_mkt': goods_mkt,
               'equity': div + p - p * (1 + r), 'bmax': bmax, 'rho_z': rho_z, 'asset_mkt': p + Bg - ss["B"] - ss["A"],
               'nA': nA, 'nB': nB, 'amax': amax, 'kmax': kmax, 'nK': nK, 'nkpc': kappap * (mc - 1 / mup),
               'wnkpc': kappaw * (vphi * ss["N"] ** (1 + 1 / frisch) - (1 - tax) * w * ss["N"] * ss["U"] / muw),
               'sigma_z': sigma_z, 'val': alpha * Z * (ss["N"] / K) ** (1 - alpha) * mc - delta - r})
    return ss


'''Part 4: Solved blocks for transition dynamics/Jacobian calculation'''


@solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1 / mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / \
           (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc


@solved(unknowns={'p': (5, 15)}, targets=['equity'], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


production_solved = solved(block_list=[labor, investment], unknowns={'Q': 1., 'K': 10.},
                           targets=['inv', 'val'], solver="broyden_custom")
