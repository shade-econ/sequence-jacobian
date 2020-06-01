import numpy as np
from numba import njit
import utils
from het_block import het
from simple_block import simple


'''Part 1: HA block'''


@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'])  # order as in grid!
def household(Va_p, Vb_p, Pi_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    # get grid dimensions
    nZ, nB, nA = Va_p.shape
    nK = k_grid.shape[0]

    # step 2: Wb(z, b', a') and Wa(z, b', a')
    Wb = matrix_times_first_dim(beta*Pi_p, Vb_p)
    Wa = matrix_times_first_dim(beta*Pi_p, Va_p)
    W_ratio = Wa / Wb

    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, np.newaxis],
                             a_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]

    # step 3: a'(z, b', a) for UNCONSTRAINED
    k_solve, pi_solve = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)
    a_endo_unc = utils.apply_coord(k_solve, pi_solve, a_grid)
    c_endo_unc = utils.apply_coord(k_solve, pi_solve, Wb) ** (-eis)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    k_solve, pi_solve = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)
    a_endo_con = utils.apply_coord(k_solve, pi_solve, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis])**(-eis) 
                    * utils.apply_coord(k_solve, pi_solve, Wb[:, 0:1, :]) ** (-eis))

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2, nK)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)
    c = addouter(z_grid, (1+rb)*b_grid, (1+ra)*a_grid) - Psi - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    return Va, Vb, a, b, c, u


def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, np.newaxis, np.newaxis] + b[:, np.newaxis] + a


def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)

    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)

    # Psi1 and Psi2 are derivatives of Psi wrt ap and a, respectively
    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra)*(Psi1 + (chi2 - 1)*Psi/adj_denominator)
    return Psi, Psi1, Psi2


@njit
def lhs_equals_rhs_interpolate(lhs, rhs):
    """Given lhs (i,j,k) and rhs (k,l), for each (i,j,l) triplet, find the k such that

    lhs[i,j,k] > rhs[k,l] and lhs[i,j,k+1] < rhs[k+1,l]
    
    i.e. where lhs == rhs in between k and k+1.
    
    Also return the pi such that 

    pi*(lhs[i,j,k] - rhs[k,l]) + (1-pi)*(lhs[i,j,k+1] - rhs[k+1,l]) == 0

    i.e. such that the point at pi*k + (1-pi)*(k+1) satisfies lhs == rhs by linear interpolation.

    If lhs[i,j,0] < rhs[0,l] already, just return k=0 and pi=1.

    **IMPORTANT: Assumes that solution k is monotonically increasing in l and that 
    lhs - rhs is monotonically decreasing in k.***
    """
    
    ni, nj, nk = lhs.shape
    nk2, nl = rhs.shape
    assert nk == nk2

    k_solve = np.empty((ni, nj, nl), dtype=np.int64)
    pi_solve = np.empty((ni, nj, nl))

    for i in range(ni):
        for j in range(nj):
            k = 0
            for l in range(nl):
                while True:
                    if lhs[i, j, k] < rhs[k, l]:
                        break
                    elif k < nk - 1:
                        k += 1
                    else:
                        break

                if k == 0:
                    k_solve[i, j, l] = 0
                    pi_solve[i, j, l] = 1
                else:
                    k_solve[i, j, l] = k-1
                    err_upper = rhs[k, l] - lhs[i, j, k]
                    err_lower = rhs[k-1, l] - lhs[i, j, k-1]
                    pi_solve[i, j, l] =  err_upper / (err_upper - err_lower)

    return k_solve, pi_solve


def step4(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, b', a)
    b_endo = (c_endo + ap_endo + addouter(-z_grid, b_grid, -(1+ra)*a_grid)
            + get_Psi_and_deriv(ap_endo, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) > 0, 'b(bp) is not increasing'
    # assert np.min(np.diff(ap_endo, axis=1)) > 0, 'ap(bp) is not increasing'
    i, pi = utils.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    ap = utils.apply_coord(i, pi, ap_endo.swapaxes(1, 2)).swapaxes(1, 2)
    bp = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2)
    return bp, ap


def step6(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2, nK):
    # b(z, k, a)
    b_endo = (c_endo + ap_endo + addouter(-z_grid, np.full(nK, b_grid[0]), -(1+ra)*a_grid)
            + get_Psi_and_deriv(ap_endo, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) < 0, 'b(kappa) is not decreasing'
    # assert np.min(np.diff(ap_endo, axis=1)) < 0, 'ap(kappa) is not decreasing'
    ap = utils.interpolate_y(b_endo[:, ::-1, :].swapaxes(1, 2), b_grid, 
                             ap_endo[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)
    return ap


'''Part 2: Simple blocks'''


@simple
def pricing(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1/mup) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) - np.log(1 + pi)
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
    inv = (K/K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1-alpha) * mc(+1) - (K(+1)/K -
           (1-delta) + (K(+1)/K - 1)**2 / (2*delta*epsI)) + K(+1)/K*Q(+1) - (1 + r(+1))*Q
    return inv, val


@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta):
    psip = mup / (mup - 1) / 2 / kappap * np.log(1 + pi) ** 2 * Y
    I = K - (1 - delta) * K(-1)
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
    ra = pshare * (div + p) / p(-1) + (1-pshare) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher


@simple
def wage(pi, w, N, muw, kappaw):
    piw = (1 + pi) * w / w(-1) - 1
    psiw = muw / (1 - muw) / 2 / kappaw * np.log(1 + piw) ** 2 * N
    return piw, psiw


@simple
def union(piw, N, tax, w, U, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N**(1+1/frisch) - muw*(1-tax)*w*N*U) + beta * np.log(1 + piw(+1)) - np.log(1 + piw)
    return wnkpc


@simple
def mkt_clearing(p, A, B, Bg):
    asset_mkt = p + Bg - B - A
    return asset_mkt


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid


household_inc = household.attach_hetinput(income)


'''Part 3: Steady state'''


def hank_ss(beta_guess=0.976, vphi_guess=2.07, chi1_guess=6.5, r=0.0125, tot_wealth=14, K=10, delta=0.02, kappap=0.1,
            muw=1.1, Bh=1.04, Bg=2.8, G=0.2, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
            phi=1.5, nZ=3, nB=50, nA=70, nK=50, bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92, noisy=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)
    k_grid = utils.agrid(amax=kmax, n=nK)
    e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

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
        beta_loc, vphi_loc, chi1_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001 or chi1_loc < 0.5:
            raise ValueError('Clearly invalid inputs')
        out = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                               k_grid=k_grid, beta=beta_loc, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2)
        asset_mkt = out['A'] + out['B'] - p - Bg
        labor_mkt = vphi_loc - muw * (1 - tax) * w * out['U']
        return np.array([asset_mkt, labor_mkt, out['B'] - Bh])

    # solve for beta, vphi, omega
    (beta, vphi, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess, chi1_guess]), noisy=noisy)

    # extra evaluation to report variables
    ss = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                          k_grid=k_grid, beta=beta, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2)

    # other things of interest
    pshare = p / (tot_wealth - Bh)

    # calculate aggregate adjustment cost and check Walras's law
    chi = get_Psi_and_deriv(ss['a'], a_grid, r, chi0, chi1, chi2)[0]
    Chi = np.vdot(ss['D'], chi)
    goods_mkt = ss['C'] + I + G + Chi + omega * ss['B'] - 1
    assert np.abs(goods_mkt) < 1E-7

    ss.update({'pi': 0, 'piw': 0, 'Q': 1, 'Y': 1, 'N': 1, 'mc': mc, 'K': K, 'Z': Z, 'I': I, 'w': w, 'tax': tax,
               'div': div, 'p': p, 'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi, 'goods_mkt': goods_mkt, 'chi': chi, 'phi': phi,
               'beta': beta, 'vphi': vphi, 'omega': omega, 'alpha': alpha, 'delta': delta, 'mup': mup, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z_grid, 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'pshare': pshare, 'rstar': r, 'i': r})
    return ss
