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
    Wb, Wa = post_decision_vfun(Va_p, Vb_p, Pi_p, beta)

    # step 3: a'(z, b', a) for UNCONSTRAINED
    lhs_unc = Wa / Wb
    Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)
    a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, k_grid, eis, nZ, nK, nA)

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    return Va, Vb, a, b, c, u


def post_decision_vfun(Va_p, Vb_p, Pi, beta):
    Wb = (Vb_p.T @ (beta * Pi.T)).T
    Wa = (Va_p.T @ (beta * Pi.T)).T
    return Wb, Wa


def Psi_fun(ap, a, ra, chi0, chi1, chi2):
    return chi1 / chi2 * np.abs((ap - (1 + ra) * a)) ** chi2 / ((1 + ra) * a + chi0)


def Psi1_fun(ap, a, ra, chi0, chi1, chi2):
    return np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)


def Psi2_fun(ap, a, ra, chi0, chi1, chi2):
    Psi1 = np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)
    return -(1 + ra) * (Psi1 + chi1 * (chi2 - 1) / chi2 * (np.abs(ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** chi2)


@njit
def step3(lhs, rhs, Wb, a_grid, eis, nZ, nB, nA):
    ap_endo = np.empty((nZ, nB, nA))
    Wb_endo = np.empty((nZ, nB, nA))
    for iz in range(nZ):
        for ibp in range(nB):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ibp, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ibp, ia] = 0
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, 0]
                elif iap == nA:
                    ap_endo[iz, ibp, ia] = a_grid[iap]
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap]
                else:
                    y0 = lhs[iz, ibp, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ibp, iap] - rhs[iap, ia]
                    ap_endo[iz, ibp, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap - 1] + (
                                ap_endo[iz, ibp, ia] - a_grid[iap - 1]) * (
                                Wb[iz, ibp, iap] - Wb[iz, ibp, iap - 1]) / (a_grid[iap] - a_grid[iap - 1])
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step4(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, b', a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + bbb - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) > 0, 'b(bp) is not increasing'
    # assert np.min(np.diff(ap_endo, axis=1)) > 0, 'ap(bp) is not increasing'
    i, pi = utils.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    ap = utils.apply_coord(i, pi, ap_endo.swapaxes(1, 2)).swapaxes(1, 2)
    bp = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2)
    return bp, ap


@njit
def step5(lhs, rhs, Wb, a_grid, k_grid, eis, nZ, nK, nA):
    ap_endo = np.empty((nZ, nK, nA))
    Wb_endo = np.empty((nZ, nK, nA))
    for iz in range(nZ):
        for ik in range(nK):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ik, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ik, ia] = 0
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, 0]
                elif iap == nA:
                    ap_endo[iz, ik, ia] = a_grid[iap]
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, iap]
                else:
                    y0 = lhs[iz, ik, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ik, iap] - rhs[iap, ia]
                    ap_endo[iz, ik, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * (
                            Wb[iz, 0, iap - 1] + (ap_endo[iz, ik, ia] - a_grid[iap - 1]) *
                            (Wb[iz, 0, iap] - Wb[iz, 0, iap - 1]) / (a_grid[iap] - a_grid[iap - 1]))
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step6(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, k, a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + b_grid[0] - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

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
    chi = Psi_fun(ss['a'], a_grid, r, chi0, chi1, chi2)
    Chi = np.vdot(ss['D'], chi)
    goods_mkt = ss['C'] + I + G + Chi + omega * ss['B'] - 1
    assert np.abs(goods_mkt) < 1E-7

    ss.update({'pi': 0, 'piw': 0, 'Q': 1, 'Y': 1, 'N': 1, 'mc': mc, 'K': K, 'Z': Z, 'I': I, 'w': w, 'tax': tax,
               'div': div, 'p': p, 'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi, 'goods_mkt': goods_mkt, 'chi': chi, 'phi': phi,
               'beta': beta, 'vphi': vphi, 'omega': omega, 'alpha': alpha, 'delta': delta, 'mup': mup, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z_grid, 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'pshare': pshare, 'rstar': r, 'i': r})
    return ss
