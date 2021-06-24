"""Couple household model with frictional labor supply."""

import numpy as np
from numba import guvectorize, njit

from sequence_jacobian import simple, agrid, markov_rouwenhorst, create_model, solved
from sequence_jacobian.blocks.discont_block import discont
from sequence_jacobian.utilities.misc import choice_prob, logsum


'''Core HA block'''


def household_init(a_grid, z_grid_m, b_grid_m, z_grid_f, b_grid_f,
                   atw, transfer, rpost, eis, vphi_m, chi_m, vphi_f, chi_f):
    _, income = labor_income(z_grid_m, b_grid_m, z_grid_f, b_grid_f, atw, transfer, 1, 1, 0, 0, 0, 0, 0, 0)
    coh = income[:, :, np.newaxis] + (1 + rpost) * a_grid[np.newaxis, np.newaxis, :]
    c_guess = 0.3 * coh
    V, Va = np.empty_like(coh), np.empty_like(coh)
    for iw in range(16):
        V[iw, ...] = util(c_guess[iw, ...], iw, eis, vphi_m, chi_m, vphi_f, chi_f)
    V = V / 0.1

    # get Va by finite difference
    Va[:, :, 1:-1] = (V[:, :, 2:] - V[:, :, :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[:, :, 0] = (V[:, :, 1] - V[:, :, 0]) / (a_grid[1] - a_grid[0])
    Va[:, :, -1] = (V[:, :, -1] - V[:, :, -2]) / (a_grid[-1] - a_grid[-2])

    return Va, V


@njit(fastmath=True)
def util(c, iw, eis, vphi_m, chi_m, vphi_f, chi_f):
    """Utility function."""
    # 1. utility from consumption.
    if eis == 1:
        u = np.log(c / 2)
    else:
        u = (c/2) ** (1 - 1/eis) / (1 - 1/eis)

    # 2. disutility from work and search
    if iw <= 3:
        u = u - vphi_m                                # E male
    if (iw >= 4) & (iw <= 11):
        u = u - chi_m                                 # U male
    if np.mod(iw, 4) == 0:
        u = u - vphi_f                                # E female
    if (np.mod(iw, 4) == 1) | (np.mod(iw, 4) == 2):
        u = u - chi_f                                 # U female
    return u


@discont(exogenous=('Pi_s', 'Pi_z'), policy='a', disc_policy='P', backward=('V', 'Va'), backward_init=household_init)
def household(V_p, Va_p, Pi_z_p, Pi_s_p, choice_set, a_grid, y_grid, lam_grid,
              z_all, b_all, z_man, b_man, z_wom, b_wom, eis, beta, rpost, vphi_m, chi_m, vphi_f, chi_f):
    """
    Backward step function EGM with upper envelope.

    Dimensions: 0: labor market status, 1: productivity, 2: assets.
    Status: 0: EE, 1: EUb, 2: EU, 3: EN, 4: UbE, 5: UbUb, 6: UbU, 7: UbN, 8: UE, 9: UUb, 10: UU, 11: UN,
            12: NE, 13: NUb, 14: NU, 15: NN
    State: 0: MM, 1: MB, 2: ML, 3: BM, 4: BB, 5: BL, 6: LM, 7: LB, 8: LL

    Parameters
    ----------
    V_p         : array(Ns, Nz, Na), status-specific value function tomorrow
    Va_p        : array(Ns, Nz, Na), partial of status-specific value function tomorrow
    Pi_s_p      : array(Ns, Nx), Markov matrix for labor market shocks
    Pi_z_p      : array(Nz, Nz), (non-status-specific Markov) matrix for productivity
    choice_set  : list(Nz), discrete choices available in each state X
    a_grid      : array(Na), exogenous asset grid
    y_grid      : array(Ns, Nz), exogenous labor income grid
    lam_grid    : array(Nx), scale of taste shocks, specific to interim state
    eis         : float, EIS
    beta        : float, discount factor
    rpost           : float, ex-post interest rate

    Returns
    -------
    V           : array(Ns, Nz, Na), status-specific value function today
    Va          : array(Ns, Nz, Na), partial of status-specific value function today
    P           : array(Nx, Ns, Nz, Na), probability of choosing status s in state x
    c           : array(Ns, Nz, Na), status-specific consumption policy today
    a           : array(Ns, Nz, Na), status-specific asset policy today
    ze          : array(Ns, Nz, Na), effective labor (average productivity if employed)
    ui          : array(Ns, Nz, Na), UI benefit claims (average productivity if unemployed)
    """
    # shapes
    Ns, Nz, Na = V_p.shape
    Nx = Pi_s_p.shape[1]

    # PART 1: update value and policy functions
    # a. discrete choice I expect to make tomorrow
    V_p_X = np.empty((Nx, Nz, Na))
    Va_p_X = np.empty((Nx, Nz, Na))
    for ix in range(Nx):
        V_p_ix = np.take(V_p, indices=choice_set[ix], axis=0)
        Va_p_ix = np.take(Va_p, indices=choice_set[ix], axis=0)
        P_p_ix = choice_prob(V_p_ix, lam_grid[ix])
        V_p_X[ix, ...] = logsum(V_p_ix, lam_grid[ix])
        Va_p_X[ix, ...] = np.sum(P_p_ix*Va_p_ix, axis=0)

    # b. compute expectation wrt labor market shock
    V_p1 = np.einsum('ij,jkl->ikl', Pi_s_p, V_p_X)
    Va_p1 = np.einsum('ij,jkl->ikl', Pi_s_p, Va_p_X)

    # b. compute expectation wrt productivity
    V_p2 = np.einsum('ij,kjl->kil', Pi_z_p, V_p1)
    Va_p2 = np.einsum('ij,kjl->kil', Pi_z_p, Va_p1)

    # d. consumption today on tomorrow's grid and endogenous asset grid today
    W = beta * V_p2
    uc_nextgrid = beta * Va_p2
    c_nextgrid = 2 ** (1 - eis) * uc_nextgrid ** (-eis)
    a_nextgrid = (c_nextgrid + a_grid[np.newaxis, np.newaxis, :] - y_grid[:, :, np.newaxis]) / (1 + rpost)

    # e. upper envelope
    imin, imax = nonconcave(uc_nextgrid)  # bounds of non-concave region
    V, c = upper_envelope(imin, imax, W, a_nextgrid, c_nextgrid, a_grid, y_grid, rpost, eis, vphi_m, chi_m,
                          vphi_f, chi_f)

    # f. update Va
    uc = 2 ** (1 / eis - 1) * c ** (-1 / eis)
    Va = (1 + rpost) * uc

    # PART 2: things we need for GE
    # 2/a. asset policy
    a = (1 + rpost) * a_grid[np.newaxis, np.newaxis, :] + y_grid[:, :, np.newaxis] - c

    # 2/b. choice probabilities (don't need jacobian)
    P = np.zeros((Nx, Ns, Nz, Na))
    for ix in range(Nx):
        V_ix = np.take(V, indices=choice_set[ix], axis=0)
        P[ix, choice_set[ix], ...] = choice_prob(V_ix, lam_grid[ix])

    # 2/c. average productivity of employed
    ze = np.zeros_like(a)
    ze[0, ...] = z_all[:, np.newaxis]
    ze[1, ...], ze[2, ...], ze[3, ...] = z_man[:, np.newaxis], z_man[:, np.newaxis], z_man[:, np.newaxis]
    ze[4, ...], ze[8, ...], ze[12, ...] = z_wom[:, np.newaxis], z_wom[:, np.newaxis], z_wom[:, np.newaxis]

    # 2/d. UI claims
    ui = np.zeros_like(a)
    ui[5, ...] = b_all[:, np.newaxis]
    ui[4, ...], ui[6, ...], ui[7, ...] = b_man[:, np.newaxis], b_man[:, np.newaxis], b_man[:, np.newaxis]
    ui[1, ...], ui[9, ...], ui[13, ...] = b_wom[:, np.newaxis], b_wom[:, np.newaxis], b_wom[:, np.newaxis]

    return V, Va, a, c, P, ze, ui


def labor_income(z_grid_m, b_grid_m, z_grid_f, b_grid_f, atw, transfer,
                 fU_m, fN_m, s_m, fU_f, fN_f, s_f, rho, expiry):
    # 1. income
    yE_m, yE_f = atw * z_grid_m + transfer, atw * z_grid_f
    yU_m, yU_f = atw * b_grid_m + transfer, atw * b_grid_f
    yN_m, yN_f = np.zeros_like(yE_m) + transfer, np.zeros_like(yE_f)
    y_EE = (yE_m[:, np.newaxis] + yE_f[np.newaxis, :]).ravel()
    y_EU = (yE_m[:, np.newaxis] + yU_f[np.newaxis, :]).ravel()
    y_EN = (yE_m[:, np.newaxis] + yN_f[np.newaxis, :]).ravel()
    y_UE = (yU_m[:, np.newaxis] + yE_f[np.newaxis, :]).ravel()
    y_UU = (yU_m[:, np.newaxis] + yU_f[np.newaxis, :]).ravel()
    y_UN = (yU_m[:, np.newaxis] + yN_f[np.newaxis, :]).ravel()
    y_NE = (yN_m[:, np.newaxis] + yE_f[np.newaxis, :]).ravel()
    y_NU = (yN_m[:, np.newaxis] + yU_f[np.newaxis, :]).ravel()
    y_NN = (yN_m[:, np.newaxis] + yN_f[np.newaxis, :]).ravel()
    y_grid = np.vstack((y_EE, y_EU, y_EN, y_EN, y_UE, y_UU, y_UN, y_UN, y_NE, y_NU, y_NN, y_NN, y_NE, y_NU, y_NN, y_NN))

    # 2. transition matrix for joint labor market status
    cov = rho * np.sqrt(s_m * s_f * (1 - s_m) * (1 - s_f))
    Pi_s_m = np.array([[1 - s_m, s_m, 0], [fU_m, (1 - fU_m) * (1 - expiry), (1 - fU_m) * expiry],
                       [fU_m, 0, 1 - fU_m], [fN_m, 0, 1 - fN_m]])
    Pi_s_f = np.array([[1 - s_f, s_f, 0], [fU_f, (1 - fU_f) * (1 - expiry), (1 - fU_f) * expiry],
                       [fU_f, 0, 1 - fU_f], [fN_f, 0, 1 - fN_f]])
    Pi_s = np.kron(Pi_s_m, Pi_s_f)

    # adjust for correlated job loss
    Pi_s[0, 0] += cov  # neither loses their job
    Pi_s[0, 4] += cov  # both lose their job
    Pi_s[0, 1] -= cov  # only female loses her job
    Pi_s[0, 3] -= cov  # only male loses his job

    return Pi_s, y_grid


"""Supporting functions for HA block"""


@guvectorize(['void(float64[:], uint32[:], uint32[:])'], '(nA) -> (),()', nopython=True)
def nonconcave(uc_nextgrid, imin, imax):
    """Obtain bounds for non-concave region."""
    nA = uc_nextgrid.shape[-1]
    vmin = np.inf
    vmax = -np.inf
    # step 1: find vmin & vmax
    for ia in range(nA - 1):
        if uc_nextgrid[ia + 1] > uc_nextgrid[ia]:
            vmin_temp = uc_nextgrid[ia]
            vmax_temp = uc_nextgrid[ia + 1]
            if vmin_temp < vmin:
                vmin = vmin_temp
            if vmax_temp > vmax:
                vmax = vmax_temp

    # 2/a Find imin (upper bound)
    if vmin == np.inf:
        imin_ = 0
    else:
        ia = 0
        while ia < nA:
            if uc_nextgrid[ia] < vmin:
                break
            ia += 1
        imin_ = ia

    # 2/b Find imax (lower bound)
    if vmax == -np.inf:
        imax_ = nA
    else:
        ia = nA
        while ia > 0:
            if uc_nextgrid[ia] > vmax:
                break
            ia -= 1
        imax_ = ia

    imin[:] = imin_
    imax[:] = imax_


@njit
def upper_envelope(imin, imax, W, a_nextgrid, c_nextgrid, a_grid, y_grid, rpost, *args):
    """
    Interpolate consumption and value function to exogenous grid. Brute force but safe.
    Parameters
    ----------
    W           : array(Ns, Nz, Na), status-specific end-of-period value function (on tomorrow's grid)
    a_nextgrid  : array(Ns, Nz, Na), endogenous asset grid (today's grid)
    c_nextgrid  : array(Ns, Nz, Na), consumption on endogenous grid (today's grid)
    a_grid      : array(Na), exogenous asset grid (tomorrow's grid)
    y_grid      : array(Ns, Nz), labor income
    rpost       : float, ex-post interest rate
    args        : (eis, vphi, chi) arguments for utility function
    Returns
    -------
    V  : array(Ns, Nz, Na), status-specific value function on exogenous grid
    c  : array(Ns, Nz, Na), consumption on exogenous grid
    """

    # 0. initialize
    Ns, Nz, Na = W.shape
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # outer loop could run in parallel
    for iw in range(Ns):
        for iz in range(Nz):
            ycur = y_grid[iw, iz]
            imaxcur = imax[iw, iz]
            imincur = imin[iw, iz]

            # 1. unconstrained case: loop through a_grid, find bracketing endogenous gridpoints and interpolate.
            # in concave region: exploit monotonicity and don't look for extra solutions
            for ia in range(Na):
                acur = a_grid[ia]

                # Region 1: below non-concave: exploit monotonicity
                if (ia <= imaxcur) | (ia >= imincur):
                    iap = 0
                    ap_low = a_nextgrid[iw, iz, iap]
                    ap_high = a_nextgrid[iw, iz, iap + 1]
                    while iap < Na - 2:  # can this go up all the way?
                        if ap_high >= acur:
                            break
                        iap += 1
                        ap_low = ap_high
                        ap_high = a_nextgrid[iw, iz, iap + 1]
                    # found bracket, interpolate value function and consumption
                    w_low, w_high = W[iw, iz, iap], W[iw, iz, iap + 1]
                    c_low, c_high = c_nextgrid[iw, iz, iap], c_nextgrid[iw, iz, iap + 1]
                    w_slope = (w_high - w_low) / (ap_high - ap_low)
                    c_slope = (c_high - c_low) / (ap_high - ap_low)
                    c_guess = c_low + c_slope * (acur - ap_low)
                    w_guess = w_low + w_slope * (acur - ap_low)
                    V[iw, iz, ia] = util(c_guess, iw, *args) + w_guess
                    c[iw, iz, ia] = c_guess

                # Region 2: non-concave region
                else:
                    # try out all segments of endogenous grid
                    for iap in range(Na - 1):
                        # does this endogenous segment bracket ia?
                        ap_low, ap_high = a_nextgrid[iw, iz, iap], a_nextgrid[iw, iz, iap + 1]
                        interp = (ap_low <= acur <= ap_high) or (ap_low >= acur >= ap_high)

                        # does it need to be extrapolated above the endogenous grid?
                        # if needed to be extrapolated below, we would be in constrained case
                        extrap_above = (iap == Na - 2) and (acur > a_nextgrid[iw, iz, Na - 1])

                        if interp or extrap_above:
                            # interpolation slopes
                            w_low, w_high = W[iw, iz, iap], W[iw, iz, iap + 1]
                            c_low, c_high = c_nextgrid[iw, iz, iap], c_nextgrid[iw, iz, iap + 1]
                            w_slope = (w_high - w_low) / (ap_high - ap_low)
                            c_slope = (c_high - c_low) / (ap_high - ap_low)

                            # implied guess
                            c_guess = c_low + c_slope * (acur - ap_low)
                            w_guess = w_low + w_slope * (acur - ap_low)

                            # value
                            v_guess = util(c_guess, iw, *args) + w_guess

                            # select best value for this segment
                            if v_guess > V[iw, iz, ia]:
                                V[iw, iz, ia] = v_guess
                                c[iw, iz, ia] = c_guess

            # 2. constrained case: remember that we have the inverse asset policy a(a')
            ia = 0
            while ia < Na and a_grid[ia] <= a_nextgrid[iw, iz, 0]:
                c[iw, iz, ia] = (1 + rpost) * a_grid[ia] + ycur
                V[iw, iz, ia] = util(c[iw, iz, ia], iw, *args) + W[iw, iz, 0]
                ia += 1

    return V, c


'''Simple blocks'''


def zgrids_couple(zm, zf):
    """Combine individual z_grid and b_grid as needed for couple level."""
    zeroes = np.zeros_like(zm)

    # all combinations
    z_all = (zm[:, np.newaxis] + zf[np.newaxis, :]).ravel()
    z_men = (zm[:, np.newaxis] + zeroes[np.newaxis, :]).ravel()
    z_wom = (zeroes[:, np.newaxis] + zf[np.newaxis, :]).ravel()
    return z_all, z_men, z_wom


@simple
def income_state_vars(mean_m, rho_m, sd_m, mean_f, rho_f, sd_f, nZ, uirate, uicap):
    # income husband
    z_grid_m, z_pdf_m, z_markov_m = markov_rouwenhorst(rho=rho_m, sigma=sd_m, N=nZ)
    z_grid_m *= mean_m
    b_grid_m = uirate * z_grid_m
    b_grid_m[b_grid_m > uicap] = uicap

    # income wife
    z_grid_f, z_pdf_f, z_markov_f = markov_rouwenhorst(rho=rho_f, sigma=sd_f, N=nZ)
    z_grid_f *= mean_f
    b_grid_f = uirate * z_grid_f
    b_grid_f[b_grid_f > uicap] = uicap

    # household income
    z_all, z_man, z_wom = zgrids_couple(z_grid_m, z_grid_f)
    b_all, b_man, b_wom = zgrids_couple(b_grid_m, b_grid_f)
    Pi_z = np.kron(z_markov_m, z_markov_f)

    return z_grid_m, z_grid_f, b_grid_m, b_grid_f, z_all, z_man, z_wom, b_all, b_man, b_wom, Pi_z


@simple
def employment_state_vars(lamM, lamB, lamL):
    choice_set = [[0, 3, 12, 15], [1, 3, 13, 15], [2, 3, 14, 15], [4, 7, 12, 15], [5, 7, 13, 15],
                  [6, 7, 14, 15], [8, 11, 12, 15], [9, 11, 13, 15], [10, 11, 14, 15]]
    lam_grid = np.array([np.sqrt(2*lamM**2), np.sqrt(lamM**2 + lamB**2), np.sqrt(lamM**2 + lamL**2),
                         np.sqrt(lamB**2 + lamM**2), np.sqrt(2*lamB**2), np.sqrt(lamB**2 + lamL**2),
                         np.sqrt(lamL**2 + lamM**2), np.sqrt(lamL**2 + lamB**2), np.sqrt(2*lamL**2)])
    return choice_set, lam_grid


@simple
def asset_state_vars(amin, amax, nA):
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return a_grid


@solved(unknowns={'fU_m': 0.25, 'fN_m': 0.1, 's_m': 0.025},
        targets=['fU_m_res', 'fN_m_res', 's_m_res'],
        solver='broyden_custom')
def flows_m(Y, fU_m, fN_m, s_m, fU_eps_m, fN_eps_m, s_eps_m):
    fU_m_res = fU_m.ss * (Y / Y.ss) ** fU_eps_m - fU_m
    fN_m_res = fN_m.ss * (Y / Y.ss) ** fN_eps_m - fN_m
    s_m_res = s_m.ss * (Y / Y.ss) ** s_eps_m - s_m
    return fU_m_res, fN_m_res, s_m_res


@solved(unknowns={'fU_f': 0.25, 'fN_f': 0.1, 's_f': 0.025},
        targets=['fU_f_res', 'fN_f_res', 's_f_res'],
        solver='broyden_custom')
def flows_f(Y, fU_f, fN_f, s_f, fU_eps_f, fN_eps_f, s_eps_f):
    fU_f_res = fU_f.ss * (Y / Y.ss) ** fU_eps_f - fU_f
    fN_f_res = fN_f.ss * (Y / Y.ss) ** fN_eps_f - fN_f
    s_f_res = s_f.ss * (Y / Y.ss) ** s_eps_f - s_f
    return fU_f_res, fN_f_res, s_f_res


'''Put it together'''

household.add_hetinput(labor_income, verbose=False)
hh = create_model([income_state_vars, employment_state_vars, asset_state_vars, flows_m, flows_f, household],
                  name='CoupleHH')
