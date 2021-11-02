'''
SIM model with labor force participation choice
- state space: (s, x, z, a)
    - s is employment
        - 0: employed, 1: unemployed 
    - x is matching
        - 0: matched, 1: unmatched
    - z is labor productivity
    - a is assets
'''
import numpy as np
from numba import njit, guvectorize

from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker, LogitChoice
from sequence_jacobian import markov_rouwenhorst, agrid


'''Setup: utility function, hetinputs, initializer'''


@njit(fastmath=True)
def util(c, emp_status, eis, vphi):
    # utility from consumption
    if eis == 1:
        u = np.log(c)
    else:
        u = c ** (1 - 1 / eis) / (1 - 1 / eis)

    # disutility from work
    if emp_status == 0:
        u = u - vphi

    return u


def make_grids(rho_z, sd_z, nZ, amin, amax, nA):
    z_grid, z_dist, Pi_z = markov_rouwenhorst(rho=rho_z, sigma=sd_z, N=nZ)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return z_grid, z_dist, Pi_z, a_grid


def labor_income(z_grid, atw, b, s, f):
    y_grid = z_grid[np.newaxis, :] * np.array([atw, b])[:, np.newaxis]
    Pi_s = np.array([[1 - s, s], [f, (1 - f)]])
    return y_grid, Pi_s


def backward_init(y_grid, a_grid, r, eis, vphi):
    # consumption rule of thumb
    coh = y_grid[:, :, np.newaxis] + (1 + r) * a_grid[np.newaxis, np.newaxis, :]
    c = 0.1 * coh
    
    # capitalize utility to get value function
    V, Va = np.empty_like(c), np.empty_like(c)
    for e in range(2):
        V[e, ...] = util(c[e, ...], e, eis, vphi) / 0.01

    # get Va by finite difference
    Va[:, :, 1:-1] = (V[:, :, 2:] - V[:, :, :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[:, :, 0] = (V[:, :, 1] - V[:, :, 0]) / (a_grid[1] - a_grid[0])
    Va[:, :, -1] = (V[:, :, -1] - V[:, :, -2]) / (a_grid[-1] - a_grid[-2])

    return V, Va 


'''Consumption-savings stage: : (s, z, a) -> (s, z, a')'''


def consav(V, Va, a_grid, y_grid, r, beta, eis, vphi):
    """DC-EGM algorithm"""
    # EGM step
    W = beta * V
    uc_nextgrid = beta * Va
    c_nextgrid = uc_nextgrid ** (-eis)
    a_nextgrid = (c_nextgrid + a_grid[np.newaxis, np.newaxis, :] - y_grid[:, :, np.newaxis]) / (1 + r)

    # upper envelope step
    imin, imax = nonconcave(uc_nextgrid)  # bounds of non-concave region
    V, c = upper_envelope(imin, imax, W, a_nextgrid, c_nextgrid, a_grid, y_grid, r, eis, vphi)

    # update Va, report asset policy
    uc = c ** (-1 / eis)
    Va = (1 + r) * uc
    a = (1 + r) * a_grid[np.newaxis, np.newaxis, :] + y_grid[:, :, np.newaxis] - c

    return V, Va, a, c


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
def upper_envelope(imin, imax, W, a_nextgrid, c_nextgrid, a_grid, y_grid, r, eis, vphi):
    """Interpolate value function and consumption to exogenous grid."""
    Ns, Nz, Na = W.shape
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    for ie in range(Ns):
        for iz in range(Nz):
            ycur = y_grid[ie, iz]
            imaxcur = imax[ie, iz]
            imincur = imin[ie, iz]

            # UNCONSTRAINED CASE: loop through a_grid, find bracketing endogenous gridpoints and interpolate.
            # in concave region: exploit monotonicity and don't look for extra solutions
            for ia in range(Na):
                acur = a_grid[ia]

                # Region 1: below non-concave: exploit monotonicity
                if (ia <= imaxcur) | (ia >= imincur):
                    iap = 0
                    ap_low = a_nextgrid[ie, iz, iap]
                    ap_high = a_nextgrid[ie, iz, iap + 1]
                    while iap < Na - 2:
                        if ap_high >= acur:
                            break
                        iap += 1
                        ap_low = ap_high
                        ap_high = a_nextgrid[ie, iz, iap + 1]
                    # found bracket, interpolate value function and consumption
                    w_low, w_high = W[ie, iz, iap], W[ie, iz, iap + 1]
                    c_low, c_high = c_nextgrid[ie, iz, iap], c_nextgrid[ie, iz, iap + 1]
                    w_slope = (w_high - w_low) / (ap_high - ap_low)
                    c_slope = (c_high - c_low) / (ap_high - ap_low)
                    c_guess = c_low + c_slope * (acur - ap_low)
                    w_guess = w_low + w_slope * (acur - ap_low)
                    V[ie, iz, ia] = util(c_guess, ie, eis, vphi) + w_guess
                    c[ie, iz, ia] = c_guess

                # Region 2: non-concave region
                else:
                    # try out all segments of endogenous grid
                    for iap in range(Na - 1):
                        # does this endogenous segment bracket ia?
                        ap_low, ap_high = a_nextgrid[ie, iz, iap], a_nextgrid[ie, iz, iap + 1]
                        interp = (ap_low <= acur <= ap_high) or (ap_low >= acur >= ap_high)

                        # does it need to be extrapolated above the endogenous grid?
                        # if needed to be extrapolated below, we would be in constrained case
                        extrap_above = (iap == Na - 2) and (acur > a_nextgrid[ie, iz, Na - 1])

                        if interp or extrap_above:
                            # interpolate value function and consumption
                            w_low, w_high = W[ie, iz, iap], W[ie, iz, iap + 1]
                            c_low, c_high = c_nextgrid[ie, iz, iap], c_nextgrid[ie, iz, iap + 1]
                            w_slope = (w_high - w_low) / (ap_high - ap_low)
                            c_slope = (c_high - c_low) / (ap_high - ap_low)
                            c_guess = c_low + c_slope * (acur - ap_low)
                            w_guess = w_low + w_slope * (acur - ap_low)
                            v_guess = util(c_guess, ie, eis, vphi) + w_guess

                            # select best value for this segment
                            if v_guess > V[ie, iz, ia]:
                                V[ie, iz, ia] = v_guess
                                c[ie, iz, ia] = c_guess

            # CONSTRAINED CASE: remember that we have the inverse asset policy a(a')
            ia = 0
            while ia < Na and a_grid[ia] <= a_nextgrid[ie, iz, 0]:
                c[ie, iz, ia] = (1 + r) * a_grid[ia] + ycur
                V[ie, iz, ia] = util(c[ie, iz, ia], ie, eis, vphi) + W[ie, iz, 0]
                ia += 1

    return V, c


'''Logit choice stage: (x, z, a) -> (s, z, a)'''


def participation(V):
    '''adjustments to flow utility associated with x -> s choice, implements constraints on discrete choice'''
    flow_u = np.zeros((2,) + V.shape) # (s, x, z, a)
    flow_u[0, 1, ...] = -np.inf       # unmatched -> employed
    return flow_u


'''Put stages together'''

consav_stage = Continuous1D(backward='Va', policy='a', f=consav, name='consav')
labsup_stage = LogitChoice(value='V', backward='Va', index=0, taste_shock_scale='taste_shock',
                           f=participation, name='dchoice')
search_stage = ExogenousMaker(markov_name='Pi_s', index=0, name='search_shock')
prod_stage = ExogenousMaker(markov_name='Pi_z', index=1, name='prod_shock')

hh = StageBlock([prod_stage, search_stage, labsup_stage, consav_stage],
                backward_init=backward_init, hetinputs=[make_grids, labor_income], name='household')

