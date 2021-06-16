"""Test the two asset HANK steady state computation"""

import numpy as np

from sequence_jacobian.models import two_asset
from sequence_jacobian import utilities as utils


def test_hank_ss():
    A, B, U = hank_ss_singlerun()
    assert np.isclose(A, 12.526539492650361)
    assert np.isclose(B, 1.0840860793350566)
    assert np.isclose(U, 4.5102870939550055)


def hank_ss_singlerun(beta=0.976, r=0.0125, tot_wealth=14, K=10, delta=0.02, Bg=2.8, G=0.2, eis=0.5,
                      chi0=0.25, chi1=6.5, chi2=2, omega=0.005, nZ=3, nB=50, nA=70, nK=50,
                      bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92):
    """Mostly cribbed from two_asset.hank_ss(), but just does backward iteration to get
    a partial equilibrium household steady state given parameters, not solving for equilibrium.
    Convenient for testing."""

    # set up grid
    b_grid = utils.discretize.agrid(amax=bmax, n=nB)
    a_grid = utils.discretize.agrid(amax=amax, n=nA)
    k_grid = utils.discretize.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, pi, Pi = utils.discretize.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    # solve analytically what we can
    I = delta * K
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    w = (1 - alpha) * mc
    tax = (r * Bg + G) / w
    ra = r
    rb = r - omega

    # figure out initializer
    calibration = {'Pi': Pi, 'a_grid': a_grid, 'b_grid': b_grid, 'e_grid': e_grid, 'k_grid': k_grid,
                   'beta': beta, 'N': 1.0, 'tax': tax, 'w': w, 'eis': eis, 'rb': rb, 'ra': ra,
                   'chi0': chi0, 'chi1': chi1, 'chi2': chi2}

    out = two_asset.household.steady_state(calibration)
    
    return out['A'], out['B'], out['U']


def test_Psi():
    np.random.seed(41234)
    chi0, chi1, chi2 = 0.25, 6.5, 2.3
    ra = 0.05

    a = np.random.rand(50) + 1
    ap = np.random.rand(50) + 1

    oPsi, oPsi1, oPsi2 = two_asset.get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2)

    Psi = Psi_correct(ap, a, ra, chi0, chi1, chi2)
    assert np.allclose(oPsi, Psi)

    # compare two-sided numerical derivative to our analytical one
    # numerical doesn't work well at kink of "abs" function, so this would fail
    # for some seeds if chi2 was less than 2
    Psi1 = (Psi_correct(ap+1E-4, a, ra, chi0, chi1, chi2) -
            Psi_correct(ap-1E-4, a, ra, chi0, chi1, chi2)) / 2E-4
    assert np.allclose(oPsi1, Psi1)

    Psi2 = (Psi_correct(ap, a+1E-4, ra, chi0, chi1, chi2) -
            Psi_correct(ap, a-1E-4, ra, chi0, chi1, chi2)) / 2E-4
    assert np.allclose(oPsi2, Psi2)


def Psi_correct(ap, a, ra, chi0, chi1, chi2):
    """Original Psi function that we know is correct, once denominator has power
    chi2-1 rather than 1 (error in original code)"""
    return chi1 / chi2 * np.abs((ap - (1 + ra) * a)) ** chi2 / ((1 + ra) * a + chi0) ** (chi2 - 1)