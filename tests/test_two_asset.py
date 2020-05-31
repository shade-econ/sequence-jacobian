import utils
import numpy as np
import two_asset


def test_hank_ss():
    A, B, U = hank_ss_singlerun()
    assert np.isclose(A, 12.526539492650361)
    assert np.isclose(B, 1.0840860793350566)
    assert np.isclose(U, 4.5102870939550055)


def hank_ss_singlerun(beta=0.976, vphi=2.07, r=0.0125, tot_wealth=14, K=10, delta=0.02, kappap=0.1, 
            muw=1.1, Bh=1.04, Bg=2.8, G=0.2, eis=0.5, frisch=1, chi0=0.25, chi1=6.5, chi2=2,
            epsI=4, omega=0.005, kappaw=0.1, phi=1.5, nZ=3, nB=50, nA=70, nK=50,
            bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92, noisy=True):
    """Mostly cribbed from two_asset.hank_ss(), but just does backward iteration to get
    a partial equilibrium household steady state given parameters, not solving for equilibrium.
    Convenient for testing."""

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)
    k_grid = utils.agrid(amax=kmax, n=nK)
    e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    # solve analytically what we can
    I = delta * K
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    w = (1 - alpha) * mc
    tax = (r * Bg + G) / w
    ra = r
    rb = r - omega

    # figure out initializer
    z_grid = two_asset.income(e_grid, tax, w, 1)
    Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))

    out = two_asset.household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, 
                           N=1, tax=tax, w=w, e_grid=e_grid, k_grid=k_grid, beta=beta,
                           eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2)
    
    return out['A'], out['B'], out['U']
