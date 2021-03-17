import pytest
import numpy as np


# Filter out warnings when the solver is trying to search in bad regions
@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*:RuntimeWarning")
def test_hank_steady_state_w_bad_init_guesses_and_bounds(one_asset_hank_dag):
    hank_model, _, _, _, ss = one_asset_hank_dag

    calibration = {"r": 0.005, "rstar": 0.005, "eis": 0.5, "frisch": 0.5, "mu": 1.2, "B_Y": 5.6,
                   "rho_s": 0.966, "sigma_s": 0.5, "kappa": 0.1, "phi": 1.5, "Y": 1, "Z": 1, "L": 1,
                   "pi": 0, "nS": 2, "amax": 150, "nA": 10}
    unknowns = {"beta": (0.95, 0.97, 0.999/(1 + 0.005)), "vphi": (0.001, 1.0, 10)}
    targets = {"asset_mkt": 0, "labor_mkt": 0}
    ss_ref = hank_model.solve_steady_state(calibration, unknowns, targets,
                                           solver="broyden1", solver_kwargs={"options": {"maxiter": 250}},
                                           constrained_kwargs={"boundary_epsilon": 5e-3, "penalty_scale": 100})

    for k in ss.keys():
        assert np.all(np.isclose(ss[k], ss_ref[k]))


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*:RuntimeWarning")
def test_two_asset_steady_state_w_bad_init_guesses_and_bounds(two_asset_hank_dag):
    two_asset_model, _, _, _, ss = two_asset_hank_dag

    calibration = {"pi": 0, "piw": 0, "Q": 1, "Y": 1, "N": 1, "r": 0.0125, "rstar": 0.0125, "i": 0.0125,
                   "tot_wealth": 14, "K": 10, "delta": 0.02, "kappap": 0.1, "muw": 1.1, "Bh": 1.04,
                   "Bg": 2.8, "G": 0.2, "eis": 0.5, "frisch": 1, "chi0": 0.25, "chi2": 2, "epsI": 4,
                   "omega": 0.005, "kappaw": 0.1, "phi": 1.5, "nZ": 3, "nB": 10, "nA": 16, "nK": 4,
                   "bmax": 50, "amax": 4000, "kmax": 1, "rho_z": 0.966, "sigma_z": 0.92}
    unknowns = {"beta": (0.5, 0.9, 0.999 / (1 + 0.0125)), "vphi": (0.001, 1.0, 10.), "chi1": (0.5, 5.5, 10.)}
    targets = {"asset_mkt": 0, "labor_mkt": 0, "B": "Bh"}
    ss_ref = two_asset_model.solve_steady_state(calibration, unknowns, targets, solver="broyden_custom",
                                                consistency_check=True)
    for k in ss.keys():
        assert np.all(np.isclose(ss[k], ss_ref[k]))
