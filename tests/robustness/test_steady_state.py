"""Tests for steady_state with worse initial guesses, making use of the constrained solution functionality"""

import pytest
import numpy as np

from sequence_jacobian.models import hank, two_asset


# Filter out warnings when the solver is trying to search in bad regions
@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*:RuntimeWarning")
def test_hank_steady_state_w_bad_init_guesses_and_bounds(one_asset_hank_dag):
    hank_model, _, _, _, ss = one_asset_hank_dag

    helper_blocks = [hank.partial_steady_state_solution]

    calibration = {"r": 0.005, "rstar": 0.005, "eis": 0.5, "frisch": 0.5, "B_Y": 5.6, "mu": 1.2,
                   "rho_s": 0.966, "sigma_s": 0.5, "kappa": 0.1, "phi": 1.5, "Y": 1, "Z": 1, "L": 1,
                   "pi": 0, "nS": 2, "amax": 150, "nA": 10}
    unknowns_ss = {"beta": (0.95, 0.97, 0.999 / (1 + 0.005)), "vphi": (0.001, 1.0, 10.), "w": 0.8}
    targets_ss = {"asset_mkt": 0, "labor_mkt": 0, "nkpc_res": 0.}
    ss_ref = hank_model.solve_steady_state(calibration, unknowns_ss, targets_ss,
                                           helper_blocks=helper_blocks, helper_targets=["nkpc_res"], solver="hybr",
                                           constrained_kwargs={"boundary_epsilon": 5e-3, "penalty_scale": 100})

    for k in ss.keys():
        assert np.all(np.isclose(ss[k], ss_ref[k]))


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*:RuntimeWarning")
def test_two_asset_steady_state_w_bad_init_guesses_and_bounds(two_asset_hank_dag):
    two_asset_model, _, _, _, ss = two_asset_hank_dag

    helper_blocks = [two_asset.partial_ss_step1, two_asset.partial_ss_step2]

    # Steady State
    calibration = {"Y": 1., "r": 0.0125, "rstar": 0.0125, "tot_wealth": 14, "delta": 0.02, "kappap": 0.1, "muw": 1.1,
                   "Bh": 1.04, "Bg": 2.8, "G": 0.2, "eis": 0.5, "frisch": 1, "chi0": 0.25, "chi2": 2,
                   "epsI": 4, "omega": 0.005, "kappaw": 0.1, "phi": 1.5, "nZ": 3, "nB": 10, "nA": 16,
                   "nK": 4, "bmax": 50, "amax": 4000, "kmax": 1, "rho_z": 0.966, "sigma_z": 0.92}
    unknowns_ss = {"beta": 0.976, "chi1": 6.5, "vphi": 1.71, "Z": 0.4678, "alpha": 0.3299, "mup": 1.015, 'w': 0.66}
    targets_ss = {"asset_mkt": 0., "B": "Bh", 'wnkpc': 0., 'pi': 0.0, "K": 10., "wealth": "tot_wealth", "N": 1.0}
    ss_ref = two_asset_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom",
                                                helper_blocks=helper_blocks,
                                                helper_targets={'wnkpc': 0., 'pi': 0.0, "K": 10.,
                                                                "wealth": "tot_wealth", "N": 1.0})
    for k in ss.keys():
        assert np.all(np.isclose(ss[k], ss_ref[k]))
