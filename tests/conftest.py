"""Fixtures used by tests."""

import pytest

from sequence_jacobian import create_model
from sequence_jacobian.models import rbc, krusell_smith, hank, two_asset


@pytest.fixture(scope='session')
def rbc_dag():
    blocks = [rbc.household, rbc.mkt_clearing, rbc.firm]
    helper_blocks = [rbc.steady_state_solution]
    rbc_model = create_model(blocks, name="RBC")

    # Steady State
    calibration = {"eis": 1., "frisch": 1., "delta": 0.025, "alpha": 0.11, "L": 1.}
    unknowns_ss = {"vphi": 0.92, "beta": 1 / (1 + 0.01), "K": 2., "Z": 1.}
    targets_ss = {"goods_mkt": 0., "r": 0.01, "euler": 0., "Y": 1.}
    ss = rbc_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="solved",
                                      helper_blocks=helper_blocks,
                                      helper_targets=["goods_mkt", "r", "euler", "Y"])

    # Transitional Dynamics/Jacobian Calculation
    exogenous = ["Z"]
    unknowns = ["K", "L"]
    targets = ["goods_mkt", "euler"]

    return rbc_model, exogenous, unknowns, targets, ss


@pytest.fixture(scope='session')
def krusell_smith_dag():
    blocks = [krusell_smith.household, krusell_smith.firm, krusell_smith.mkt_clearing, krusell_smith.income_state_vars,
              krusell_smith.asset_state_vars]
    helper_blocks = [krusell_smith.firm_steady_state_solution]
    ks_model = create_model(blocks, name="Krusell-Smith")

    # Steady State
    calibration = {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho": 0.966, "sigma": 0.5, "L": 1.0,
                   "nS": 2, "nA": 10, "amax": 200, "r": 0.01}
    unknowns_ss = {"beta": (0.98 / 1.01, 0.999 / 1.01), "Z": 0.85, "K": 3.}
    targets_ss = {"asset_mkt": 0., "Y": 1., "r": 0.01}
    ss = ks_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="brentq",
                                     helper_blocks=helper_blocks, helper_targets=["Y", "r"])

    # Transitional Dynamics/Jacobian Calculation
    exogenous = ["Z"]
    unknowns = ["K"]
    targets = ["asset_mkt"]

    return ks_model, exogenous, unknowns, targets, ss


@pytest.fixture(scope='session')
def one_asset_hank_dag():
    blocks = [hank.household, hank.firm, hank.monetary, hank.fiscal, hank.mkt_clearing, hank.nkpc,
              hank.income_state_vars, hank.asset_state_vars]
    helper_blocks = [hank.partial_steady_state_solution]
    hank_model = create_model(blocks, name="One-Asset HANK")

    # Steady State
    calibration = {"r": 0.005, "rstar": 0.005, "eis": 0.5, "frisch": 0.5, "B_Y": 5.6, "mu": 1.2,
                   "rho_s": 0.966, "sigma_s": 0.5, "kappa": 0.1, "phi": 1.5, "Y": 1, "Z": 1, "L": 1,
                   "pi": 0, "nS": 2, "amax": 150, "nA": 10}
    unknowns_ss = {"beta": 0.986, "vphi": 0.8, "w": 0.8}
    targets_ss = {"asset_mkt": 0, "labor_mkt": 0, "nkpc_res": 0.}
    ss = hank_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom",
                                       helper_blocks=helper_blocks, helper_targets=["nkpc_res"])

    # Transitional Dynamics/Jacobian Calculation
    exogenous = ["rstar", "Z"]
    unknowns = ["pi", "w", "Y"]
    targets = ["nkpc_res", "asset_mkt", "labor_mkt"]

    return hank_model, exogenous, unknowns, targets, ss


@pytest.fixture(scope='session')
def two_asset_hank_dag():
    blocks = [two_asset.household, two_asset.make_grids,
              two_asset.pricing_solved, two_asset.arbitrage_solved, two_asset.production_solved,
              two_asset.dividend, two_asset.taylor, two_asset.fiscal, two_asset.share_value,
              two_asset.finance, two_asset.wage, two_asset.union, two_asset.mkt_clearing]
    helper_blocks = [two_asset.partial_ss_step1, two_asset.partial_ss_step2]
    two_asset_model = create_model(blocks, name="Two-Asset HANK")

    # Steady State
    calibration = {"Y": 1., "r": 0.0125, "rstar": 0.0125, "tot_wealth": 14, "delta": 0.02, "kappap": 0.1, "muw": 1.1,
                   "Bh": 1.04, "Bg": 2.8, "G": 0.2, "eis": 0.5, "frisch": 1, "chi0": 0.25, "chi2": 2,
                   "epsI": 4, "omega": 0.005, "kappaw": 0.1, "phi": 1.5, "nZ": 3, "nB": 10, "nA": 16,
                   "nK": 4, "bmax": 50, "amax": 4000, "kmax": 1, "rho_z": 0.966, "sigma_z": 0.92}
    unknowns_ss = {"beta": 0.976, "chi1": 6.5, "vphi": 1.71, "Z": 0.4678, "alpha": 0.3299, "mup": 1.015, 'w': 0.66}
    targets_ss = {"asset_mkt": 0., "B": "Bh", 'wnkpc': 0., 'piw': 0.0, "K": 10., "wealth": "tot_wealth", "N": 1.0}
    ss = two_asset_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="broyden_custom",
                                            helper_blocks=helper_blocks,
                                            helper_targets=["wnkpc", "piw", "K", "wealth", "N"])

    # Transitional Dynamics/Jacobian Calculation
    exogenous = ["rstar", "Z", "G"]
    unknowns = ["r", "w", "Y"]
    targets = ["asset_mkt", "fisher", "wnkpc"]

    return two_asset_model, exogenous, unknowns, targets, ss
