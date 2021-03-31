"""Test all models' non-linear transitional dynamics computations"""

import numpy as np
import copy

from sequence_jacobian import two_asset, combine, get_H_U
from sequence_jacobian import utilities as utils


# TODO: Figure out a more robust way to check similarity of the linear and non-linear solution.
#   As of now just checking that the tolerance for difference (by infinity norm) is below a manually checked threshold
def test_rbc_td(rbc_dag):
    rbc_model, exogenous, unknowns, targets, ss = rbc_dag

    T, impact, rho, news = 30, 0.01, 0.8, 10
    G = rbc_model.solve_jacobian(ss, exogenous, unknowns, targets, T=T)

    dZ = np.empty((T, 2))
    dZ[:, 0] = impact * ss['Z'] * rho**np.arange(T)
    dZ[:, 1] = np.concatenate((np.zeros(news), dZ[:-news, 0]))
    dC = 100 * G['C']['Z'] @ dZ / ss['C']

    td_nonlin = rbc_model.solve_impulse_nonlinear(ss, {"Z": dZ[:, 0]}, unknowns=unknowns, targets=targets)
    td_nonlin_news = rbc_model.solve_impulse_nonlinear(ss, {"Z": dZ[:, 1]}, unknowns=unknowns, targets=targets)

    dC_nonlin = 100 * (td_nonlin['C'] / ss['C'] - 1)
    dC_nonlin_news = 100 * (td_nonlin_news['C'] / ss['C'] - 1)

    assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 3e-2
    assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 7e-2


def test_ks_td(krusell_smith_dag):
    ks_model, exogenous, unknowns, targets, ss = krusell_smith_dag

    T = 30
    G = ks_model.solve_jacobian(ss, exogenous, unknowns, targets, T=T)

    for shock_size, tol in [(0.01, 7e-3), (0.1, 0.6)]:
        dZ = shock_size * 0.8 ** np.arange(T)

        td_nonlin = ks_model.solve_impulse_nonlinear(ss, {"Z": dZ}, unknowns=unknowns, targets=targets,
                                                     monotonic=True)
        dr_nonlin = 10000 * (td_nonlin['r'] - ss['r'])
        dr_lin = 10000 * G['r']['Z'] @ dZ

        assert np.linalg.norm(dr_nonlin - dr_lin, np.inf) < tol


def test_hank_td(one_asset_hank_dag):
    hank_model, exogenous, unknowns, targets, ss = one_asset_hank_dag

    T = 30
    G = hank_model.solve_jacobian(ss, exogenous, unknowns, targets, T=T, save=True)

    rho_r, sig_r = 0.61, -0.01/4
    drstar = sig_r * rho_r ** (np.arange(T))

    H_U = get_H_U(hank_model.blocks, unknowns, targets, T, ss, use_saved=True)
    H_U_factored = utils.misc.factor(H_U)

    td_nonlin = hank_model.solve_impulse_nonlinear(ss, {"rstar": drstar}, unknowns, targets,
                                                   H_U_factored=H_U_factored)

    dC_nonlin = 100 * (td_nonlin['C'] / ss['C'] - 1)
    dC_lin = 100 * G['C']['rstar'] @ drstar / ss['C']

    assert np.linalg.norm(dC_nonlin - dC_lin, np.inf) < 3e-3


def test_two_asset_td(two_asset_hank_dag):
    two_asset_model, exogenous, unknowns, targets, ss = two_asset_hank_dag

    T = 30
    G = two_asset_model.solve_jacobian(ss, exogenous, unknowns, targets, T=T, save=True)

    for shock_size, tol in [(0.1, 3e-4), (1, 2e-2)]:
        drstar = shock_size * -0.0025 * 0.6 ** np.arange(T)

        td_nonlin = two_asset_model.solve_impulse_nonlinear(ss, {"rstar": drstar}, unknowns, targets, use_saved=True)

        dY_nonlin = 100 * (td_nonlin['Y'] - 1)
        dY_lin = 100 * G['Y']['rstar'] @ drstar

        assert np.linalg.norm(dY_nonlin - dY_lin, np.inf) < tol


def test_two_asset_solved_v_simple_td(two_asset_hank_dag):
    two_asset_model, exogenous, unknowns, targets, ss = two_asset_hank_dag

    household = copy.deepcopy(two_asset.household)
    household.add_hetoutput(two_asset.adjustment_costs, verbose=False)
    blocks_simple = [household, two_asset.make_grids,
                     two_asset.pricing, two_asset.arbitrage, two_asset.labor, two_asset.investment,
                     two_asset.dividend, two_asset.taylor, two_asset.fiscal,
                     two_asset.finance, two_asset.wage, two_asset.union, two_asset.mkt_clearing]
    two_asset_model_simple = combine(blocks_simple, name="Two-Asset HANK w/ SimpleBlocks")
    unknowns_simple = ["r", "w", "Y", "pi", "p", "Q", "K"]
    targets_simple = ["asset_mkt", "fisher", "wnkpc", "nkpc", "equity", "inv", "val"]

    T = 30
    G = two_asset_model.solve_jacobian(ss, exogenous, unknowns, targets, T=T, save=True)
    G_simple = two_asset_model_simple.solve_jacobian(ss, exogenous, unknowns_simple, targets_simple, T=T, save=True)

    drstar = -0.0025 * 0.6 ** np.arange(T)

    dY = 100 * G['Y']['rstar'] @ drstar
    td_nonlin = two_asset_model.solve_impulse_nonlinear(ss, {"rstar": drstar}, unknowns, targets, use_saved=True)
    dY_nonlin = 100 * (td_nonlin['Y'] - 1)

    dY_simple = 100 * G_simple['Y']['rstar'] @ drstar
    td_nonlin_simple = two_asset_model_simple.solve_impulse_nonlinear(ss, {"rstar": drstar}, unknowns_simple,
                                                                      targets_simple, use_saved=True)

    dY_nonlin_simple = 100 * (td_nonlin_simple['Y'] - 1)

    assert np.linalg.norm(dY_nonlin - dY_nonlin_simple, np.inf) < 2e-7
    assert np.linalg.norm(dY - dY_simple, np.inf) < 0.02