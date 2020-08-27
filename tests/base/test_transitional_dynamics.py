"""Test all models' non-linear transitional dynamics computations"""

import numpy as np
import copy

from sequence_jacobian import two_asset, nonlinear, jacobian
from sequence_jacobian import utilities as utils


# TODO: Figure out a more robust way to check similarity of the linear and non-linear solution.
#   As of now just checking that the tolerance for difference (by infinity norm) is below a manually checked threshold

def test_rbc_td(rbc_model):
    blocks, exogenous, unknowns, targets, ss = rbc_model

    T, impact, rho, news = 30, 0.01, 0.8, 10
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss)

    dZ = np.empty((T, 2))
    dZ[:, 0] = impact * ss['Z'] * rho**np.arange(T)
    dZ[:, 1] = np.concatenate((np.zeros(news), dZ[:-news, 0]))
    dC = 100 * G['C']['Z'] @ dZ / ss['C']

    td_nonlin = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns, targets=targets,
                                   Z=ss["Z"]+dZ[:, 0], verbose=False)
    td_nonlin_news = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns, targets=targets,
                                        Z=ss["Z"]+dZ[:, 1], verbose=False)
    dC_nonlin = 100 * (td_nonlin['C'] / ss['C'] - 1)
    dC_nonlin_news = 100 * (td_nonlin_news['C'] / ss['C'] - 1)

    assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 3e-2
    assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 7e-2


def test_ks_td(krusell_smith_model):
    blocks, exogenous, unknowns, targets, ss = krusell_smith_model

    T = 30
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss)

    for shock_size, tol in [(0.01, 7e-3), (0.1, 0.6)]:
        Z = ss['Z'] + shock_size * 0.8 ** np.arange(T)

        td_nonlin = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns,
                                       targets=targets, monotonic=True, Z=Z, verbose=False)
        dr_nonlin = 10000 * (td_nonlin['r'] - ss['r'])
        dr_lin = 10000 * G['r']['Z'] @ (Z - ss['Z'])

        assert np.linalg.norm(dr_nonlin - dr_lin, np.inf) < tol


def test_hank_td(one_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = one_asset_hank_model

    T = 30
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss, save=True)

    rho_r, sig_r = 0.61, -0.01/4
    drstar = sig_r * rho_r ** (np.arange(T))
    rstar = ss['r'] + drstar

    H_U = jacobian.get_H_U(blocks, unknowns, targets, T, ss, use_saved=True)
    H_U_factored = utils.misc.factor(H_U)

    td_nonlin = nonlinear.td_solve(ss, blocks, unknowns, targets, H_U_factored=H_U_factored, rstar=rstar, verbose=False)

    dC_nonlin = 100 * (td_nonlin['C'] / ss['C'] - 1)
    dC_lin = 100 * G['C']['rstar'] @ drstar / ss['C']

    assert np.linalg.norm(dC_nonlin - dC_lin, np.inf) < 3e-3


def test_two_asset_td(two_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = two_asset_hank_model

    T = 30
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss, save=True)

    for shock_size, tol in [(0.1, 3e-4), (1, 2e-2)]:
        drstar = -0.0025 * 0.6 ** np.arange(T)
        rstar = ss["r"] + shock_size * drstar

        td_nonlin = nonlinear.td_solve(ss, blocks, unknowns, targets, rstar=rstar, use_saved=True, verbose=False)

        dY_nonlin = 100 * (td_nonlin['Y'] - 1)
        dY_lin = shock_size * 100 * G['Y']['rstar'] @ drstar

        assert np.linalg.norm(dY_nonlin - dY_lin, np.inf) < tol


def test_two_asset_solved_v_simple_td(two_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = two_asset_hank_model

    household = copy.deepcopy(two_asset.household)
    household.add_hetoutput(two_asset.adjustment_costs, verbose=False)
    blocks_simple = [household, two_asset.make_grids,
                     two_asset.pricing, two_asset.arbitrage, two_asset.labor, two_asset.investment,
                     two_asset.dividend, two_asset.taylor, two_asset.fiscal,
                     two_asset.finance, two_asset.wage, two_asset.union, two_asset.mkt_clearing,
                     two_asset.partial_steady_state_solution]
    unknowns_simple = ["r", "w", "Y", "pi", "p", "Q", "K"]
    targets_simple = ["asset_mkt", "fisher", "wnkpc", "nkpc", "equity", "inv", "val"]

    T = 30
    G = jacobian.get_G(blocks, exogenous, unknowns, targets, T, ss=ss, save=True)
    G_simple = jacobian.get_G(blocks_simple, exogenous, unknowns_simple, targets_simple, T, ss=ss, save=True)

    drstar = -0.0025 * 0.6 ** np.arange(T)

    dY = 100 * G['Y']['rstar'] @ drstar
    td_nonlin = nonlinear.td_solve(ss, blocks, unknowns, targets,
                                   rstar=ss['r']+drstar, use_saved=True, verbose=False)
    dY_nonlin = 100 * (td_nonlin['Y'] - 1)

    dY_simple = 100 * G_simple['Y']['rstar'] @ drstar
    td_nonlin_simple = nonlinear.td_solve(ss, blocks_simple, unknowns_simple, targets_simple,
                                          rstar=ss['r']+drstar, use_saved=True, verbose=False)

    dY_nonlin_simple = 100 * (td_nonlin_simple['Y'] - 1)

    assert np.linalg.norm(dY_nonlin - dY_nonlin_simple, np.inf) < 2e-7
    assert np.linalg.norm(dY - dY_simple, np.inf) < 0.02