"""Test all models' non-linear transitional dynamics computations"""

import numpy as np

from sequence_jacobian import combine
from sequence_jacobian.examples import two_asset
from sequence_jacobian.examples.hetblocks import household_twoasset as hh


# TODO: Figure out a more robust way to check similarity of the linear and non-linear solution.
#   As of now just checking that the tolerance for difference (by infinity norm) is below a manually checked threshold
def test_rbc_td(rbc_dag):
    rbc_model, ss, unknowns, targets, exogenous = rbc_dag

    T, impact, rho, news = 30, 0.01, 0.8, 10
    G = rbc_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

    dZ = np.empty((T, 2))
    dZ[:, 0] = impact * ss['Z'] * rho**np.arange(T)
    dZ[:, 1] = np.concatenate((np.zeros(news), dZ[:-news, 0]))
    dC = 100 * G['C']['Z'] @ dZ / ss['C']

    td_nonlin = rbc_model.solve_impulse_nonlinear(ss, unknowns, targets, inputs={"Z": dZ[:, 0]}, outputs=['C'])
    td_nonlin_news = rbc_model.solve_impulse_nonlinear(ss, unknowns, targets, inputs={"Z": dZ[:, 1]}, outputs=['C'])

    dC_nonlin = 100 * td_nonlin['C'] / ss['C']
    dC_nonlin_news = 100 * td_nonlin_news['C'] / ss['C']

    assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 3e-2
    assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 7e-2


def test_ks_td(krusell_smith_dag):
    ks_model, ss, unknowns, targets, exogenous = krusell_smith_dag

    T = 30
    G = ks_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

    for shock_size, tol in [(0.01, 7e-3), (0.1, 0.6)]:
        dZ = shock_size * 0.8 ** np.arange(T)

        td_nonlin = ks_model.solve_impulse_nonlinear(ss, unknowns, targets, {"Z": dZ})
        dr_nonlin = 10000 * td_nonlin['r']
        dr_lin = 10000 * G['r']['Z'] @ dZ

        assert np.linalg.norm(dr_nonlin - dr_lin, np.inf) < tol


def test_hank_td(one_asset_hank_dag):
    hank_model, ss, unknowns, targets, exogenous = one_asset_hank_dag

    T = 30
    household = hank_model['household']
    J_ha = household.jacobian(ss=ss, T=T, inputs=['Div', 'Tax', 'r', 'w'])
    G = hank_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T, Js={'household': J_ha})

    rho_r, sig_r = 0.61, -0.01/4
    drstar = sig_r * rho_r ** (np.arange(T))

    td_nonlin = hank_model.solve_impulse_nonlinear(ss, unknowns, targets, {"rstar": drstar}, Js={'household': J_ha})

    dC_nonlin = 100 * td_nonlin['C'] / ss['C']
    dC_lin = 100 * G['C']['rstar'] @ drstar / ss['C']

    assert np.linalg.norm(dC_nonlin - dC_lin, np.inf) < 3e-3


# TODO: needs to compute Jacobian of hetoutput `Chi`
def test_two_asset_td(two_asset_hank_dag):
    two_asset_model, ss, unknowns, targets, exogenous = two_asset_hank_dag

    T = 30
    household = two_asset_model['household']
    J_ha = household.jacobian(ss=ss, T=T, inputs=['N', 'r', 'ra', 'rb', 'tax', 'w'])
    G = two_asset_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T, Js={'household': J_ha})

    for shock_size, tol in [(0.1, 3e-4), (1, 2e-2)]:
        drstar = shock_size * -0.0025 * 0.6 ** np.arange(T)

        td_nonlin = two_asset_model.solve_impulse_nonlinear(ss, unknowns, targets, {"rstar": drstar},
                                                            Js={'household': J_ha})

        dY_nonlin = 100 * td_nonlin['Y']
        dY_lin = 100 * G['Y']['rstar'] @ drstar

        assert np.linalg.norm(dY_nonlin - dY_lin, np.inf) < tol


def test_two_asset_solved_v_simple_td(two_asset_hank_dag):
    two_asset_model, ss, unknowns, targets, exogenous = two_asset_hank_dag

    household = hh.household.add_hetinputs([two_asset.income, two_asset.make_grids])
    blocks_simple = [household, two_asset.pricing, two_asset.arbitrage,
                     two_asset.labor, two_asset.investment, two_asset.dividend,
                     two_asset.taylor, two_asset.fiscal, two_asset.share_value,
                     two_asset.finance, two_asset.wage, two_asset.union,
                     two_asset.mkt_clearing]
    two_asset_model_simple = combine(blocks_simple, name="Two-Asset HANK w/ SimpleBlocks")
    unknowns_simple = ["r", "w", "Y", "pi", "p", "Q", "K"]
    targets_simple = ["asset_mkt", "fisher", "wnkpc", "nkpc", "equity", "inv", "val"]

    T = 30
    household = two_asset_model['household']
    J_ha = household.jacobian(ss=ss, T=T, inputs=['N', 'r', 'ra', 'rb', 'tax', 'w'])
    G = two_asset_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T, Js={'household': J_ha})
    G_simple = two_asset_model_simple.solve_jacobian(ss, unknowns_simple, targets_simple, exogenous, T=T,
                                                     Js={'household': J_ha})

    drstar = -0.0025 * 0.6 ** np.arange(T)

    dY = 100 * G['Y']['rstar'] @ drstar
    td_nonlin = two_asset_model.solve_impulse_nonlinear(ss, unknowns, targets, {"rstar": drstar},
                                                        Js={'household': J_ha})
    dY_nonlin = 100 * (td_nonlin['Y'] - 1)

    dY_simple = 100 * G_simple['Y']['rstar'] @ drstar
    td_nonlin_simple = two_asset_model_simple.solve_impulse_nonlinear(ss,
                                                                      unknowns_simple, targets_simple,
                                                                      {"rstar": drstar}, Js={'household': J_ha})

    dY_nonlin_simple = 100 * (td_nonlin_simple['Y'] - 1)

    assert np.linalg.norm(dY_nonlin - dY_nonlin_simple, np.inf) < 2e-7
    assert np.linalg.norm(dY - dY_simple, np.inf) < 0.02
