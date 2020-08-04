"""Test all models' non-linear transitional dynamics computations"""

import numpy as np

from sequence_jacobian import nonlinear, jacobian
from sequence_jacobian.models import rbc, krusell_smith, hank, two_asset


def test_rbc_td(rbc_model):
    blocks, exogenous, unknowns, targets, ss = rbc_model

    T, impact, rho, news = 100, 0.01, 0.8, 10
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss)

    dZ = np.empty((T, 2))
    dZ[:, 0] = impact * ss['Z'] * rho**np.arange(T)
    dZ[:, 1] = np.concatenate((np.zeros(news), dZ[:-news, 0]))
    dC = 100 * G['C']['Z'] @ dZ / ss['C']

    td_nonlin = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns, targets=targets,
                                   Z=ss["Z"]+dZ[:, 0], noisy=False)
    td_nonlin_news = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns, targets=targets,
                                        Z=ss["Z"]+dZ[:, 1], noisy=False)
    dC_nonlin = 100 * (td_nonlin['C'] / ss['C'] - 1)
    dC_nonlin_news = 100 * (td_nonlin_news['C'] / ss['C'] - 1)

    assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 1e-3
    assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 1e-3


from sequence_jacobian import steady_state, utils
# def test_ks_jac(krusell_smith_model):
# blocks, exogenous, unknowns, targets, ss = krusell_smith_model
blocks = [krusell_smith.household, krusell_smith.firm, krusell_smith.mkt_clearing, krusell_smith.income_state_vars,
          krusell_smith.asset_state_vars, krusell_smith.firm_steady_state_solution]

# Steady State
calibration = {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho": 0.966, "sigma": 0.5, "L": 1.0,
               "nS": 2, "nA": 10, "amax": 200, "r": 0.01}
ss_unknowns = {"beta": (0.98/1.01, 0.999/1.01)}
ss_targets = {"K": "A"}
ss = steady_state(blocks, calibration, ss_unknowns, ss_targets,
                  solver="brentq", consistency_check=True, full_output=True)

# Transitional Dynamics/Jacobian Calculation
exogenous = ["Z"]
unknowns = ["K"]
targets = ["asset_mkt"]

household, firm, mkt_clearing, _, _, _ = blocks
# blocks = [household, firm, mkt_clearing]

T = 300
G2 = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                    targets=targets, T=T, ss=ss)

# G2 = jacobian.get_G(block_list=[firm, mkt_clearing, household], exogenous=exogenous, unknowns=unknowns,
#                     targets=targets, T=T, ss=ss)

J_firm = firm.jac(ss, shock_list=['K', 'Z'])
J_ha = household.jac(ss, T=T, shock_list=['r', 'w'])
J_curlyK_K = J_ha['A']['r'] @ J_firm['r']['K'] + J_ha['A']['w'] @ J_firm['w']['K']
J_curlyK_Z = J_ha['A']['r'] @ J_firm['r']['Z'] + J_ha['A']['w'] @ J_firm['w']['Z']
J = {**J_firm, 'curlyK': {'K': J_curlyK_K, 'Z': J_curlyK_Z}}
H_K = J['curlyK']['K'] - np.eye(T)
H_Z = J['curlyK']['Z']
G = {'K': -np.linalg.solve(H_K, H_Z)}  # H_K^(-1)H_Z
G['r'] = J['r']['Z'] + J['r']['K'] @ G['K']
G['w'] = J['w']['Z'] + J['w']['K'] @ G['K']
G['Y'] = J['Y']['Z'] + J['Y']['K'] @ G['K']
G['C'] = J_ha['C']['r'] @ G['r'] + J_ha['C']['w'] @ G['w']

for o in G:
    assert np.allclose(G2[o]['Z'], G[o])

# def test_ks_td(krusell_smith_model):
#     blocks, exogenous, unknowns, targets, ss = krusell_smith_model
#
#     T, impact, rho, news = 300, 0.01, 0.8, 10
#     G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
#                        targets=targets, T=T, ss=ss)
#
#     Z = ss['Z'] + 0.01 * 0.8 ** np.arange(T)
#
#     td_nonlin = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns,
#                                    targets=targets, monotonic=True, Z=Z)
#     dr_nonlin = 10000 * (td_nonlin['r'] - ss['r'])
#     dr_lin = 10000 * G['r'] @ (Z - ss['Z'])
#
#     # assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 1e-3
#     # assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 1e-3
