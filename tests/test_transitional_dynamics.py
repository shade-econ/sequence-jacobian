"""Test all models' non-linear transitional dynamics computations"""

import numpy as np

from sequence_jacobian import nonlinear, jacobian
from sequence_jacobian.models import rbc, krusell_smith, hank, two_asset


def test_rbc_td(rbc_model):
    T, impact, rho, news = 100, 0.01, 0.8, 10
    blocks, exogenous, unknowns, targets, ss = rbc_model
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


def test_ks_td(ks_model):
    T, impact, rho, news = 100, 0.01, 0.8, 10
    blocks, exogenous, unknowns, targets, ss = ks_model
    G = jacobian.get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
                       targets=targets, T=T, ss=ss)

    Z = ss['Z'] + 0.01 * 0.8 ** np.arange(T)

    td_nonlin = nonlinear.td_solve(ss=ss, block_list=blocks, unknowns=unknowns,
                                   targets=targets, monotonic=True, Z=Z)
    dr_nonlin = 10000 * (td_nonlin['r'] - ss['r'])
    dr_lin = 10000 * G['r'] @ (Z - ss['Z'])

    # assert np.linalg.norm(dC[:, 0] - dC_nonlin, np.inf) < 1e-3
    # assert np.linalg.norm(dC[:, 1] - dC_nonlin_news, np.inf) < 1e-3
