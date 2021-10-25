import numpy as np
import pytest
from sequence_jacobian.examples import krusell_smith

def test_jacobian_h(krusell_smith_dag):
    _, ss, dag, *_ = krusell_smith_dag
    hh = dag['household']

    lowacc = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=0.05)
    midacc = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=1E-3)
    usual = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=1E-4)
    nooption = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10)

    assert np.array_equal(usual['C','r'], nooption['C','r'])
    assert np.linalg.norm(usual['C','r'] - midacc['C','r']) < np.linalg.norm(usual['C','r'] - lowacc['C','r'])

    midacc_alt = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, options={'household': {'h': 1E-3}})
    assert np.array_equal(midacc['C', 'r'], midacc_alt['C', 'r'])

    lowacc = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 0.05}})
    midacc = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 1E-3}})
    usual = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 1E-4}})

    assert np.linalg.norm(usual['C','K'] - midacc['C','K']) < np.linalg.norm(usual['C','K'] - lowacc['C','K'])


def test_jacobian_steady_state(krusell_smith_dag):
    dag = krusell_smith_dag[2]
    calibration = {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho": 0.966, "sigma": 0.5,
                   "L": 1.0, "nS": 2, "nA": 10, "amax": 200, "r": 0.01, 'beta': 0.96,
                   "Z": 0.85, "K": 3.}

    pytest.raises(ValueError, dag.steady_state, calibration, options={'household': {'backward_maxit': 10}})

    ss1 = dag.steady_state(calibration)
    ss2 = dag.steady_state(calibration, options={'household': {'backward_maxit': 100000}})
    assert ss1['A'] == ss2['A']


def test_steady_state_solution(krusell_smith_dag):
    dag_ss, ss, *_ = krusell_smith_dag

    calibration = {'eis': 1.0, 'delta': 0.025, 'alpha': 0.11, 'rho': 0.966, 'sigma': 0.5,
                   'Y': 1.0, 'L': 1.0, 'nS': 2, 'nA': 10, 'amax': 200, 'r': 0.01}
    unknowns_ss = {'beta': (0.98 / 1.01, 0.999 / 1.01)}
    targets_ss = {'asset_mkt': 0.}

    ss2 = dag_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="brentq",
                                    ttol=1E-2, ctol=1E-2)
    assert not np.isclose(ss['asset_mkt'], ss2['asset_mkt'])
