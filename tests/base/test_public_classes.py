"""Test public-facing classes"""

import numpy as np

from sequence_jacobian import het
from sequence_jacobian.steady_state.classes import SteadyStateDict
from sequence_jacobian.blocks.support.impulse import ImpulseDict


def test_steadystatedict():
    toplevel = {"A": 1., "B": 2.}
    internal = {"block1": {"a": np.array([0, 1]), "D": np.array([[0, 0.5], [0.5, 0]]),
                           "Pi": np.array([[0.5, 0.5], [0.5, 0.5]])}}
    raw_output = {"A": 1., "B": 2., "a": np.array([0, 1]), "D": np.array([[0, 0.5], [0.5, 0]]),
                  "Pi": np.array([[0.5, 0.5], [0.5, 0.5]])}

    @het(exogenous="Pi", policy="a", backward="Va")
    def block1(Va_p, Pi_p, t1, t2):
        Va = Va_p
        a = t1 + t2
        return Va, a

    ss1 = SteadyStateDict(toplevel, internal=internal)
    ss2 = SteadyStateDict(raw_output, internal=block1)

    # Test that both ways of instantiating SteadyStateDict given by ss1 and ss2 are equivalent
    def check_steady_states(ss1, ss2):
        assert set(ss1.keys()) == set(ss2.keys())
        for k in ss1:
            assert np.isclose(ss1[k], ss2[k])

        assert set(ss1.internal.keys()) == set(ss2.internal.keys())
        for k in ss1.internal:
            assert set(ss1.internal[k].keys()) == set(ss2.internal[k].keys())
            for kk in ss1.internal[k]:
                assert np.all(np.isclose(ss1.internal[k][kk], ss2.internal[k][kk]))

    check_steady_states(ss1, ss2)

    # Test iterable indexing
    assert ss1[["A", "B"]] == {"A": 1., "B": 2.}

    # Test updating
    toplevel_new = {"C": 2., "D": 4.}
    internal_new = {"block1_new": {"a": np.array([2, 0]), "D": np.array([[0.25, 0.25], [0.25, 0.25]]),
                    "Pi": np.array([[0.2, 0.8], [0.8, 0.2]])}}
    ss_new = SteadyStateDict(toplevel_new, internal=internal_new)

    ss1.update(ss_new)
    ss2.update(toplevel_new, internal_namespaces=internal_new)

    check_steady_states(ss1, ss2)


def test_impulsedict(krusell_smith_dag):
    ks_model, exogenous, unknowns, targets, ss = krusell_smith_dag
    T = 200

    # Linearized impulse responses as deviations
    ir_lin = ks_model.solve_impulse_linear(ss, {'Z': 0.01 * 0.5**np.arange(T)}, unknowns, targets)

    # Get method
    assert isinstance(ir_lin, ImpulseDict)
    assert isinstance(ir_lin[['C']], ImpulseDict)
    assert isinstance(ir_lin['C'], np.ndarray)

    # Merge method
    temp = ir_lin[['C', 'K']] | ir_lin[['r']]
    assert list(temp.impulse.keys()) == ['C', 'K', 'r']

    # SS and scalar multiplication
    dC1 = 100 * ir_lin['C'] / ss['C']
    dC2 = 100 * ir_lin[['C']] / ss
    assert np.allclose(dC1, dC2['C'])
