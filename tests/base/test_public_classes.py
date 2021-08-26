"""Test public-facing classes"""

import numpy as np
import pytest

from sequence_jacobian import het
from sequence_jacobian.steady_state.classes import SteadyStateDict
from sequence_jacobian.blocks.support.impulse import ImpulseDict
from sequence_jacobian.blocks.support.bijection import Bijection


def test_impulsedict(krusell_smith_dag):
    ks_model, exogenous, unknowns, targets, ss = krusell_smith_dag
    T = 200

    # Linearized impulse responses as deviations
    ir_lin = ks_model.solve_impulse_linear(ss, unknowns, targets, inputs={'Z': 0.01 * 0.5**np.arange(T)}, outputs=['C', 'K', 'r'])

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


def test_bijection():
    # generate and invert
    mymap = Bijection({'a': 'a1', 'b': 'b1'})
    mymapinv = mymap.inv
    assert mymap['a'] == 'a1' and mymap['b'] == 'b1'
    assert mymapinv['a1'] == 'a' and mymapinv['b1'] == 'b'

    # duplicate keys rejected
    with pytest.raises(ValueError):
        Bijection({'a': 'a1', 'b': 'a1'})

    # composition with another bijection (flows backwards)
    mymap2 = Bijection({'a1': 'a2'})
    assert (mymap2 @ mymap)['a'] == 'a2'

    # composition with SteadyStateDict
    ss = SteadyStateDict({'a': 2.0, 'b': 1.0}, internal={})
    ss_remapped = ss @ mymap
    assert isinstance(ss_remapped, SteadyStateDict)
    assert ss_remapped['a1'] == ss['a'] and ss_remapped['b1'] == ss['b']
