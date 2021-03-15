"""Test public-facing classes"""

import numpy as np
from sequence_jacobian import create_model
from sequence_jacobian.blocks.support.impulse import ImpulseDict


def test_impulsedict(krusell_smith_model):
    blocks, exogenous, unknowns, targets, ss = krusell_smith_model
    household, firm, mkt_clearing, _, _, _ = blocks
    T = 200

    # Linearized impulse responses as deviations, nonlinear as levels
    ks = create_model(*blocks, name='KS')
    ir_lin = ks.solve_impulse_linear(ss, {'Z': 0.01 * 0.5**np.arange(T)}, unknowns, targets)
    ir_nonlin = ks.solve_impulse_nonlinear(ss, {'Z': 0.01 * 0.5 ** np.arange(T)}, unknowns, targets)

    # Get method
    assert isinstance(ir_lin, ImpulseDict)
    assert isinstance(ir_lin[['C']], ImpulseDict)
    assert isinstance(ir_lin['C'], np.ndarray)

    # Merge method
    temp = ir_lin[['C', 'K']] | ir_lin[['r']]
    assert list(temp.impulse.keys()) == ['C', 'K', 'r']

    # Normalize and scalar multiplication
    dC1 = 100 * ir_lin['C'] / ss['C']
    dC2 = 100 * ir_lin[['C']].normalize()
    assert np.allclose(dC1, dC2['C'])

    # Levels and deviations
    assert np.linalg.norm(ir_nonlin.deviations()['C'] - ir_lin['C']) < 1E-4
    assert np.linalg.norm(ir_nonlin['C'] - ir_lin.levels()['C']) < 1E-4
