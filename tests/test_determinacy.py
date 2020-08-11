"""Test all models' determinacy calculations"""

import numpy as np

from sequence_jacobian import jacobian, determinacy


def test_hank_determinacy(one_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = one_asset_hank_model
    T = 100

    # Stable Case
    A = jacobian.get_H_U(blocks, unknowns, targets, T, ss, asymptotic=True, save=True)
    wn = determinacy.winding_criterion(A)
    assert wn == 0

    # Unstable Case
    ss_unstable = {**ss, "phi": 0.75}
    A_unstable = jacobian.get_H_U(blocks, unknowns, targets, T, ss_unstable, asymptotic=True, use_saved=True)
    wn_unstable = determinacy.winding_criterion(A_unstable)
    assert wn_unstable == -1


def test_two_asset_determinacy(two_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = two_asset_hank_model
    T = 100

    A = jacobian.get_H_U(blocks, unknowns, targets, T, ss, asymptotic=True)
    wn = determinacy.winding_criterion(A)
    assert wn == 0