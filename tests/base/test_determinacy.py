"""Test all models' determinacy calculations"""
import pytest

from sequence_jacobian import jacobian, determinacy


# TODO: This warning occurs because the arguments determining the number of grid points in constructing a grid
#   using np.linspace must be an `int`, whereas Ignore wraps floats. Eventually this will be subsumed in code that
#   is more custom-built to handle grid instantiation, which will likely live within the hetinput of a HetBlock.
#   For now we will just suppress the deprecation, since once this is implemented it will not matter so much whether
#   input arguments, which are `int`s, are promoted to floats.
@pytest.mark.filterwarnings("ignore:.*cannot be safely interpreted as an integer.*:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:.*cannot be safely interpreted as an integer.*:DeprecationWarning")
def test_two_asset_determinacy(two_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = two_asset_hank_model
    T = 100

    A = jacobian.get_H_U(blocks, unknowns, targets, T, ss, asymptotic=True)
    wn = determinacy.winding_criterion(A)
    assert wn == 0