"""Test SimpleBlock functionality"""
import copy

import numpy as np
import pytest

from sequence_jacobian import simple
from sequence_jacobian.steady_state.classes import SteadyStateDict


@simple
def F(K, L, Z, alpha):
    Y = Z * K(-1)**alpha * L**(1-alpha)
    FK = alpha * Y / K
    FL = (1-alpha) * Y / L
    return Y, FK, FL


@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K/K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1-alpha) * mc(+1) - (K(+1)/K -
           (1-delta) + (K(+1)/K - 1)**2 / (2*delta*epsI)) + K(+1)/K*Q(+1) - (1 + r(+1))*Q
    return inv, val


@simple
def taylor(r, pi, phi):
    i = r.ss + phi * (pi - pi.ss)
    return i


@pytest.mark.parametrize("block,ss", [(F, SteadyStateDict({"K": 1, "L": 1, "Z": 1, "alpha": 0.5})),
                                      (investment, SteadyStateDict({"Q": 1, "K": 1, "r": 0.05, "N": 1, "mc": 1,
                                                                    "Z": 1, "delta": 0.05, "epsI": 2, "alpha": 0.5})),
                                      (taylor, SteadyStateDict({"r": 0.05, "pi": 0.01, "phi": 1.5}))])
def test_block_consistency(block, ss):
    """Make sure ss, td, and jac methods are all consistent with each other.
    Requires that all inputs of simple block allow calculating Jacobians"""
    # get ss output
    ss_results = block.steady_state(ss)

    # now if we put in constant inputs, td should give us the same!
    td_results = block.impulse_nonlinear(ss_results, exogenous={k: np.zeros(20) for k in ss.keys()})
    for k, v in td_results.impulse.items():
        assert np.all(v == 0)

    # now get the Jacobian
    J = block.jacobian(ss, inputs=block.inputs)

    # now perturb the steady state by small random vectors
    # and verify that the second-order numerical derivative implied by .td
    # is equivalent to what we get from jac

    h = 1E-5
    all_shocks = {i: np.random.rand(10) for i in block.inputs}
    td_up = block.impulse_nonlinear(ss_results, exogenous={i: h*shock for i, shock in all_shocks.items()})
    td_dn = block.impulse_nonlinear(ss_results, exogenous={i: -h*shock for i, shock in all_shocks.items()})
    
    linear_impulses = {o: (td_up.impulse[o] - td_dn.impulse[o])/(2*h) for o in td_up.impulse}
    linear_impulses_from_jac = {o: sum(J[o][i] @ all_shocks[i] for i in all_shocks if i in J[o]) for o in td_up.impulse}

    for o in linear_impulses:
        assert np.all(np.abs(linear_impulses[o] - linear_impulses_from_jac[o]) < 1E-5)
