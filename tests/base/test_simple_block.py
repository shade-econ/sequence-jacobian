"""Test SimpleBlock functionality"""

from sequence_jacobian import simple, utilities
import numpy as np
import pytest


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


@pytest.mark.parametrize("block,ss", [(F, (1, 1, 1, 0.5)),
                                      (investment, (1, 1, 0.05, 1, 1, 1, 0.05, 2, 0.5)),
                                      (taylor, (0.05, 0.01, 1.5))])
def test_block_consistency(block, ss):
    """Make sure ss, td, and jac methods are all consistent with each other.
    Requires that all inputs of simple block allow calculating Jacobians"""
    # get ss output
    ss_results = block.ss(*ss)

    # now if we put in constant inputs, td should give us the same!
    ss = dict(zip(block.input_list, ss))
    td_results = block.td(ss, **{k: np.full(20, v) for k, v in ss.items()})
    for k, v in td_results.items():
        assert np.all(v == ss_results[k])

    # now get the Jacobian
    J = block.jac(ss, shock_list=block.input_list)

    # now perturb the steady state by small random vectors
    # and verify that the second-order numerical derivative implied by .td
    # is equivalent to what we get from jac

    h = 1E-5
    all_shocks = {i: np.random.rand(10) for i in block.input_list}
    td_up = block.td(ss, **{i: ss[i] + h*shock for i, shock in all_shocks.items()})
    td_dn = block.td(ss, **{i: ss[i] - h*shock for i, shock in all_shocks.items()})
    
    linear_impulses = {o: (td_up[o] - td_dn[o])/(2*h) for o in td_up}
    linear_impulses_from_jac = {o: sum(J[o][i] @ all_shocks[i] for i in all_shocks if i in J[o]) for o in td_up}

    for o in linear_impulses:
        assert np.all(np.abs(linear_impulses[o] - linear_impulses_from_jac[o]) < 1E-5)
