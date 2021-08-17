"""Test all models' estimation calculations"""

import pytest
import numpy as np

from sequence_jacobian import get_G, estimation


# See test_determinacy.py for the to-do describing this suppression
@pytest.mark.filterwarnings("ignore:.*cannot be safely interpreted as an integer.*:DeprecationWarning")
def test_krusell_smith_estimation(krusell_smith_dag):
    ks_model, exogenous, unknowns, targets, ss = krusell_smith_dag

    np.random.seed(41234)
    T = 50
    G = ks_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

    # Step 1: Stacked impulse responses
    rho = 0.9
    sigma_persist = 0.1
    sigma_trans = 0.2

    dZ1 = rho**(np.arange(T))
    dY1, dC1, dK1 = G['Y']['Z'] @ dZ1, G['C']['Z'] @ dZ1, G['K']['Z'] @ dZ1
    dX1 = np.stack([dZ1, dY1, dC1, dK1], axis=1)

    dZ2 = np.arange(T) == 0
    dY2, dC2, dK2 = G['Y']['Z'] @ dZ2, G['C']['Z'] @ dZ2, G['K']['Z'] @ dZ2
    dX2 = np.stack([dZ2, dY2, dC2, dK2], axis=1)

    dX = np.stack([dX1, dX2], axis=2)

    # Step 2: Obtain covariance at all leads and lags
    sigmas = np.array([sigma_persist, sigma_trans])
    Sigma = estimation.all_covariances(dX, sigmas)

    # Step 3: Log-likelihood calculation
    # random 100 observations
    Y = np.random.randn(100, 4)

    # 0.05 measurement error in each variable
    sigma_measurement = np.full(4, 0.05)

    # calculate log-likelihood
    ll = estimation.log_likelihood(Y, Sigma, sigma_measurement)
    assert np.isclose(ll, -59921.410111251025)