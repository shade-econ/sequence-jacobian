import numpy as np
import scipy.linalg as linalg
from numba import njit


def all_covariances(dX, sigmas, T):
    """Use Fast Fourier Transform to compute covariance function between O vars up to T lags

    Parameters
    ----------
    dX     : array (T*O*Z), stacked impulse responses of nO variables to nZ shocks
    sigmas : array (Z), standard deviations of shocks
    T      : int, truncation horizon

    Returns
    ----------
    Sigma : array (T*O*O), covariance function between O variables for 0, ..., T lags
    """
    dft = np.fft.rfftn(dX, s=(2 * T - 2,), axes=(0,))
    total = (dft.conjugate() * sigmas) @ dft.swapaxes(1, 2)
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]


@njit
def build_full_covariance_matrix(Sigma, sigmas_measure, tau):
    """Takes in T*O*O array Sigma with covariances at each lag t,
    assembles them into (tau*O)*(tau*O) matrix of covariances, including measurement errors.
    """
    T, O, O = Sigma.shape
    V = np.empty((tau, O, tau, O))
    for t1 in range(tau):
        for t2 in range(tau):
            if abs(t1-t2) >= T:
                V[t1, :, t2, :] = np.zeros((O, O))
            else:
                if t1 < t2:
                    V[t1, : , t2, :] = Sigma[t2-t1, :, :]
                elif t1 > t2:
                    V[t1, : , t2, :] = Sigma[t1-t2, :, :].T
                else:
                    # want exactly symmetric
                    V[t1, :, t2, :] = (np.diag(sigmas_measure**2) + (Sigma[0, :, :]+Sigma[0, :, :].T)/2)
    return V.reshape((tau*O, tau*O))


def log_likelihood(V, w):
    """Implements multivariate normal log-likelihood formula for observations w and variance V"""
    V_factored = linalg.cho_factor(V)
    quadratic_form = np.dot(w, linalg.cho_solve(V_factored, w))
    log_determinant = 2*np.sum(np.log(np.diag(V_factored[0])))
    return -(log_determinant + quadratic_form) / 2
