import numpy as np
import scipy.linalg as linalg
from numba import njit

'''Part 1: compute covariances at all lags and log likelihood'''


def all_covariances(M, sigmas):
    """Use Fast Fourier Transform to compute covariance function between O vars up to T-1 lags.

    See equation (108) in appendix B.5 of paper for details.

    Parameters
    ----------
    M      : array (T*O*Z), stacked impulse responses of nO variables to nZ shocks (MA(T-1) representation) 
    sigmas : array (Z), standard deviations of shocks

    Returns
    ----------
    Sigma : array (T*O*O), covariance function between O variables for 0, ..., T-1 lags
    """
    T = M.shape[0]
    dft = np.fft.rfftn(M, s=(2 * T - 2,), axes=(0,))
    total = (dft.conjugate() * sigmas**2) @ dft.swapaxes(1, 2)
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]


def log_likelihood(Y, Sigma, sigma_measurement=None):
    """Given second moments, compute log-likelihood of data Y.

    Parameters
    ----------
    Y       : array (Tobs*O)
                stacked data for O observables over Tobs periods
    Sigma   : array (T*O*O)
                covariance between observables in model for 0, ... , T lags (e.g. from all_covariances)
    sigma_measurement : [optional] array (O)
                            std of measurement error for each observable, assumed zero if not provided

    Returns
    ----------
    L : scalar, log-likelihood
    """
    Tobs, nO = Y.shape
    if sigma_measurement is None:
        sigma_measurement = np.zeros(nO)
    V = build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)
    y = Y.ravel()
    return log_likelihood_formula(y, V)


'''Part 2: helper functions'''


def log_likelihood_formula(y, V):
    """Implements multivariate normal log-likelihood formula using Cholesky with data vector y and variance V.
       Calculates -log det(V)/2 - y'V^(-1)y/2
    """
    V_factored = linalg.cho_factor(V)
    quadratic_form = np.dot(y, linalg.cho_solve(V_factored, y))
    log_determinant = 2*np.sum(np.log(np.diag(V_factored[0])))
    return -(log_determinant + quadratic_form) / 2


@njit
def build_full_covariance_matrix(Sigma, sigma_measurement, Tobs):
    """Takes in T*O*O array Sigma with covariances at each lag t,
    assembles them into (Tobs*O)*(Tobs*O) matrix of covariances, including measurement errors.
    """
    T, O, O = Sigma.shape
    V = np.empty((Tobs, O, Tobs, O))
    for t1 in range(Tobs):
        for t2 in range(Tobs):
            if abs(t1-t2) >= T:
                V[t1, :, t2, :] = np.zeros((O, O))
            else:
                if t1 < t2:
                    V[t1, : , t2, :] = Sigma[t2-t1, :, :]
                elif t1 > t2:
                    V[t1, : , t2, :] = Sigma[t1-t2, :, :].T
                else:
                    # want exactly symmetric
                    V[t1, :, t2, :] = (np.diag(sigma_measurement**2) + (Sigma[0, :, :]+Sigma[0, :, :].T)/2)
    return V.reshape((Tobs*O, Tobs*O))
