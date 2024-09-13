"""Functions for calculating the log likelihood of a model from its impulse responses"""

import numpy as np
import scipy.linalg as linalg
from numba import njit
import warnings

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


try:
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph import Apply, Op

    # TODO: add dictionary handling into likelihood call
    # TODO: improve shock parameterization
    class DensityModel(Op):
        """
        Operation class for estimating a DSGE model in PyMC; specifically given a
        model object, its steady state, some data, and a likelihood function which
        can be customized to recompute a Jacobian only when necessary.
        """
        def __init__(
                self, data, steady_state, model, likelihood_func, unknowns, targets, exogenous, **kwargs
            ):
            # save important model info
            self.steady_state = steady_state
            self.model = model
            self.logpdf = likelihood_func

            # check that all series are of equal length
            T_data = [len(v) for v in data.values()]
            assert all(x == T_data[0] for x in T_data)

            # munge the data into a numpy array
            self.data = np.empty((T_data[0], len(data.keys())))
            for no, o in enumerate(data.keys()):
                self.data[:, no] = data[o]

            # record the initial Jacobian
            outputs = list(data.keys())
            self.jacobian = model.solve_jacobian(
                steady_state, unknowns, targets, exogenous, outputs, **kwargs
            )

        def make_node(self, *args) -> Apply:
            inputs = [pt.as_tensor(arg) for arg in args]
            outputs = [pt.dscalar()]

            return Apply(self, inputs, outputs)
        
        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            logposterior = self.logpdf(
                *inputs, self.data, self.steady_state, self.model, self.jacobian
            )

            outputs[0][0] = np.asarray(logposterior)

except ImportError:
    class DensityModel:
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "Attempted to call DensityModel when PyMC is not yet installed"
            )