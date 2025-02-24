"""Functions for calculating the log likelihood of a model from its impulse responses"""

import numpy as np
import scipy.linalg as linalg
from numba import njit
import warnings

from .blocks.combined_block import create_model

from .samplers import *
from .utilities.distributions import *
from .utilities.shocks import stacked_responses

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


def gaussian_log_likelihood(Y, Sigma, sigma_measurement=None):
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

def log_likelihood(data, shocks, jacobian, outputs, exogenous, T, **kwargs):
    """Given the Jacobian of a parameterized model, calculate the Gaussian likelihood for a provided sequence
       of shocks.
    """
    # construct impulse response functions
    impulses = shocks.generate_impulses(T)
    M = stacked_responses(impulses, jacobian, outputs)

    # approximate the autocovariances
    Sigma = all_covariances(M, 1)

    # compute the log posterior likelihood
    return gaussian_log_likelihood(data, Sigma, **kwargs)


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


class DensityModel:
    """
    Operation class for estimating a DSGE model in PyMC; specifically given a
    model object, its steady state, some data, and a likelihood function which
    can be customized to recompute a Jacobian only when necessary.
    """
    def __init__(
            self, data, steady_state, model, shock_func, unknowns, targets, exogenous, T, sigmas=None, **kwargs
        ):
        # save important model info
        self.steady_state = steady_state
        self.model = model
        self.assemble_shocks = shock_func
        self.T = T

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
            steady_state, unknowns, targets, exogenous, outputs, T=T, **kwargs
        )

        # this is certainly sloppy, but it works for the meantime
        self.precompute = True
        self.model_info = {
            "unknowns": unknowns,
            "targets": targets,
            "inputs": exogenous,
            "outputs": outputs
        }

        # add measurement covariance to improve likelihood estimation
        self.meas_cov = sigmas
    
    def construct_jacobian_model(self, inputs, **kwargs):
        param_names = set(inputs.keys())
        self.jac_model = self.reduce_model(param_names, T=self.T)
        self.precompute = False
        return None

    def reduce_model(self, param_names, T):
        model = self.model

        inputs   = model.make_ordered_set(self.model_info["inputs"])
        unknowns = model.make_ordered_set(self.model_info["unknowns"])
        targets  = model.make_ordered_set(self.model_info["targets"])
        outputs  = model.make_ordered_set(self.model_info["outputs"])

        actual_outputs, unknowns_as_outputs = model.process_outputs(
            self.steady_state, unknowns, outputs
        )

        ss = model.M.inv @ self.steady_state
        reqs = model._required
        vector_valued = ss._vector_valued()
        
        inputs  = (model.M.inv @ (inputs | unknowns) | reqs) - vector_valued
        outputs = (model.M.inv @ ((actual_outputs | targets) - unknowns) | reqs) - vector_valued
        
        new_blocks = []
        for block in model.blocks:
            if model.make_ordered_set(block.inputs) & set(param_names):
                new_blocks.append(block)
            else:
                new_blocks.append(
                    block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T=T)
                )

        return create_model(new_blocks, name="reduced Jacobian")
    
    # this is technically slower for estimation where the jacobian does not update
    def log_likelihood(self, params, **kwargs):
        if self.precompute:
            self.construct_jacobian_model(params, **kwargs)

        # must pass only model parameters to updated steady state
        return self._likelihood(
            dict((k, params[k]) for k in self.model.inputs if k in params),
            self.assemble_shocks(params)
        )

    def _likelihood(self, params, shocks, **kwargs):
        outputs, inputs = self.model_info["outputs"], self.model_info["inputs"]
        unknowns, targets = self.model_info["unknowns"], self.model_info["targets"]

        if not params:
            # only parameter changes are shocks
            jacobian = self.jacobian
        else:
            # we need a new jacobian calculation
            ss_new = self.steady_state.copy()
            ss_new.update(params)
            jacobian = self.jac_model.solve_jacobian(
                ss_new, unknowns, targets, inputs, outputs, T=self.T, **kwargs
            )

        return log_likelihood(
            self.data, shocks, jacobian, outputs, inputs,
            T=self.T, sigma_measurement=self.meas_cov
        )

    # TODO: this is messy and needs to be cleaned up quite a bit
    def back_out_shocks(self, params, preperiods=0, **kwargs):
        shocks = self.assemble_shocks(params)
        outputs, inputs = self.model_info["outputs"], self.model_info["inputs"]
        unknowns, targets = self.model_info["unknowns"], self.model_info["targets"]

        impulses = shocks.generate_impulses(self.T)
        ss_new = self.steady_state.copy()
        ss_new.update(
            dict((k, params[k]) for k in self.model.inputs if k in params)
        )

        y = self.data
        jacobian = self.jac_model.solve_jacobian(
            ss_new, unknowns, targets, inputs, outputs, T=self.T, **kwargs
        )

        M = stacked_responses(impulses, jacobian, outputs)
        Ty, Oy = y.shape
        Tm, O, E = M.shape
        Ty_with_pre = Ty + preperiods

        A_full = construct_stacked_A(M, To=Ty_with_pre, To_out=Ty, sigma=self.meas_cov)
        y = y.reshape(Ty*O)

        # Step 2: Solve OLS
        eps_hat = np.linalg.lstsq(A_full, y, rcond=None)[0]  # this is To*E x 1 dimensional array
        eps_hat = eps_hat.reshape((Ty_with_pre, E))

        # Step 3: Decompose data
        for _ in range(E):
            A_full = A_full.reshape((Ty, O, Ty_with_pre, E))
            Ds = np.sum(A_full*eps_hat, axis=2)

        # Cut away pre periods from eps_hat
        eps_hat = eps_hat[preperiods:,:]

        return eps_hat, Ds

# TODO: stream line this as well, the amount of reshaping is a little crazy
def construct_stacked_A(As, To, To_out=None, sigma=None, reshape=True):
    Tm, O, E = As.shape

    if To_out is None:
        To_out = To

    A_full = np.zeros((To_out, O, To, E))
    for o in range(O):
        for itshock in range(To):
            # if To > To_out, allow the first To - To_out shocks to happen before the To_out time periods
            if To <= To_out:
                iA_full = itshock
                iAs = 0

                shock_length = min(Tm, To_out - iA_full)
            else:
                # this would be the correct start time of the shock
                iA_full = itshock - (To - To_out)

                # since it can be negative, only start IRFs at later date
                iAs = - min(iA_full, 0)

                # correct iA_full by that date
                iA_full += - min(iA_full, 0)

                shock_length = min(Tm, To_out - iA_full)

            for e in range(E):
                # I removed the scaling by std dev since the new object technically takes care of that
                A_full[iA_full:iA_full + shock_length, o, itshock, e] = As[iAs:iAs + shock_length, o, e]

    if reshape:
        A_full = A_full.reshape((To_out * O, To * E))
    return A_full