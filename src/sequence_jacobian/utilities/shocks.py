from numba import njit, prange
import numpy as np

from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
from ..classes.result_dict import ResultDict


class Shock:
    def simulate_impulse(self, T: int):
        return NotImplementedError("Only ARMA shocks are supported.")


class ARMA(Shock):
    """
    An ARMA(p,q) shock parameterized by it's polynomial ratio as well as the
    impulse.

    phi(p) * y = theta(q) * eps    where eps ~ N(0, sigma)
    """
    def __init__(self, phi: np.ndarray[float], theta: np.ndarray[float], sigma: Optional[float] = 1):
        self.phi = phi
        self.theta = theta
        self.sigma = sigma

        # get dimensions
        self.p = phi.size
        self.q = theta.size

        self.parameters = {"phi": phi, "theta": theta, "sigma": sigma}

    def simulate_impulse(self, T: int):
        return _simulate_impulse(self.phi, self.theta, self.sigma, T)

    # try using something like pandas.util._decorators.@cached_readonly    
    def prior(self):
        pass

    def reparameterize(self, new_params):
        for param, value in new_params.items():
            new_param = _alloc_ndarray(value) if param != "sigma" else value
            setattr(self, param, new_param)


class AR(ARMA):
    """
    An AR(p,q) shock parameterized by it's autoregressive polynomial as well as
    the impulse.

    phi(p) * y = eps    where eps ~ N(0, sigma)
    """
    def __init__(self, phi, sigma = 1.0):
        super().__init__(phi, np.array([]), sigma)
        self.parameters = {"phi": phi, "sigma": sigma}


class MA(ARMA):
    """
    An MA(q) shock parameterized by it's moving average polynomial as well as
    the impulse.

    y = theta(q) * eps    where eps ~ N(0, sigma)
    """
    def __init__(self, theta, sigma = 1.0):
        super().__init__(np.array([]), theta, sigma)
        self.parameters = {"theta": theta, "sigma": sigma}


# @njit
def _simulate_impulse(phi, theta, sigma, T: int):
    """
    Generates an impulse path for a given ARMA(p,q) process
    """
    x = np.empty((T,))

    p = phi.size
    q = theta.size
    
    for t in range(T):
        if t == 0:
            x[t] = sigma
        else:
            ar_sum = 0
            for i in range(min(p, t)):
                ar_sum += phi[i]*x[t-1-i]
            ma_term = 0
            if 0 < t <= q:
                ma_term = theta[t-1]
            x[t] = ar_sum - ma_term

    return x

# ensures that parameters are of the proper dimension
def _alloc_ndarray(poly):
    if isinstance(poly, Real):
        return np.array([poly])
    elif isinstance(poly, list):
        return np.array(poly)
    else:
        return poly


# TODO: generate impulses for parameters with alternative shocks
class ShockDict(ResultDict):
    def __init__(self, data):
        if isinstance(data, ShockDict):
            super().__init__(data)
        else:
            if not isinstance(data, dict):
                raise ValueError('ShockDicts are initialized with a `dict` of top-level shocks.')
            super().__init__(data)

        # this is not a great solution either...
        self.parameters = {k: v.parameters for k,v in self.toplevel.items()}

    def generate_impulses(self, T: int):
        impulses = {}
        for k, v in self.items():
            if isinstance(v, Shock):
                impulses[k] = v.simulate_impulse(T)
            else:
                raise NotImplementedError('Multi-scenario shocks not yet supported.')
        
        return impulses
    
    def reparameterize(self, parameters: dict[str: dict]):
        # TODO: this does not change the parameters attr at the "top level"
        for k, v in self.toplevel.items():
            v.reparameterize(parameters[k])
            v.parameters = parameters[k]

## DATA GENERATING PROCESS TOOLS ##############################################

def simulate(impulses, outputs, T_sim):
    """
    impulses: list of ImpulseDicts, each an impulse to independent unit normal shock
    outputs: list of outputs we want in simulation
    T_sim: length of simulation

    simulation: dict mapping each output to length-T_sim simulated series
    """

    simulation = {}
    epsilons = [np.random.randn(T_sim+impulses[0].T-1) for _ in impulses]
    for o in outputs:
        simulation[o] = sum(
            simul_shock(imp[o], eps) for imp, eps in zip(impulses, epsilons)
        )
        
    return simulation


@njit(parallel=True)
def simul_shock(impulse, epsilons):
    """
    Take in any impulse response dX to epsilon shock, plus path of epsilons, and simulate
    """

    T = len(impulse)
    T_eps = len(epsilons)
    impulse_tilde = np.empty(T_eps - T + 1) 
    
    impulse_flipped = impulse[::-1].copy()
    for t in prange(T_eps - T + 1):
        impulse_tilde[t] = np.vdot(impulse_flipped, epsilons[t:t + T])

    return impulse_tilde