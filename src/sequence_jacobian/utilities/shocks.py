from numba import njit
import numpy as np
# import numpy.typing as npt

from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List

class Shock:
    def simulate_impulse(self, T):
        return NotImplementedError


class ARMA(Shock):
    """
    An ARMA(p,q) shock parameterized by it's polynomial ratio as well as the
    impulse.

    
    """
    def __init__(self, phi: List[Real], theta: List[Real], sigma: Optional[Real] = 1.0):
        self.phi = phi
        self.theta = theta
        self.sigma = sigma

        # get dimensions
        self.p = phi.size
        self.q = theta.size

    def simulate_impulse(self, T: int):
        return _simulate_impulse(self.phi, self.theta, self.sigma, T)


class AR(ARMA):
    def __init__(self, phi, sigma=1.0):
        return super().__init__(phi, np.array([]), sigma)


class MA(ARMA):
    def __init__(self, theta, sigma=1.0):
        return super().__init__(np.array([]), theta, sigma)

@njit
def _simulate_impulse(phi, theta, sigma, T: int):
    """
    Generates shocks for a given ARMA(p,q) process
    """
    x = np.empty((T,))

    n_ar = phi.size
    n_ma = theta.size
    
    for t in range(T):
        if t == 0:
            x[t] = sigma
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += phi[i]*x[t-1-i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = theta[t-1]
            x[t] = ar_sum - ma_term

    return x