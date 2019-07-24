import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import rfft, rfftn, irfft, irfftn
from numba import njit

class AsymptoticTimeInvariant:
    """Represents the asymptotic behavior of infinite matrix that is asymptotically time invariant,
    given by vector v of -(tau-1), ... , 0, ..., tau-1 asymptotic column entries around main diagonal"""

    __array_priority__ = 2000

    def __init__(self, v):
        self.v = v

        # v should be -(tau-1), ... , 0, ..., tau-1 asymp column around main diagonal
        self.tau = (len(v)+1) // 2
        assert self.tau*2 - 1 == len(v), f'{len(v)}'

    # note: no longer doing caching of these because FFT so quick that complexity unnecessary
    @property
    def vfft(self):
        return rfft(self.v, 4*self.tau-1)

    @property
    def vfft_leftzero(self):
        zeroed = self.v.copy()
        zeroed[:self.tau+1] = 0
        return rfft(zeroed, 4*self.tau-1)

    @property
    def vfft_rightzero(self):
        zeroed = self.v.copy()
        zeroed[self.tau:] = 0
        return rfft(zeroed, 4*self.tau-1)

    def changetau(self, tau):
        """return new with lower or higher tau, trimming or padding with zeros as needed"""
        if tau == self.tau:
            return self
        elif tau < self.tau:
            return AsymptoticTimeInvariant(self.v[self.tau - tau: tau + self.tau - 1])
        else:
            v = np.zeros(2*tau-1)
            v[tau - self.tau: tau + self.tau - 1] = self.v
            return AsymptoticTimeInvariant(v)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.v[slice(i.start+self.tau-1, i.stop+self.tau-1, i.step)]
        else:
            return self.v[i+self.tau-1]

    @property
    def T(self):
        """Transpose"""
        return AsymptoticTimeInvariant(self.v[::-1])

    def __pos__(self):
        return self

    def __neg__(self):
        return AsymptoticTimeInvariant(-self.v)

    def __matmul__(self, other):
        if isinstance(other, AsymptoticTimeInvariant):
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)
            return AsymptoticTimeInvariant(irfft(newself.vfft*other.vfft, 4*newself.tau-1)[newself.tau:-newself.tau])
        elif hasattr(other, 'asymptotic_time_invariant'):
            return self @ other.asymptotic_time_invariant
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        return self @ other

    def __add__(self, other):
        if isinstance(other, AsymptoticTimeInvariant):
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)
            return AsymptoticTimeInvariant(newself.v + other.v)
        elif hasattr(other, 'asymptotic_vector'):
            return self + other.asymptotic_vector
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return AsymptoticTimeInvariant(a*self.v)

    def __rmul__(self, a):
        return self * a

    def __repr__(self):
        return f'AsymptoticTimeInvariant({self.v!r})'

    def __eq__(self, other):
        return np.array_equal(self.v, other.v) if isinstance(other, AsymptoticTimeInvariant) else False
    